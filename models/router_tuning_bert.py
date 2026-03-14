# models/router_tuning_bert.py
"""
Router Tuning BERT classifier (ACT for base Transformer).

论文公式 (3): y = M ⊙ F(x) + x
- F(x) = Attention sublayer 的输出 (dense + dropout, 残差之前)
- M = router 的二值 mask
- + x = 残差连接 (无条件执行)
- 之后 LayerNorm + FFN sublayer 也无条件执行

即: router 只决定 attention 要不要算，residual + LayerNorm + FFN 始终跑。
M=0 时: y = 0 + x → LayerNorm(x) → FFN (attention 被跳过)
M=1 时: y = Attn(x) + x → LayerNorm → FFN (正常计算)

Training:  所有 token 都算 attention，用 STE mask gate 输出 → 梯度可回传
Inference: 只对 kept tokens 算 attention，skipped tokens attention=0 → 省 FLOPs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ====================== Utility ======================

def ste_binarize(prob: torch.Tensor, tau: float = 0.5):
    """
    router需要输出0/1的硬决策，但阀值操作梯度为0，无法反向传播训练
    Backforward的时候，Pytorch autograd只看计算图里哪些变量带requires_grad:
    mask = mask_hard.detach() - prob.detach() + prob 
    mask_hard.detach() 和 prob.detach() 都是常数，autograd 忽略它们。所以反向时：
    ∂mask/∂prob = ∂(常数 + prob)/∂prob = 1，相当于直接无视这个梯度
    """
    mask_hard = (prob >= tau).to(prob.dtype)
    mask = mask_hard.detach() - prob.detach() + prob
    return mask, mask_hard


# ====================== Routers ======================

class TokenRouter(nn.Module):
    """Token-level router: sigmoid(W · x_i) -> [B, L, 1]"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)   # sigmoid(0)=0.5, 初始时约一半 token 被保留
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(hidden_states))  # [B, L, 1]


class SampleRouter(nn.Module):
    """Sequence-level router: sigmoid(W · mean(x)) -> [B, 1]"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.sigmoid(self.proj(pooled))  # [B, 1]


class NoRouter(nn.Module):
    """Placeholder: 该层不做路由，正常计算。"""

    def forward(self, *args, **kwargs):
        return None


# ====================== Model ======================

class RouterTuningBERTClassifier(nn.Module):
    """
    Router-Tuning BERT classifier.

    - forward():                drop-in replacement，返回 logits（eval 兼容）
    - forward_with_routing():   返回 (logits, router_stats, l_mod_total)（训练用）
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        routing_mode: str = "token",    # "token" or "sample"
        tau: float = 0.5,
        target_keep_ratio: float = 0.7,
        routed_layers=None,             # None = 所有层都路由
    ):
        super().__init__()
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.n_layers = self.bert.config.num_hidden_layers
        hidden = self.bert.config.hidden_size
        self.routing_mode = routing_mode
        self.tau = tau
        self.target_keep_ratio = target_keep_ratio

        if routed_layers is None:
            routed_layers = list(range(self.n_layers))
        self.routed_layers = set(routed_layers)

        routers = []
        for i in range(self.n_layers):
            if i in self.routed_layers:
                if routing_mode == "token":
                    routers.append(TokenRouter(hidden))
                elif routing_mode == "sample":
                    routers.append(SampleRouter(hidden))
                else:
                    raise ValueError(f"Unknown routing_mode: {routing_mode}")
            else:
                routers.append(NoRouter())
        self.routers = nn.ModuleList(routers)

    # ---- Freeze ----

    def freeze_backbone(self):
        """Freeze BERT backbone + classifier, only train routers."""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    # ---- BERT layer decomposition ----
    # BERT Post-LN 结构:
    #   attn_out = dense(dropout(SelfAttention(x)))   ← 这部分被 router gate
    #   h = LayerNorm(attn_out + x)                   ← residual + LN, 无条件执行
    #   ffn_out = FFN(h)                               ← 无条件执行
    #   out = LayerNorm(ffn_out + h)                   ← 无条件执行

    def _run_attention(self, layer_module, hidden_states, extended_mask):
        """
        Self-Attention + dense + dropout, 残差之前的输出。
        这是论文公式 (3) 中的 F(x)。
        """
        self_outputs = layer_module.attention.self(
            hidden_states,
            attention_mask=extended_mask,
            head_mask=None,
            output_attentions=False,
        ) # self_outputs[0]: attn output [B, L, H]
        attn_out = layer_module.attention.output.dense(self_outputs[0])
        attn_out = layer_module.attention.output.dropout(attn_out)
        return attn_out  # [B, L, H]

    def _run_post_attention(self, layer_module, hidden_states, attn_out):
        """
        hidden_states: 输入到该层的 x
        attn_out: attention sublayer 的输出 (dense + dropout, 残差之前)
        Residual + LayerNorm + FFN sublayer (无条件执行)。
        对应 BERT 的: LN(attn_out + x) → FFN → LN(ffn_out + h)
        """
        # Attention residual + LayerNorm
        h = layer_module.attention.output.LayerNorm(attn_out + hidden_states)
        # FFN sublayer
        intermediate_output = layer_module.intermediate(h)
        layer_output = layer_module.output(intermediate_output, h)
        return layer_output  # [B, L, H]

    # ---- Budget loss ----

    def _compute_budget_loss(self, mask, mask_hard, attention_mask):
        """
        Compute keep_rate (for logging) and budget penalty (for loss).
        L_budget = ReLU(actual_kept - target_budget)

        mask:      STE 版本，有梯度 → 用来算 l_mod（参与反向传播）
        mask_hard: 硬 0/1，无梯度 → 用来算 keep_rate（仅日志记录）
        """
        if self.routing_mode == "token":
            token_mask = attention_mask.to(mask.dtype).unsqueeze(-1)  # [B, L] -> [B, L, 1]
            # keep_rate: 用 mask_hard 算真实保留比例（日志用）
            m_sum_hard = (mask_hard * token_mask).sum()
            denom = token_mask.sum().clamp_min(1.0)
            # l_mod: 用 mask (STE) 算，梯度可以回传到 router
            m_sum = (mask * token_mask).sum()
        else:  # sample
            m_sum_hard = mask_hard.sum()
            denom = mask_hard.new_tensor(mask_hard.size(0)).clamp_min(1.0)
            m_sum = mask.sum()

        keep_rate = (m_sum_hard / denom).item()
        l_mod = F.relu(m_sum - self.target_keep_ratio * denom)
        return keep_rate, l_mod

    # ---- Inference: attention-only skipping ----

    def _token_attn_skip_inference(self, layer_module, hidden_states, attention_mask, mask_hard):
        """
        推理时 token-level: 只对 kept tokens 算 attention，skipped tokens attention=0
        Kept tokens 之间互相 attend (MoD 风格)。
        返回: gated attention output [B, L, H] (残差之前)
        """
        keep_bool = (mask_hard.squeeze(-1) > 0) & (attention_mask > 0)  # [B, L]
        attn_out = torch.zeros_like(hidden_states)

        for b in range(hidden_states.size(0)):
            keep_idx = torch.nonzero(keep_bool[b], as_tuple=False).squeeze(-1)
            if keep_idx.numel() == 0:
                continue
            sub_hidden = hidden_states[b : b + 1, keep_idx, :]  # [1, K, H]
            sub_mask = torch.ones(
                1, keep_idx.numel(),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            sub_ext_mask = self.bert.get_extended_attention_mask(
                sub_mask, sub_mask.size(), sub_hidden.device
            )
            sub_attn_out = self._run_attention(layer_module, sub_hidden, sub_ext_mask)
            attn_out[b, keep_idx, :] = sub_attn_out[0]

        return attn_out

    def _sample_attn_skip_inference(self, layer_module, hidden_states, extended_mask, mask_hard):
        """
        推理时 sample-level: 只对 kept samples 算 attention，skipped samples attention=0。
        返回: gated attention output [B, L, H] (残差之前)
        """
        keep_sample = mask_hard.squeeze(-1) > 0  # [B]
        attn_out = torch.zeros_like(hidden_states)

        if keep_sample.any():
            keep_idx = torch.nonzero(keep_sample, as_tuple=False).squeeze(-1)
            kept_attn_out = self._run_attention(
                layer_module, hidden_states[keep_idx], extended_mask[keep_idx]
            )
            attn_out[keep_idx] = kept_attn_out

        return attn_out

    # ---- Core: one routed layer ----

    def _routed_layer(self, layer_module, hidden_states, extended_mask, attention_mask, router):
        """
        One Transformer layer with router-gated attention.

        论文公式 (3): y = M ⊙ F(x) + x
        - F(x) = attention sublayer (self-attn + dense + dropout)  ← 被 gate
        - + x = residual                                           ← 无条件
        - → LayerNorm → FFN sublayer                               ← 无条件

        Returns: (hidden_states, keep_rate, l_mod)
        """
        # 1. Router decision
        if self.routing_mode == "token":
            prob = router(hidden_states)                  # [B, L, 1]
        else:
            prob = router(hidden_states, attention_mask)   # [B, 1]

        mask, mask_hard = ste_binarize(prob, self.tau)

        # 始终保留 CLS token（position 0），确保分类 head 的输入不被跳过
        if self.routing_mode == "token":
            mask = mask.clone()
            mask_hard = mask_hard.clone()
            mask[:, 0, :] = 1.0
            mask_hard[:, 0, :] = 1.0

        keep_rate, l_mod = self._compute_budget_loss(mask, mask_hard, attention_mask)

        # 2. Gated attention: attn_out = M ⊙ F(x)
        if self.training:
            attn_out = self._run_attention(layer_module, hidden_states, extended_mask)
            gate = mask.unsqueeze(1) if self.routing_mode == "sample" else mask
            attn_out = attn_out * gate  # STE gate
        else:
            if self.routing_mode == "token":
                attn_out = self._token_attn_skip_inference(
                    layer_module, hidden_states, attention_mask, mask_hard
                )
            else:
                attn_out = self._sample_attn_skip_inference(
                    layer_module, hidden_states, extended_mask, mask_hard
                )

        # 3. Residual + LayerNorm + FFN (无条件执行)
        hidden_states = self._run_post_attention(layer_module, hidden_states, attn_out)

        return hidden_states, keep_rate, l_mod

    # ---- Forward ----

    def forward_with_routing(self, input_ids, attention_mask):
        """
        训练用: 返回 logits + routing 统计 + budget loss。

        Returns:
            logits:       [B, num_labels]
            router_stats: {"keep_rates": list, "routed_layers": list}
            l_mod_total:  scalar budget loss
        """
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, token_type_ids=None,
        )
        extended_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )

        hidden_states = embedding_output
        keep_rates = [None] * self.n_layers
        l_mod_total = hidden_states.new_tensor(0.0)

        for i, layer_module in enumerate(self.bert.encoder.layer):
            router = self.routers[i]

            if isinstance(router, NoRouter):
                # 非路由层: 正常计算整层
                hidden_states = layer_module(hidden_states, extended_mask)[0]
                continue

            hidden_states, keep_rate, l_mod = self._routed_layer(
                layer_module, hidden_states, extended_mask, attention_mask, router
            )
            keep_rates[i] = keep_rate
            l_mod_total = l_mod_total + l_mod

        # CLS pooling + classification
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        router_stats = {
            "keep_rates": keep_rates,
            "routed_layers": sorted(self.routed_layers),
        }
        return logits, router_stats, l_mod_total

    def forward(self, input_ids, attention_mask):
        """
        Drop-in replacement: 只返回 logits，兼容 baseline 评估流程。
        训练时请用 forward_with_routing() 获取 router_stats 和 l_mod。
        """
        logits, _, _ = self.forward_with_routing(input_ids, attention_mask)
        return logits
