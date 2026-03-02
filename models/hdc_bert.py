# models/hdc_bert.py
"""
HDC-BERT (Hierarchical Dynamic Computation) classifier.

将 BERT 的 L 层分为两个阶段:
  Stage A (layers 0 .. split_layer-1): DeeBERT 风格 early-exit
    - 每层有一个 off-ramp classifier，基于 entropy 决定是否提前退出
    - 不破坏 attention 计算图，保护基础表示

  Stage B (layers split_layer .. n_layers-1): Token-level routing
    - 每层有一个 TokenRouter，决定哪些 token 跳过 attention
    - 公式: y = M * F(x) + x (F=Attention, M=router mask)

Training: 3 阶段
  Stage 1: 标准 fine-tune backbone + final classifier (off-ramps/routers 冻结)
  Stage 2: 冻结 backbone, 只训练 off-ramp classifiers
  Stage 3: 冻结 backbone + off-ramps, 只训练 token routers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# ---- 从现有模块复用 ----
from .deebert import entropy_from_logits
from .router_tuning_bert import TokenRouter, ste_binarize


class HDCBERTClassifier(nn.Module):
    """
    HDC-BERT classifier.

    - forward():                  drop-in replacement，返回 logits (eval 兼容)
    - forward_all_offramps():     Stage 2 训练用，返回所有 off-ramp logits
    - forward_with_routing():     Stage 3 训练用，返回 (logits, router_stats, l_mod_total)
    - forward_hdc_inference():    完整 HDC 推理 (early-exit + token routing)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        split_layer: int = 6,           # [0..split_layer-1] = Stage A, [split_layer..n-1] = Stage B
        tau: float = 0.5,               # STE 二值化阈值
        target_keep_ratio: float = 0.7, # token routing 保留目标
    ):
        super().__init__()
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.n_layers = self.bert.config.num_hidden_layers
        hidden = self.bert.config.hidden_size
        self.split_layer = split_layer
        self.tau = tau
        self.target_keep_ratio = target_keep_ratio

        # Stage A: off-ramp classifiers (layers 0..split_layer-1)
        self.offramp_classifiers = nn.ModuleList(
            [nn.Linear(hidden, num_labels) for _ in range(split_layer)]
        )

        # Stage B: token routers (layers split_layer..n_layers-1)
        self.routers = nn.ModuleList(
            [TokenRouter(hidden) for _ in range(self.n_layers - split_layer)]
        )

        # Final classifier (on CLS of last layer)
        self.classifier = nn.Linear(hidden, num_labels)

    # ====================== Freeze Methods ======================

    def freeze_for_stage1(self):
        """Stage 1: 训练 backbone + final classifier，冻结 off-ramps + routers。"""
        for p in self.offramp_classifiers.parameters():
            p.requires_grad = False
        for p in self.routers.parameters():
            p.requires_grad = False

    def freeze_for_stage2(self):
        """Stage 2: 只训练 off-ramp classifiers，冻结其余所有。"""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False
        for p in self.routers.parameters():
            p.requires_grad = False
        for p in self.offramp_classifiers.parameters():
            p.requires_grad = True

    def freeze_for_stage3(self):
        """Stage 3: 只训练 token routers，冻结其余所有。"""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False
        for p in self.offramp_classifiers.parameters():
            p.requires_grad = False
        for p in self.routers.parameters():
            p.requires_grad = True

    # ====================== BERT Layer Decomposition ======================
    # 和 RouterTuningBERTClassifier 中完全相同的逻辑

    def _run_attention(self, layer_module, hidden_states, extended_mask):
        """
        Self-Attention + dense + dropout, 残差之前的输出。
        论文公式 (3) 中的 F(x)。
        """
        self_outputs = layer_module.attention.self(
            hidden_states,
            attention_mask=extended_mask,
            head_mask=None,
            output_attentions=False,
        )
        attn_out = layer_module.attention.output.dense(self_outputs[0])
        attn_out = layer_module.attention.output.dropout(attn_out)
        return attn_out  # [B, L, H]

    def _run_post_attention(self, layer_module, hidden_states, attn_out):
        """
        Residual + LayerNorm + FFN sublayer (无条件执行)。
        """
        h = layer_module.attention.output.LayerNorm(attn_out + hidden_states)
        intermediate_output = layer_module.intermediate(h)
        layer_output = layer_module.output(intermediate_output, h)
        return layer_output  # [B, L, H]

    # ====================== Budget Loss ======================

    def _compute_budget_loss(self, mask, mask_hard, attention_mask):
        """
        Token-level budget loss.
        L_budget = ReLU(actual_kept - target_budget)
        mask:      STE 版本 (有梯度) → 算 l_mod
        mask_hard: 硬 0/1 (无梯度) → 算 keep_rate (日志)
        """
        token_mask = attention_mask.to(mask.dtype).unsqueeze(-1)  # [B, L, 1]
        m_sum_hard = (mask_hard * token_mask).sum()
        denom = token_mask.sum().clamp_min(1.0)
        m_sum = (mask * token_mask).sum()
        keep_rate = (m_sum_hard / denom).item()
        l_mod = F.relu(m_sum - self.target_keep_ratio * denom)
        return keep_rate, l_mod

    # ====================== Inference: Token Skip ======================

    def _token_attn_skip_inference(self, layer_module, hidden_states, attention_mask, mask_hard):
        """
        推理时 token-level: 只对 kept tokens 算 attention，skipped tokens attention=0。
        Kept tokens 之间互相 attend (MoD 风格)。
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

    # ====================== Forward Methods ======================

    def forward(self, input_ids, attention_mask):
        """
        Drop-in replacement: 直接用 self.bert() 前向，返回 logits。
        用于 Stage 1 训练和标准评估。
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0]  # [B, H]
        cls = self.dropout(cls)
        logits = self.classifier(cls)          # [B, num_labels]
        return logits

    def forward_all_offramps(self, input_ids, attention_mask):
        """
        Stage 2 训练用: 跑完所有层，收集 Stage A 各层的 off-ramp logits。

        Returns:
            list of split_layer 个 Tensor，每个 [B, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        # hidden_states[0] = embedding output
        # hidden_states[i+1] = output of layer i

        offramp_logits_list = []
        for i in range(self.split_layer):
            cls = hidden_states[i + 1][:, 0]          # [B, H]
            cls = self.dropout(cls)
            logits = self.offramp_classifiers[i](cls)  # [B, num_labels]
            offramp_logits_list.append(logits)

        return offramp_logits_list

    def forward_with_routing(self, input_ids, attention_mask):
        """
        Stage 3 训练用: 手动逐层前向。
        - Stage A layers: 正常计算 (无 routing)
        - Stage B layers: router-gated attention

        Returns:
            logits:       [B, num_labels]
            router_stats: {"keep_rates": list}
            l_mod_total:  scalar budget loss
        """
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, token_type_ids=None,
        )
        extended_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )

        hidden_states = embedding_output
        keep_rates = []
        l_mod_total = hidden_states.new_tensor(0.0)

        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i < self.split_layer:
                # Stage A: 正常计算整层
                hidden_states = layer_module(hidden_states, extended_mask)[0]
            else:
                # Stage B: router-gated attention
                router_idx = i - self.split_layer
                router = self.routers[router_idx]
                prob = router(hidden_states)  # [B, L, 1]
                mask, mask_hard = ste_binarize(prob, self.tau)
                keep_rate, l_mod = self._compute_budget_loss(mask, mask_hard, attention_mask)

                if self.training:
                    attn_out = self._run_attention(layer_module, hidden_states, extended_mask)
                    attn_out = attn_out * mask  # STE gate: M * F(x)
                else:
                    attn_out = self._token_attn_skip_inference(
                        layer_module, hidden_states, attention_mask, mask_hard
                    )

                # Residual + LayerNorm + FFN (无条件执行)
                hidden_states = self._run_post_attention(layer_module, hidden_states, attn_out)
                keep_rates.append(keep_rate)
                l_mod_total = l_mod_total + l_mod

        # CLS pooling + classification
        cls = hidden_states[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        router_stats = {"keep_rates": keep_rates}
        return logits, router_stats, l_mod_total

    @torch.no_grad()
    def forward_hdc_inference(self, input_ids, attention_mask, entropy_threshold: float = 0.2):
        """
        完整 HDC 推理: Stage A early-exit + Stage B token routing.

        1. 逐层跑 Stage A，每层检查 off-ramp entropy
        2. 若 batch 全部置信 → 提前退出
        3. 否则进入 Stage B，逐层做 token-level routing

        Returns:
            logits:    [B, num_labels]
            exit_info: dict with 'exited_layer', 'stage', 'keep_rates'
        """
        self.eval()

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, token_type_ids=None,
        )
        extended_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )

        hidden_states = embedding_output
        keep_rates = []

        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i < self.split_layer:
                # Stage A: 跑完整层，然后检查 off-ramp
                hidden_states = layer_module(hidden_states, extended_mask)[0]

                cls = hidden_states[:, 0]
                cls_dropped = self.dropout(cls)
                offramp_logits = self.offramp_classifiers[i](cls_dropped)
                ent = entropy_from_logits(offramp_logits)  # [B]

                if torch.all(ent < entropy_threshold):
                    return offramp_logits, {
                        "exited_layer": i + 1,  # 1-indexed
                        "stage": "A",
                        "keep_rates": [],
                    }
            else:
                # Stage B: token-level routing (推理跳过模式)
                router_idx = i - self.split_layer
                router = self.routers[router_idx]
                prob = router(hidden_states)
                _, mask_hard = ste_binarize(prob, self.tau)

                # 记录 keep_rate (日志用)
                token_mask = attention_mask.to(mask_hard.dtype).unsqueeze(-1)
                m_sum = (mask_hard * token_mask).sum()
                denom = token_mask.sum().clamp_min(1.0)
                keep_rates.append((m_sum / denom).item())

                # 只对 kept tokens 算 attention
                attn_out = self._token_attn_skip_inference(
                    layer_module, hidden_states, attention_mask, mask_hard
                )
                hidden_states = self._run_post_attention(layer_module, hidden_states, attn_out)

        # 走完所有层，用 final classifier
        cls = hidden_states[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        return logits, {
            "exited_layer": "final",
            "stage": "B",
            "keep_rates": keep_rates,
        }
