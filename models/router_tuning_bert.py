'''
Router-Tuning BERT 模型（Mixture-of-Depth 风格）
论文核心思想：冻住 BERT backbone，只训练极轻量的 router 来决定每层 Attention 是否跳过

1) 复制 baseline 的骨架
   self.bert = AutoModel.from_pretrained(model_name) 保留
   self.classifier = nn.Linear(hidden_size, num_labels) 保留
   增加：self.routers = ModuleList([...]) 每层一个 router

2) Router 有两种粒度
   TokenRouter:  对每个 token 独立打分 → sigmoid(W x_i) → [B, L, 1]
   SampleRouter: 对整条序列打分（mean pool 后） → sigmoid(W mean(x)) → [B, 1]

3) 门控机制
   router 输出 prob → 与阈值 tau 比较 → 二值 mask（STE 让梯度能回传）
   y = mask * Attention(x) + x （mask=0 时跳过 Attention，直接残差）

4) Loss = L_task + lambda * L_MoD
   L_MoD = ReLU(实际保留量 - 目标保留量)，鼓励跳过更多层

5) forward() 返回 (logits, router_stats, l_mod)
   与 train.py 中 train_epoch_router_tuning() 配合使用
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ==================== 工具函数 ====================

def ste_binarize(prob, tau=0.5):
    """
    Straight-Through Estimator 二值化
    前向：硬阈值 prob >= tau → 0/1
    反向：梯度直接穿过，当作恒等函数

    Args:
        prob: 任意形状的概率值（经过 sigmoid）
        tau:  阈值，默认 0.5

    Returns:
        mask:      STE mask，前向是 0/1，反向有梯度
        mask_hard: 纯 0/1 硬 mask，用于统计和 penalty 计算
    """
    mask_hard = (prob >= tau).to(prob.dtype)                # 前向：硬阈值
    mask = mask_hard.detach() - prob.detach() + prob        # detach()保留mask_hard的数值，但阻断它的梯度
    return mask, mask_hard


class TokenRouter(nn.Module):
    """
    Token-level router: 对序列中每个 token 独立打分
    R(x)_i = sigmoid(W @ x_i)
    W 初始化为全零 → sigmoid(0) = 0.5 = tau → 训练初始所有层都不跳过
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states):
        # hidden_states: [B, L, H] 这个H是 BERT 的 hidden_size
        return torch.sigmoid(self.proj(hidden_states))       # [B, L, 1]


class SampleRouter(nn.Module):
    """
    Sample-level (Sequence-level) router: 对整条序列算一个统一分数
    R(x) = sigmoid(W @ mean_pool(x))
    整条序列要么全走这一层，要么全跳
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H]
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1) # 把padding的token也算进去平均（不合理）
        else:
            # 用 attention_mask 做 masked mean pooling，忽略 padding token，attention_mask: [B, L]
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, L, 1], 为了hidden_states * mask广播
            denom = mask.sum(dim=1).clamp_min(1.0)                      # [B, 1]
            pooled = (hidden_states * mask).sum(dim=1) / denom           # [B, H]
        return torch.sigmoid(self.proj(pooled))              # [B, 1]


class NoRouter(nn.Module):
    """不做 routing 的占位符，用于非 routed 层"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None


class RouterTuningBERTClassifier(nn.Module):
    """
    Router-Tuning BERT 分类器
    - 只对 Attention 子层做门控，MLP 仍然全量执行
    - forward() 返回 (logits, router_stats, l_mod)
    - 支持 token-level 和 sample-level 两种 routing 模式
    """

    def __init__(self, model_name, num_labels, dropout=0.1,
                 routing_mode="token", tau=0.5,
                 target_keep_ratio=0.7, routed_layers=None):
        """
        Args:
            model_name:        预训练模型名，如 'bert-base-uncased'
            num_labels:        分类类别数（AG News = 4）
            dropout:           dropout 概率
            routing_mode:      'token' 或 'sample'
            tau:               二值化阈值，默认 0.5（配合全零初始化）
            target_keep_ratio: 目标保留比例，L_MoD 会惩罚超过这个比例的层
            routed_layers:     哪些层做 routing，None 表示所有层
        """
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

        # 为每一层创建对应的 router（不做 routing 的层用 NoRouter 占位）
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

    def freeze_backbone(self):
        """冻住已经微调过的 BERT backbone 和 classifier，只留 router 可训练"""
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    def _gate_attention(self, layer_module, hidden_states,
                        attention_mask, raw_attention_mask, router):
        """
        对单层的 Attention 输出做门控，MLP 照常执行

        流程：
            1. 正常跑 Self-Attention → attn_out
            2. 如果该层有 router：
               - router 算出 prob → STE 二值化得 mask
               - attn_out = attn_out * mask （mask=0 → 跳过 attention）
               - 计算 keep_rate 和 L_MoD penalty
            3. 残差 + LayerNorm → MLP → 该层输出

        Args:
            layer_module:        BERT 的一个 Transformer 层
            hidden_states:       该层输入 [B, L, H]
            attention_mask:      扩展后的 attention mask（给 self-attention 用）
            raw_attention_mask:  原始 0/1 mask [B, L]（给 router 和统计用）
            router:              该层的 router 模块

        Returns:
            layer_output: 该层输出 [B, L, H]
            keep_rate:    该层实际保留比例（float），非 routed 层返回 None
            l_mod:        该层的 MoD penalty（标量 tensor）
        """
        # ---- Step 1: Self-Attention ----
        attn_outputs = layer_module.attention.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
        )
        context_layer = attn_outputs[0]                       # [B, L, H]

        # attention output: dense + dropout（还没做 LayerNorm 和残差）
        attn_out = layer_module.attention.output.dense(context_layer)   # [B, L, H]
        attn_out = layer_module.attention.output.dropout(attn_out)      # [B, L, H]

        # ---- Step 2: Router 门控 ----
        keep_rate = None
        l_mod = hidden_states.new_tensor(0.0)

        if not isinstance(router, NoRouter):
            if self.routing_mode == "token":
                # 每个 token 独立决策
                prob = router(hidden_states)                              # [B, L, 1]
                mask, mask_hard = ste_binarize(prob, self.tau)            # [B, L, 1]
                attn_out = attn_out * mask                                # mask=0 的 token 跳过 attention

                # 统计：只算非 padding token 的保留比例
                token_mask = raw_attention_mask.to(attn_out.dtype).unsqueeze(-1)  # [B, L, 1]
                m_sum = (mask_hard * token_mask).sum()                    # 实际保留的 token 数
                denom = token_mask.sum().clamp_min(1.0)                   # 总有效 token 数
                keep_rate = (m_sum / denom).item()
                budget = self.target_keep_ratio * denom                   # 目标保留量
                l_mod = F.relu(m_sum - budget)                            # 超过预算才惩罚

            elif self.routing_mode == "sample":
                # 整条序列统一决策
                prob_sample = router(hidden_states, raw_attention_mask)    # [B, 1]
                mask_sample, mask_hard_sample = ste_binarize(prob_sample, self.tau)  # [B, 1]
                mask = mask_sample.unsqueeze(1)                           # [B, 1, 1] → broadcast 到 [B, L, H]
                attn_out = attn_out * mask                                # mask=0 的样本整条跳过

                # 统计：以样本为单位
                m_sum = mask_hard_sample.sum()                            # 保留的样本数
                denom = mask_hard_sample.new_tensor(mask_hard_sample.size(0)).clamp_min(1.0)
                keep_rate = (m_sum / denom).item()
                budget = self.target_keep_ratio * denom
                l_mod = F.relu(m_sum - budget)

        # ---- Step 3: 残差 + LayerNorm → MLP ----
        attn_out = layer_module.attention.output.LayerNorm(attn_out + hidden_states)  # [B, L, H]
        intermediate_output = layer_module.intermediate(attn_out)                      # [B, L, intermediate_size]
        layer_output = layer_module.output(intermediate_output, attn_out)              # [B, L, H]

        return layer_output, keep_rate, l_mod

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]

        Returns:
            logits:       [B, num_labels]  分类 logits
            router_stats: dict，包含每层的 keep_rate 和 routed_layers 列表
            l_mod_total:  标量 tensor，所有 routed 层的 MoD penalty 之和
        """
        device = input_ids.device
        input_shape = input_ids.size()

        # 这个embedding_output 是微调后的BERT
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=None,
        )  # [B, L, H]

        # 扩展 attention mask 给 self-attention 用（加了因果 mask 和维度扩展）
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_shape, device
        )  # [B, 1, 1, L]

        # ---- 逐层过 Transformer + Router 门控 ----
        hidden_states = embedding_output
        keep_rates = [None for _ in range(self.n_layers)]
        l_mod_total = hidden_states.new_tensor(0.0)

        for i, layer_module in enumerate(self.bert.encoder.layer):
            router = self.routers[i]
            hidden_states, keep_rate, l_mod = self._gate_attention(
                layer_module,
                hidden_states,
                extended_attention_mask,
                attention_mask,       # 原始 0/1 mask，给 router 用
                router,
            )
            if keep_rate is not None:
                keep_rates[i] = keep_rate
                l_mod_total = l_mod_total + l_mod

        # ---- 取 [CLS] → 分类 ----
        pooled_output = hidden_states[:, 0]                   # [B, H]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)               # [B, num_labels]

        router_stats = {
            "keep_rates": keep_rates,
            "routed_layers": sorted(self.routed_layers),
        }
        return logits, router_stats, l_mod_total
