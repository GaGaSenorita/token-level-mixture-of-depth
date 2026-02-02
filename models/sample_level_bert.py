# src/models/bert_sample_level.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, BertModel


@dataclass
class SampleLevelOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor                 # [B, num_labels]
    extra: Dict[str, Any]                # exit_layer, exit_conf, exit_hist, all_logits(optional)


class BertSampleLevelForSequenceClassification(nn.Module):
    """
    Sample-level early-exit BERT:
    - multiple exit heads at selected layers
    - confidence-based stopping per sample
    """

    def __init__(
        self,
        pretrained_name: str,
        num_labels: int,
        exit_layers: List[int] = [4, 6, 8, 10, 12],   # 1-based layer index for readability
        exit_threshold: float = 0.9,
        dropout: float = 0.1,
        aux_ce_weight: float = 0.0,                  # optional: supervise intermediate exits
        distill_weight: float = 0.0,                 # optional: KL shallow->final
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.exit_layers = exit_layers
        self.exit_threshold = exit_threshold
        self.aux_ce_weight = aux_ce_weight
        self.distill_weight = distill_weight
        self.temperature = temperature

        cfg = AutoConfig.from_pretrained(pretrained_name)
        cfg.output_hidden_states = True  # IMPORTANT: we need all layers
        self.bert = BertModel.from_pretrained(pretrained_name, config=cfg)

        hidden = cfg.hidden_size
        self.drop = nn.Dropout(dropout)

        # one head per exit layer
        # map: layer_index(1..12) -> head
        self.exit_heads = nn.ModuleDict({
            str(L): nn.Linear(hidden, num_labels) for L in exit_layers
        })

    def _cls_logits(self, hidden_states: torch.Tensor, head: nn.Module) -> torch.Tensor:
        # hidden_states: [B, T, H]
        cls = hidden_states[:, 0, :]      # [CLS] token
        cls = self.drop(cls)
        return head(cls)                  # [B, num_labels]

    @torch.no_grad()
    def _confidence(self, logits: torch.Tensor) -> torch.Tensor:
        # confidence: max softmax prob, shape [B]
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_all_logits: bool = False,
    ) -> SampleLevelOutput:
        """
        Returns:
          logits: final chosen logits per sample (maybe from early exit)
          extra:
            - exit_layer: [B] (int) chosen layer
            - exit_conf:  [B] confidence at exit
            - exit_hist:  dict layer->count
            - all_logits: dict layer->logits (optional)
        """
        device = input_ids.device
        B = input_ids.size(0)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # hidden_states: tuple length = (embeddings + 12 layers) for BERT-base
        # hidden_states[i]: [B, T, H]
        hidden_states = outputs.hidden_states

        done = torch.zeros(B, dtype=torch.bool, device=device)  # which samples already exited
        chosen_logits = torch.zeros(B, self.num_labels, device=device)
        exit_layer = torch.full((B,), -1, dtype=torch.long, device=device)
        exit_conf = torch.zeros(B, device=device)

        all_logits: Dict[int, torch.Tensor] = {}
        exit_hist: Dict[int, int] = {L: 0 for L in self.exit_layers}

        # iterate exits in increasing depth
        for L in self.exit_layers:
            # HF indexing: hidden_states[0] is embedding output, hidden_states[1] is layer1 ...
            h = hidden_states[L]  # because L is 1-based layer index
            logits_L = self._cls_logits(h, self.exit_heads[str(L)])  # [B, C]

            if return_all_logits:
                all_logits[L] = logits_L

            # compute confidence and decide who exits here
            conf_L = self._confidence(logits_L)  # [B]
            should_exit = (conf_L >= self.exit_threshold) & (~done)

            if should_exit.any():
                idx = torch.where(should_exit)[0]
                chosen_logits[idx] = logits_L[idx]
                exit_layer[idx] = L
                exit_conf[idx] = conf_L[idx]
                done[idx] = True
                exit_hist[L] += int(idx.numel())

            # if all exited, break early
            if done.all():
                break

        # for any not exited, force them to use the last exit layer logits
        if (~done).any():
            L_last = self.exit_layers[-1]
            if return_all_logits and (L_last in all_logits):
                logits_last = all_logits[L_last]
            else:
                h_last = hidden_states[L_last]
                logits_last = self._cls_logits(h_last, self.exit_heads[str(L_last)])

            idx = torch.where(~done)[0]
            chosen_logits[idx] = logits_last[idx]
            exit_layer[idx] = L_last
            # note: confidence computed for completeness
            exit_conf[idx] = self._confidence(logits_last[idx])
            exit_hist[L_last] += int(idx.numel())
            done[idx] = True

        loss = None
        if labels is not None:
            # main loss on chosen logits
            loss_main = F.cross_entropy(chosen_logits, labels)

            loss_aux = torch.tensor(0.0, device=device)
            # optional: supervise intermediate exits too (simple multi-exit training)
            if self.aux_ce_weight > 0.0:
                for L in self.exit_layers:
                    # use stored logits if available, else recompute cheaply (still ok)
                    if return_all_logits and (L in all_logits):
                        logits_L = all_logits[L]
                    else:
                        logits_L = self._cls_logits(hidden_states[L], self.exit_heads[str(L)])
                    loss_aux = loss_aux + F.cross_entropy(logits_L, labels)
                loss_aux = loss_aux / len(self.exit_layers)

            loss_distill = torch.tensor(0.0, device=device)
            # optional: self-distill shallow -> final (teacher = last exit)
            if self.distill_weight > 0.0 and len(self.exit_layers) >= 2:
                L_teacher = self.exit_layers[-1]
                teacher = all_logits.get(L_teacher) if return_all_logits else None
                if teacher is None:
                    teacher = self._cls_logits(hidden_states[L_teacher], self.exit_heads[str(L_teacher)])
                teacher_prob = F.softmax(teacher / self.temperature, dim=-1).detach()

                for L in self.exit_layers[:-1]:
                    student = all_logits.get(L) if return_all_logits else None
                    if student is None:
                        student = self._cls_logits(hidden_states[L], self.exit_heads[str(L)])
                    student_logprob = F.log_softmax(student / self.temperature, dim=-1)
                    loss_distill = loss_distill + F.kl_div(
                        student_logprob, teacher_prob, reduction="batchmean"
                    ) * (self.temperature ** 2)
                loss_distill = loss_distill / (len(self.exit_layers) - 1)

            loss = loss_main + self.aux_ce_weight * loss_aux + self.distill_weight * loss_distill

        extra = {
            "exit_layer": exit_layer,   # [B]
            "exit_conf": exit_conf,     # [B]
            "exit_hist": exit_hist,     # dict
        }
        if return_all_logits:
            extra["all_logits"] = {k: v for k, v in all_logits.items()}

        return SampleLevelOutput(loss=loss, logits=chosen_logits, extra=extra)
