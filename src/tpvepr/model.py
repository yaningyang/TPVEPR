from __future__ import annotations

import torch
from torch import nn


class RelationAwareSelfAttention(nn.Module):
    """Multi-head self-attention with pairwise promoter/allele relation bias."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        relation_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)
        self.relation_weight = nn.Parameter(torch.empty(num_heads, relation_dim))
        self.relation_scale = nn.Parameter(torch.zeros(num_heads))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weight)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, group_size, d_model = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size,
            group_size,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        relation_bias = torch.einsum(
            "bijd,hd->bhij",
            torch.tanh(relations),
            self.relation_weight,
        )
        logits = logits + relation_bias * torch.nn.functional.softplus(
            self.relation_scale
        ).view(1, -1, 1, 1)
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)

        attention = self.dropout(torch.softmax(logits, dim=-1))
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch_size, group_size, d_model)
        output = self.output(output) * mask.unsqueeze(-1)
        return output, attention


class RelationAwareTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        relation_dim: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = RelationAwareSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            relation_dim=relation_dim,
            dropout=dropout,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(
            self.norm_attention(tokens),
            mask,
            relations,
        )
        tokens = (tokens + attended) * mask.unsqueeze(-1)
        tokens = (tokens + self.ffn(self.norm_ffn(tokens))) * mask.unsqueeze(-1)
        return tokens, attention


class TPVEPR(nn.Module):
    """Permutation-equivariant ranker for promoter variants in tissue-gene sets."""

    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: list[int],
        relation_numeric_indices: list[int],
        prior_numeric_indices: list[int],
        embedding_dim: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.relation_numeric_indices = relation_numeric_indices
        self.prior_numeric_indices = prior_numeric_indices
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim, padding_idx=0)
                for cardinality in categorical_cardinalities
            ]
        )

        candidate_input_dim = num_numeric * 3 + embedding_dim * len(categorical_cardinalities)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        relation_dim = len(relation_numeric_indices) * 3 * 2
        self.blocks = nn.ModuleList(
            [
                RelationAwareTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    relation_dim=relation_dim,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        prior_dim = len(prior_numeric_indices) * 3
        self.prior_head = nn.Sequential(
            nn.Linear(prior_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model + prior_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def _select_feature_triplet(
        self,
        numeric: torch.Tensor,
        relative: torch.Tensor,
        indices: list[int],
    ) -> torch.Tensor:
        feature_count = numeric.shape[-1]
        index = torch.as_tensor(indices, device=numeric.device)
        return torch.cat(
            [
                numeric.index_select(-1, index),
                relative[..., :feature_count].index_select(-1, index),
                relative[..., feature_count:].index_select(-1, index),
            ],
            dim=-1,
        )

    def _pairwise_relations(
        self,
        numeric: torch.Tensor,
        relative: torch.Tensor,
    ) -> torch.Tensor:
        relation_base = self._select_feature_triplet(
            numeric,
            relative,
            self.relation_numeric_indices,
        )
        difference = relation_base.unsqueeze(2) - relation_base.unsqueeze(1)
        return torch.cat([difference, difference.abs()], dim=-1)

    def forward(
        self,
        numeric: torch.Tensor,
        relative: torch.Tensor,
        categorical: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_attention: bool = False,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | dict[str, torch.Tensor]:
        embeddings = [
            embedding(categorical[..., index])
            for index, embedding in enumerate(self.embeddings)
        ]
        candidate_input = torch.cat([numeric, relative] + embeddings, dim=-1)
        hidden = self.candidate_encoder(candidate_input) * mask.unsqueeze(-1)
        relations = self._pairwise_relations(numeric, relative)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, attention = block(hidden, mask, relations)
            if return_attention:
                attentions.append(attention)

        mask_float = mask.unsqueeze(-1).float()
        context = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        expanded_context = context.unsqueeze(1).expand_as(hidden)
        residual = self.residual_head(
            torch.cat([hidden, expanded_context, hidden - expanded_context], dim=-1)
        ).squeeze(-1)

        prior_input = self._select_feature_triplet(
            numeric,
            relative,
            self.prior_numeric_indices,
        )
        prior = self.prior_head(prior_input).squeeze(-1)
        gate = self.residual_gate(torch.cat([hidden, prior_input], dim=-1)).squeeze(-1)
        residual_component = (
            torch.nn.functional.softplus(self.residual_scale) * gate * residual
        )
        scores = prior + residual_component

        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        score_mean = (scores * mask).sum(dim=1, keepdim=True) / valid_count
        scores = (scores - score_mean).masked_fill(~mask, -1e4)

        if return_components:
            return {
                "scores": scores,
                "prior": prior.masked_fill(~mask, -1e4),
                "residual": residual_component.masked_fill(~mask, 0.0),
                "gate": gate.masked_fill(~mask, 0.0),
            }
        if return_attention:
            return scores, attentions
        return scores
