# adapted from:
# https://pytorch.org/docs/stable/_modules/torch/\
# nn/modules/transformer.html#TransformerEncoderLayer
from typing import Optional, Union, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# use local copy of multiheadattention to allow modifications
from model.transformer.multiheadattention import MultiheadAttention
from torch.nn.modules.normalization import LayerNorm
from copy import deepcopy


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, activation=nn.functional.relu, bias=True, dropout_p = 0.0):
        super().__init__()

        self.activation = activation
        self.dropout_internal = nn.Dropout(dropout_p)
        self.dropout_ff = nn.Dropout(dropout_p)

        self.up_projection = nn.Linear(dim, hidden_dim, bias)
        self.down_projection = nn.Linear(hidden_dim, dim, bias)
        
    def forward(self, x):
        x = self.up_projection(x)
        x = self.activation(x)
        x = self.dropout_internal(x)
        x = self.down_projection(x)
        x = self.dropout_ff(x)
        return x
    
# inspired by https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py
class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], router: nn.Module, num_experts_per_tok=1, compute_aux_loss=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts_per_tok = num_experts_per_tok
        self.compute_aux_loss = compute_aux_loss

    def aux_loss(self):
        return 0 
    
    def forward(self, inputs: torch.Tensor):

        router_logits = self.router(inputs)
        experts_weights, expert_indicies = torch.topk(router_logits, self.num_experts_per_tok)

        experts_weights = F.softmax(experts_weights, dim=-1) 

        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_indx, tok_indx, nth_expert = torch.where(expert_indicies == i) # only works with 3d inputs (b, t, expert) for now
            results[batch_indx, tok_indx, :] += experts_weights[batch_indx, tok_indx, nth_expert, None] * expert(inputs[batch_indx, tok_indx, :])
        
        if self.compute_aux_loss:
            pass

        return results


class MoeTransformerBlock(nn.Module):
    def __init__(self, model_params, train_params, num_experts = 8, num_experts_per_tok = 1, 
                 activation=nn.functional.relu, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True):
        super().__init__()
        self.model_params = model_params
        train_params = deepcopy(train_params)
        self.self_attn = MultiheadAttention(self.model_params,
                                            dropout=train_params.dropout,
                                            bias=bias, batch_first=batch_first)
       
        d_model = model_params.dim

        # MoE layer
        dim_feedforward = d_model * model_params.dim_ff_factor
        experts = [FeedForward(d_model, dim_feedforward, activation, bias, train_params.dropout) for _ in range(num_experts)]
        router = nn.Linear(d_model, num_experts, bias=False)
        self.moe_layer = MoELayer(experts, router, num_experts_per_tok)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout_sa = nn.Dropout(train_params.dropout)
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)

        for expert in self.moe_layer.experts:
            if not hasattr(expert, 'activation'):
                expert.activation = nn.functional.relu

    def forward(self, src, src_mask=None, attn_requests=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            attn_res, attn_pattern = self._sa_block(
                self.norm1(x), src_mask, attn_requests=attn_requests)
            x = x + attn_res
            x = x + self._moe_ff_block(self.norm2(x))
        else:
            attn_res, attn_pattern = self._sa_block(
                x, src_mask, attn_requests=attn_requests)
            x = self.norm1(x + attn_res)
            x = self.norm2(x + self._moe_ff_block(x))
        return x, attn_pattern

    # self-attention block
    def _sa_block(self, x, attn_mask=None, attn_requests=None):
        x, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, attn_requests=attn_requests)
        return self.dropout_sa(x), attn_weights

    # MoE feed forward block 
    def _moe_ff_block(self, x):
        x = self.moe_layer(x)
        return x 
