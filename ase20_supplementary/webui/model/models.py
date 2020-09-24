import torch
import torch as th
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_scatter import scatter
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax

import inspect
import math


def default_config(name):

    if name == 'sum_tree':
        return {
            'reducer': 'sum',
            'hidden': 16
        }

    if name == 'transformer_tree':
        return {
            'reducer': 'transformer',
            'hidden': 64,
            'max_depth': 3,
            'reducer_config': {
                'num_attention_heads': 1,
                'intermediate_dim_ratio': 1,
                'residual_dropout': 0.1,
                'connect_dropout': 0.1
            }
        }

    return {}


def reduce_by_signature(clazz, mapping):

    sig = inspect.signature(clazz)
    res = {}

    for k, V in mapping.items():
        if k in sig.parameters:
            res[k] = V
    return res


def get_model_class(name):

    if name.endswith("_tree"):
        return TreeModel

    raise ValueError("Unknown model type: %s" % name)


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def create_model(name, experiment, cfg=None):

    if cfg is None:
        cfg = {}

    cfg['num_features'] = experiment.num_features
    cfg['num_classes'] = experiment.num_classes

    print(cfg)

    return create_cfg_model(name, cfg)


def create_cfg_model(name, cfg=None):

    create_cfg = default_config(name)

    if cfg is not None:
        create_cfg.update(cfg)

    clazz = get_model_class(name)

    return clazz(
        **reduce_by_signature(clazz, create_cfg)
    )


class TransformerConfig:

    def __init__(self,
                 hidden_size=32,
                 num_attention_heads=1,
                 intermediate_dim_ratio=1,
                 connect_dropout=0.1,
                 residual_dropout=0.1,
                 value_transform=True):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.intermediate_dim_ratio = intermediate_dim_ratio
            self.connect_dropout = connect_dropout
            self.residual_dropout = residual_dropout
            self.value_transform = value_transform


def build_reducer(reducer, config):
    if reducer == 'sum':
        return SumReducer(), False

    if reducer == 'transformer':
        config = TransformerConfig(**config)
        return TransformerReducer(config), True

    raise ValueError("Unknown reducer %s" % reducer)


class SumReducer(th.nn.Module):

    def forward(self, iteration, childs, parents, edges):
        tmp = global_add_pool(childs, edges, size=parents.size(0))
        return tmp + parents

class SparseAttention(th.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size


        self.key = th.nn.Linear(config.hidden_size, self.all_head_size)
        self.query = th.nn.Linear(config.hidden_size, self.all_head_size)

        self.key_norm = th.nn.LayerNorm(config.hidden_size)
        self.query_norm = th.nn.LayerNorm(config.hidden_size)

        if config.value_transform:
            self.value = th.nn.Linear(config.hidden_size, self.all_head_size)
            self.output = th.nn.Linear(self.all_head_size, config.hidden_size)
        else:
            assert self.all_head_size == config.hidden_size
            self.value = th.nn.Identity()
            self.output = th.nn.Identity()

        self.dropout = th.nn.Dropout(config.connect_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, keys, queries, assignment):

        keys = self.key_norm(keys)
        queries = self.query_norm(queries)

        mixed_keys = self.key(keys)
        mixed_queries = self.query(queries)
        mixed_value = self.value(keys)

        mixed_keys = self.transpose_for_scores(mixed_keys)
        mixed_queries = self.transpose_for_scores(mixed_queries)
        mixed_value = self.transpose_for_scores(mixed_value)

        raw_att = (mixed_keys * mixed_queries.index_select(0, assignment)).sum(dim=-1)

        raw_att = raw_att / math.sqrt(self.attention_head_size)

        att = softmax(raw_att, assignment)

        att = self.dropout(att)

        att = att.view(-1, self.num_heads, 1)

        value = mixed_value
        context = (value * att).contiguous()

        new_context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_layer_shape)

        context = scatter(context, assignment.view(-1, 1),
                          dim=0, dim_size=queries.size(0))
                          
        context = self.output(context)

        return (context, att)


class ResidualModule(th.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = th.nn.Dropout(dropout)

    def forward(self, x, subLayer):
        return x + self.dropout(subLayer)


class rFF(th.nn.Module):

    def __init__(self, hidden, m=1):
        super().__init__()
        self.in_norm = th.nn.LayerNorm(hidden)
        self.in_ff = th.nn.Linear(hidden, m*hidden)
        self.out_ff = th.nn.Linear(m*hidden, hidden)

    def forward(self, x):

        out = self.in_norm(x)
        out = self.in_ff(out)
        out = gelu_new(out)
        out = self.out_ff(out)

        return out

class TransformerReducer(th.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = SparseAttention(config)
        self.att_res = ResidualModule(dropout=config.residual_dropout)
        self.rff = rFF(config.hidden_size, m=config.intermediate_dim_ratio)
        self.res = ResidualModule(dropout=config.residual_dropout)

    def forward(self, iteration, children, parents, assign):

        att_out, attention = self.attention(children, parents, assign)
        att_out = self.att_res(parents, att_out)
        out = self.rff(att_out)
        out = self.res(att_out, out)

        return out, attention


class MultiplexerReducer(th.nn.Module):

    def __init__(self, reducer, reducer_config, max_depth):
        super().__init__()
        multi = []

        self.attach = False

        for _ in range(max_depth):
            reducer_model, attach = build_reducer(reducer, reducer_config)
            self.attach = self.attach or attach
            multi.append(
                reducer_model
            )

        self.multiplexer = th.nn.ModuleList(
            multi
        )

    def forward(self, iteration, children, parents, assign):

        return self.multiplexer[iteration](iteration, children, parents, assign)


class TreeModel(th.nn.Module):

    def __init__(self, num_features, hidden, num_classes,
                 max_depth=-1,
                 max_len=-1,
                 reducer="sum",
                 reducer_config={},
                 dropout=0.0,
                 return_attach=False):

        super().__init__()

        self.return_attach = return_attach

        self.embedding = th.nn.Embedding(num_features, hidden)

        reducer_config['hidden_size'] = hidden

        if max_depth > 0:
            self.reducer = MultiplexerReducer(
                reducer, reducer_config, max_depth
            )
            self.attach = self.reducer.attach
        else:
            self.reducer, self.attach = build_reducer(reducer, reducer_config)

        self.norm = th.nn.LayerNorm(hidden)
        self.dropout = th.nn.Dropout(dropout)
        self.output = th.nn.Linear(hidden, num_classes)

    def forward(self, x, depth_mask, edge_index, batch=None):

        max_depth = depth_mask.max().item()

        nodes = self.embedding(x)

        child_mask = depth_mask == max_depth
        buffer = nodes[child_mask]

        attach = []

        for lookup in range(max_depth - 1, -1, -1):
            parent_mask = depth_mask == lookup
            parents = nodes[parent_mask]
            edges = edge_index[child_mask]

            buffer = self.reducer(lookup, buffer, parents, edges)
            if self.attach:
                buffer, a = buffer
                attach += [a]
            child_mask = parent_mask

        buffer = self.norm(buffer)
        interm = self.dropout(buffer)
        buffer = self.output(interm)

        if self.attach and self.return_attach:
            attach = attach + [interm]
            return buffer, attach

        return buffer
