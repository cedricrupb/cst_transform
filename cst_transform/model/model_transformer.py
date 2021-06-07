import math
import torch as th

from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, softmax

try:
    from tree import TreeModel
except ImportError:
    from .tree import TreeModel


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + th.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * th.pow(x, 3))))


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

        self.value = th.nn.Linear(config.hidden_size, self.all_head_size)
        self.output = th.nn.Linear(self.all_head_size, config.hidden_size)

        self.dropout = th.nn.Dropout(config.attention_dropout_ratio)

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


        assert context.size(0) == queries.size(0), "Expected a dim %d but got %d" % (queries.size(0), context.size(0))
        assert context.size(1) == queries.size(1), "Expected a dim %d but got %d" % (queries.size(1), context.size(1))

        context = self.output(context)

        return (context, att)


class ResidualModule(th.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = th.nn.Dropout(dropout)

    def forward(self, x, subLayer):
        return x + self.dropout(subLayer)


class HierarchicalAttentionOutput(th.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = SparseAttention(config)
        self.residual = ResidualModule(config.residual_dropout_ratio)

    def forward(self, parents, children, edges):
        result, attention = self.attention(children, parents, edges)

        assert result.size(0) == parents.size(0)

        return self.residual(parents, result), attention


class rFF(th.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_norm = th.nn.LayerNorm(config.hidden_size)
        self.in_ff = th.nn.Linear(config.hidden_size, config.intermediate_size)
        self.out_ff = th.nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):

        out = self.in_norm(x)
        out = self.in_ff(out)
        out = gelu_new(out)
        out = self.out_ff(out)

        return out


class HierarchicalTransformerLayer(th.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = HierarchicalAttentionOutput(config)
        self.feedforward = rFF(config)
        self.residual = ResidualModule(config.residual_dropout_ratio)

    def forward(self, parents, children, edges, return_attention=False):
        intermediate, attention = self.attention(parents, children, edges)

        inter_output = self.feedforward(intermediate)

        output = self.residual(intermediate, inter_output)

        if return_attention:
            return output, attention
        return output


class HierarchicalTransformerStack(th.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = th.nn.ModuleList(
            [HierarchicalTransformerLayer(config) for _ in range(config.num_layers_per_depth)]
        )

    def forward(self, parents, children, edges, return_attention=False):
        all_attention = () if return_attention else None

        for layer_module in self.layers:

            layer_output = layer_module(
                parents, children, edges,
                return_attention=return_attention
            )

            if return_attention:
                all_attention += (layer_output[1],)
                layer_output = layer_output[0]

            parents = layer_output

        if return_attention:
            return parents, all_attention

        return parents


class HierarchicalTransformer(TreeModel):

    def __init__(self, config):

        layers = th.nn.ModuleList([
            HierarchicalTransformerLayer(config) if config.num_layers_per_depth == 1 else HierarchicalTransformerStack(config)
            for _ in range(config.max_depth)
        ])

        super().__init__(layers)


class HierarchicalEncoder(th.nn.Module):

    def __init__(self, config, num_features, output_size, dropout=0.0):
        super().__init__()

        self.embedding = th.nn.Embedding(
            num_features, config.hidden_size
        )

        self.transform = HierarchicalTransformer(
            config
        )

        self.output_norm = th.nn.LayerNorm(config.hidden_size)
        self.output_dropout = th.nn.Dropout(dropout)
        self.output_transform = th.nn.Linear(config.hidden_size, output_size)

    def forward(self, x, depth_mask, edge_index, return_attention=False, return_hidden=False):

        nodes = self.embedding(x)

        output = self.transform(
            nodes, depth_mask, edge_index,
            return_attach=return_attention
        )

        if return_attention: output, attention = output[0], output[1]

        output = self.output_norm(output)
        output = self.output_dropout(output)
        logits = self.output_transform(output)

        if return_attention or return_hidden:
            logits = (logits,)
            
            if return_attention: logits += (attention,)
            if return_hidden:    logits += (output,)

        return logits
