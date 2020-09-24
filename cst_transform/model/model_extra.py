import torch as th


try:
    from .model_transformer import HierarchicalTransformer
except ImportError:
    from model_transformer import HierarchicalTransformer



class PosDepthEmbedding(th.nn.Module):

    def __init__(self, num_features, config):
        super().__init__()

        self.max_seq_length = config.max_seq_length

        self.embedding = th.nn.Embedding(
            num_features, config.hidden_size
        )

        self.postion_embedding = th.nn.Embedding(
            config.max_seq_length, config.hidden_size
        )

        self.depth_embedding = th.nn.Embedding(
            config.max_depth+1, config.hidden_size
        )


    def forward(self, nodes, depth_mask, position=None):

        if position is None:
            position = th.ones(nodes.shape, dtype=th.long, device=nodes.device)

        assert nodes.size(0) == depth_mask.size(0)
        assert nodes.size(0) == position.size(0)

        # If a sequence is longer than max sequence, handle it as a set
        position [position >= self.max_seq_length] = self.max_seq_length - 1

        node_embedding = self.embedding(nodes)
        depth_embedding = self.depth_embedding(depth_mask)
        postion_embedding = self.postion_embedding(position)

        return node_embedding + depth_embedding + postion_embedding



class ExtraHierarchicalEncoder(th.nn.Module):

    def __init__(self, config, num_features, output_size, dropout=0.0):
        super().__init__()

        self.embedding = PosDepthEmbedding(
            num_features, config
        )

        self.transform = HierarchicalTransformer(
            config
        )

        self.output_norm = th.nn.LayerNorm(config.hidden_size)
        self.output_dropout = th.nn.Dropout(dropout)
        self.output_transform = th.nn.Linear(config.hidden_size, output_size)

    def forward(self, x, depth_mask, edge_index, position=None, return_attention=False):

        nodes = self.embedding(
            x, depth_mask, position=position
        )

        output = self.transform(
            nodes, depth_mask, edge_index,
            return_attach=return_attention
        )

        if return_attention:
            output, attention = output[0], output[1]

        output = self.output_norm(output)
        output = self.output_dropout(output)
        output = self.output_transform(output)

        if return_attention:
            return output, attention

        return output
