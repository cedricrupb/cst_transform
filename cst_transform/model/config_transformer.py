

class HTransformerConfig:
    """
        Configuration for hierarchical transformer.
    """
    def __init__(
        self,
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=1,
        num_layers_per_depth=1,
        max_depth=3,
        attention_dropout_ratio=0.1,
        residual_dropout_ratio=0.1
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_layers_per_depth = num_layers_per_depth
        self.max_depth = max_depth
        self.attention_dropout_ratio = attention_dropout_ratio
        self.residual_dropout_ratio = residual_dropout_ratio
