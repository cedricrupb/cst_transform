import inspect

from .tree import TreeModel
from .embedder import PositionalEncoding

from .config_transformer import HTransformerConfig

from .model_transformer import (
    HierarchicalTransformerLayer,
    HierarchicalTransformer,
    HierarchicalEncoder
)

from .config_extra import ExtraTransformerConfig

from .model_extra import (
    PosDepthEmbedding,
    ExtraHierarchicalEncoder
)


def default_config(name):

    if name == 'hierarchical_encoder':
        return {
            'config': {
                'hidden_size': 64,
                'intermediate_size': 64,
                'max_depth': 3,
                'attention_dropout_ratio': 0.1,
                'residual_dropout_ratio': 0.1
            }
        }

    if name == 'extra_encoder':
        return {
            'config': {
                'max_seq_length': 128,
                'hidden_size': 64,
                'intermediate_size': 64,
                'max_depth': 3,
                'attention_dropout_ratio': 0.1,
                'residual_dropout_ratio': 0.1
            }
        }

    return {}


def get_config_class(name):
    if name == "hierarchical_encoder":
        return HTransformerConfig

    if name == "extra_encoder":
        return ExtraTransformerConfig

    raise ValueError("Unknown model type: %s" % name)


def get_model_class(name):

    if name == "hierarchical_encoder":
        return HierarchicalEncoder

    if name == "extra_encoder":
        return ExtraHierarchicalEncoder

    raise ValueError("Unknown model type: %s" % name)


def create_config(name, config):
    config_clazz = get_config_class(name)

    return config_clazz(**reduce_by_signature(
        config_clazz, config
    ))


def create_model(name, experiment, cfg=None):

    if cfg is None:
        cfg = {}

    cfg['num_features'] = experiment.num_features
    cfg['output_size'] = experiment.num_classes

    print(cfg)

    return create_cfg_model(name, cfg)


def create_cfg_model(name, cfg=None):

    create_cfg = default_config(name)

    if cfg is not None:
        create_cfg.update(cfg)

    if 'config' in create_cfg:
        create_cfg['config'] = create_config(name, create_cfg['config'])

    clazz = get_model_class(name)

    return clazz(
        **reduce_by_signature(clazz, create_cfg)
    )


def reduce_by_signature(clazz, mapping):

    sig = inspect.signature(clazz)
    res = {}

    for k, V in mapping.items():
        if k in sig.parameters:
            res[k] = V
    return res
