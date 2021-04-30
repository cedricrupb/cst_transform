import argparse
import os
import json

import torch

from collections import OrderedDict

from cst_transform.data.vocab_utils import MultiIndexer
from cst_transform.data import preprocessor as p
from cst_transform.data.dataset import prepped_to_tensor, load_prepped

from cst_transform.experiments.svcomp import Experiment
from cst_transform.model import create_model

num_features = 366
num_classes = 3

checkpoints = {
    "bmc-ki"     : "checkpoints/bmc",
    "sc"         : "checkpoints/compositions",
    "algorithms" : "checkpoints/algorithms",
    "tools"      : "checkpoints/tools"
}

def parse_program(file_path, indexer):
    
    ast_parser = p.Pipeline(
        [
            p.ClangASTParser(),
            p.AST2CSTProcessor(token_index=indexer,
                                set_semantic=True),
            p.CSTCollate()
        ]
    )
    
    proto = ast_parser(file_path)

    return prepped_to_tensor(load_prepped(proto))


def _map_checkpoint_config(config):

    model_cfg = {
        "hidden_size": config["hidden"],
        "intermediate_size": config["hidden"] * config["reducer_config"]["intermediate_dim_ratio"],
        "max_depth": config["max_depth"]
    }

    return {"config": model_cfg}



def _prepare_model(args_path, config_path, num_classes, ase_checkpoint=True):
    with open(args_path) as i:
        args = json.load(i)

    with open(config_path) as i:
        model_cfg = json.load(i)

    if ase_checkpoint:
        model_cfg = _map_checkpoint_config(model_cfg)

    experiment = Experiment(
        None, num_features, num_classes, None
    )

    model = create_model(
        args['model_type'], experiment, model_cfg
    )

    return model


def _map_key(key):
    
    if key.startswith("reducer"):
        key = key.replace("reducer", "transform")
    
    if "multiplexer" in key:
        key = key.replace("multiplexer", "reducers")
    
    if key.startswith("transform.reducers") and "rff" in key:
        key = key.replace("rff", "feedforward")

    if "attention" in key: key = key.replace("attention", "attention.attention")

    if key.startswith("output"):
        key = key.replace("output", "output_transform")

    if key.startswith("norm"):
         key = key.replace("norm", "output_norm")

    return key

def _map_old_checkpoint(state_dict):
    return OrderedDict((_map_key(k), v) for k, v in state_dict.items())


def load_model(model_dir, num_classes, ase_checkpoint=True):
    if not os.path.exists(model_dir):
        raise ValueError("Model path %s does not exists." % model_dir)

    model_path = os.path.join(model_dir, 'model.pt')
    args_path = os.path.join(model_dir, 'args.json')
    config_path = os.path.join(model_dir, 'config.json')

    model = _prepare_model(args_path, config_path, num_classes, ase_checkpoint)
    param = torch.load(model_path, map_location=torch.device("cpu") )
    
    if ase_checkpoint:
        param = _map_old_checkpoint(param)
    
    model.load_state_dict(param)
    model.eval()

    return model


def predict(model, data, classes):

    data.x[data.x < 0] = 36

    pred = model(
        data.x, data.depth_mask, data.edge_index
    ).squeeze()

    mask = torch.tensor([1 if c.startswith("-") else 0 for c in classes]).float()
    pred = pred * (1 - mask) - 1e9 * mask 

    probs = torch.nn.Softmax(dim=-1)(pred)
    
    class_probs = [(c, probs[i].item()) for i, c in enumerate(classes)]
    class_probs = sorted(class_probs, key=lambda x: x[1], reverse=True)
    return class_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_file")

    parser.add_argument("--checkpoint", default="tools")

    args = parser.parse_args()

    checkpoint = args.checkpoint

    base = os.path.abspath(__file__)
    base = os.path.dirname(base)

    if checkpoint in checkpoints:
        checkpoint = checkpoints[checkpoint]
        checkpoint = os.path.join(base, checkpoint)

    index_path = os.path.join(base, "cst_transform/resources/token_clang.json")
    label_path = os.path.join(checkpoint, "label_order.json")

    if not os.path.exists(index_path):
        print("Cannot find index file at %s" % index_path)
        exit()

    indexer = MultiIndexer()
    with open(index_path, "r") as i:
        indexer.from_json_io(i)
    indexer.close()

    with open(label_path, "r") as i:
        labels = json.load(i)

    # Load model
    print("Load model from checkpoint %s" % checkpoint)
    model = load_model(checkpoint, len(labels))

    print("Parse program from file %s" % args.program_file)
    program_repr = parse_program(args.program_file, indexer)

    prediction = predict(model, program_repr, labels)

    print("Prediction:")

    for tool, prob in prediction:
        print("%s: %.2f%%" % (tool, 100 * prob))

    print()
    prediction = [k for k, v in prediction if not k.startswith("-")]
    print("Best prediction: %s" % prediction[0])
