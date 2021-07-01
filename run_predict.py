#!/usr/bin/env python3
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


def quick_load_linear(model_dir, num_classes, ase_checkpoint=False):
    if not os.path.exists(model_dir):
        raise ValueError("Model path %s does not exists." % model_dir)

    model_path = os.path.join(model_dir, 'model.pt')
    config_path = os.path.join(model_dir, 'config.json')

    with open(config_path, 'r') as f:
        hidden_size = json.load(f)["hidden"]
    print("Predictor: Linear model (%d x %d)" % (hidden_size, num_classes))
    model = torch.nn.Linear(hidden_size, num_classes)
    param = torch.load(model_path, map_location=torch.device("cpu") )
    
    if ase_checkpoint:
        param = _map_old_checkpoint(param)

    linear_params = {
        "weight": param["output_transform.weight"],
        "bias"  : param["output_transform.bias"]
    }
    
    model.load_state_dict(linear_params)
    model.eval()

    return model


def _log_to_classes(logits, classes):
    mask = torch.tensor([1 if c.startswith("-") else 0 for c in classes]).float()
    logits = logits * (1 - mask) - 1e9 * mask 

    probs = torch.nn.Softmax(dim=-1)(logits)
    
    class_probs = [(c, probs[i].item()) for i, c in enumerate(classes)]

    return sorted(class_probs, key=lambda x: x[1], reverse=True)


def predict(model, data, classes):

    data.x[data.x < 0] = 36

    pred, embedding = model(
        data.x, data.depth_mask, data.edge_index, return_hidden=True
    )
    pred = pred.squeeze()

    return _log_to_classes(pred, classes), embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_file", nargs='?')

    parser.add_argument("--checkpoint", default="tools")
    parser.add_argument("--ase_checkpoint", action="store_true")

    # Embedding
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--embed_file", help="If embed is activated, write to this file. Otherwise, load the embedding from the path.")

    # Confidence
    parser.add_argument("--target", help="Optional argument. Produces a confidence score for the given target if possible")

    args = parser.parse_args()

    if not args.embed_file or args.embed: # The only case where no program file is required
        if not args.program_file:
            print("An program file can be substituted with an embedding file.")
            print("However, an embedding file is not provided. Therefore, a program file is needed.")
            print("Note: While producing an embedding file (--embed), a program file is also needed.")
            print("Exit.")
            exit()

    checkpoint = args.checkpoint

    ase_checkpoint = args.ase_checkpoint
    base = os.path.abspath(__file__)
    base = os.path.dirname(base)

    if checkpoint in checkpoints:
        ase_checkpoint = True
        checkpoint = checkpoints[checkpoint]
        checkpoint = os.path.join(base, checkpoint)

    index_path = os.path.join(base, "cst_transform/resources/token_clang.json")
    label_path = os.path.join(checkpoint, "label_order.json")

    if not os.path.exists(index_path):
        print("Cannot find index file at %s" % index_path)
        exit()
    
    if args.embed_file and not args.embed and not os.path.exists(args.embed_file):
        print("Try to use precomputed embedding %s, but it does not exists." % args.embed_file)
        exit()

    indexer = MultiIndexer()
    with open(index_path, "r") as i:
        indexer.from_json_io(i)
    indexer.close()

    with open(label_path, "r") as i:
        labels = json.load(i)

    if args.target and args.target not in labels:
        print("Target label %s is not supported by checkpoint %s. Supported target labels are: %s"
                % (args.target, args.checkpoint, str(labels)))
        exit()

    if args.embed_file and not args.embed:
        # Quick load model and run prediction from precomputed embedding
        print("Load linear classifier from checkpoint %s" % checkpoint)
        model = quick_load_linear(checkpoint, len(labels), ase_checkpoint)

        with open(args.embed_file, "r") as i:
            embedding = torch.FloatTensor(json.load(i))
        
        prediction = _log_to_classes(model(embedding), labels)
    
    else:
         # Load model
        print("Load model from checkpoint %s" % checkpoint)
        model = load_model(checkpoint, len(labels), ase_checkpoint)

        print("Parse program from file %s" % args.program_file)
        program_repr = parse_program(args.program_file, indexer)

        prediction, embedding = predict(model, program_repr, labels)

        if args.embed:
            print("Embedding:")
            print(", ".join([str(float(v)) for v in embedding[0]]))

            if args.embed_file:
                print("Save embedding to %s" % args.embed_file)
                with open(args.embed_file, "w") as o:
                    json.dump([float(v) for v in embedding[0]], o)

            exit()

    if not args.target:
        print("Prediction:")

        for tool, prob in prediction:
            print("%s: %.2f%%" % (tool, 100 * prob))

        print()
        prediction = [k for k, v in prediction if not k.startswith("-")]
        print("Best prediction: %s" % prediction[0])
    else:
        probs = [p[1] for p in prediction]
        min_prob, max_prob = min(probs), max(probs)

        target_prob = [p for p in prediction if p[0] == args.target]
        print("Target label: %s" % args.target)
        print("Confidence: %f" % ((target_prob[0][1] - min_prob) / (max_prob - min_prob)))