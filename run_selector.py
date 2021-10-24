#!/usr/bin/env python3
import argparse
import os
import json

import numpy as np

import joblib
from utils import handle_script_info, get_info

import run_predict as rt

from cst_transform.data.vocab_utils import MultiIndexer


class MaxSelector:

    def __init__(self, scorers):
        self.scorers = scorers

    def predict(self, X):

        predictions = []

        for scorer in self.scorers:
            p = scorer.predict_proba(X)
            p = p[:, 1] # Assume correct is class 1
            predictions.append(p)

        predictions = np.stack(predictions).transpose()
        selection = predictions.argmax(axis = 1)

        return selection, predictions


def build_selector(folder):
    
    config_path = os.path.join(folder, "config.json")

    if not os.path.exists(config_path):
        print("A configuration file in \"%s\" does not exist. Abort." % folder)
        exit()
    
    with open(config_path, "r") as cfg_file:
        config = json.load(cfg_file)

    labels = []
    scorers = []
    
    for label, scorer_path in config["checkpoints"].items():
        labels.append(label)
        scorer_path = os.path.join(folder, scorer_path)
        scorers.append(joblib.load(scorer_path))

    selector = MaxSelector(scorers)
    selector.tools = labels
    selector.embed = config["embedding"]

    return selector


def embed_code(checkpoint, input_file):

    base = os.path.abspath(__file__)
    base = os.path.dirname(base)

    if checkpoint in rt.checkpoints:
        ase_checkpoint = True
        checkpoint = rt.checkpoints[checkpoint]
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
    model = rt.load_model(checkpoint, len(labels), ase_checkpoint)

    print("Parse program from file %s" % input_file)
    program_repr = rt.parse_program(input_file, indexer)

    _, embedding = rt.predict(model, program_repr, labels)

    return np.array([float(v) for v in embedding[0]])


if __name__ == "__main__":

    handle_script_info()

    parser = argparse.ArgumentParser()
    parser.add_argument("selector_folder")
    parser.add_argument("input_file")

    parser.add_argument("--target")
    parser.add_argument("--embed_file", action="store_true")

    args = parser.parse_args()

    selector = build_selector(args.selector_folder)

    if args.embed_file:
        with open(args.input_file, "r") as i:
            embedding = np.array(json.load(i))
    else:
        embedding = embed_code(selector.embed, args.input_file)

    selection, prediction = selector.predict(embedding.reshape(1, -1))
    
    if not args.target:
        print("Prediction:")

        for i, tool in enumerate(selector.tools):
            prob = prediction[0, i]
            print("%s: %.2f%%" % (tool, 100 * prob))

        print()
        print("Best prediction: %s" % selector.tools[selection[0]])
    else:
        position   = selector.tools.index(args.target)
        confidence = prediction[0, position]

        print("Target label: %s" % args.target)
        print("Confidence: %f" % confidence)
