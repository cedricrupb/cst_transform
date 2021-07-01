import argparse
import os
import json
import yaml
import traceback

from tqdm import tqdm
from glob import glob

import run_predict as rt

from cst_transform.data.vocab_utils import MultiIndexer


def _substitute(org_files, sub_files):
    index = {}

    for f in org_files:
        base, _ = os.path.splitext(f)
        index[base] = f

    for f in sub_files:
        base, _ = os.path.splitext(f)
        index[base] = f
    
    return list(index.values())

def _load_model(checkpoint):
    base = os.path.abspath(__file__)
    base = os.path.dirname(base)
    checkpoint = rt.checkpoints[checkpoint]
    
    index_path = os.path.join(base, "cst_transform/resources/token_clang.json")
    label_path = os.path.join(checkpoint, "label_order.json")

    indexer = MultiIndexer()
    with open(index_path, "r") as i:
        indexer.from_json_io(i)
    indexer.close()

    with open(label_path, "r") as i:
        labels = json.load(i)

    return rt.load_model(checkpoint, len(labels), True), indexer, labels


def parse_yml(file):
    with open(file, "r") as i:
        content = yaml.safe_load(i)

    base_dir = os.path.dirname(file)
    file_path = content["input_files"]
    file_path = os.path.join(base_dir, file_path)
    
    return file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_file")
    parser.add_argument("--file_index")

    parser.add_argument("--checkpoint", default="tools")

    args = parser.parse_args()

    if args.file_index:
        with open(args.file_index, "r") as i:
            files = [L.strip() for L in i.readlines()]
        
        files = [os.path.join(args.input_folder, f) for f in files]
        files = [f for f in files if os.path.isfile(f)]
    else:
        c_files = glob(os.path.join(args.input_folder, "**", "*.c"), recursive=True)
        i_files = glob(os.path.join(args.input_folder, "**", "*.i"), recursive=True)

        files   = _substitute(c_files, i_files) # Whenever an .i-File is available use it instead of the .c-File

    model, indexer, labels = _load_model(args.checkpoint)

    found = set()

    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as i:
            for line in i:
                path = json.loads(line)["path"]
                found.add(path)
        
        args.output_file = args.output_file + "-ct"

    T = tqdm(files)

    with open(args.output_file, "w") as file_writer:
        for file in T:
            file = parse_yml(file)
            if file in found: continue
            folder = os.path.dirname(file)
            folder = os.path.basename(folder)
            T.set_description("Folder: %s" % folder)

            try:
                program_repr = rt.parse_program(file, indexer)

                _, embedding = rt.predict(model, program_repr, labels)

                output = {
                    "path": file, "embedding": [float(v) for v in embedding[0]]
                }

                file_writer.write(json.dumps(output)+"\n")
            except Exception:
                traceback.print_exc()
                continue