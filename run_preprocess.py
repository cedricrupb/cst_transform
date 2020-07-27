import os
import argparse

import json

from tqdm import tqdm
from glob import glob


from cst_transform.data import preprocessor as p
from cst_transform.data import vocab_utils, lmdb_utils



def scan_for_files(base_dir):
    return glob(os.path.join(base_dir, "**", "*.[c|i]"), recursive=True)


def _handle_yml(path):

    with open(path, "r") as i:
        content = i.readlines()

    base_path = os.path.dirname(path)

    c_file = None

    for line in content:
        if line.startswith("input_files"):
            path = line.strip().split(" ")[1]
            path = path.strip()[1:-1]
            path = os.path.join(base_path, path)
            c_file = path
            break
    return c_file


def scan_for_sets(pattern):

    out = []
    basepath, _ = os.path.split(pattern)

    for setpath in glob(pattern):
        with open(setpath, "r") as i:
            paths = i.readlines()

        for path in paths:
            path = path.strip()
            if len(path) > 0:
                path = os.path.join(basepath, path)
                for path in glob(path):
                    if path.endswith("yml"):
                        out.append(_handle_yml(path))
                    else:
                        out.append(path)

    return out


def _stream_proto_objs(scan, pipeline, processed_files=None):
    c = 0
    nix = 0
    for f in tqdm(scan):
        R = pipeline(f)
        if R is None:
            c += 1
            print("Number of failed attempts: %d" % c)
        else:
            o = nix
            nix += 1

            if processed_files:
                processed_files.append(f)

            yield o, R
    print("Number of failed attempts: %d" % c)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir",
                        help="Directory containing either C files or\
                                Set description (if --set_file).\
                                Script will parse recursively.")
    parser.add_argument("output_lmdb_dir",
                        help="Directory which will store the LMDB database.")

    parser.add_argument("--token_index",
                        help="A token multi index which translate tokens to numeric ids.")

    parser.add_argument("--set_file", action="store_true",
                        help="If set, then an input set file is expected for input_dir")

    parser.add_argument("--seq_semantic", action="store_true",
                        help="If set, handle ast elements as a sequence instead of set")

    args = parser.parse_args()

    file_scan = None
    if args.set_file:
        file_scan = scan_for_sets(args.input_dir)
    else:
        file_scan = scan_for_files(args.input_dir)

    cst_indexer = vocab_utils.MultiIndexer()
    if args.token_index and os.path.exists(args.token_index):
        with open(args.token_index, "r") as i:
            cst_indexer.from_json_io(i)

    lmdb_index_path = os.path.join(
        args.output_lmdb_dir, "index.lst"
    )

    processed_files = []

    pipeline = p.Pipeline(
        [
            p.ClangASTParser(),
            p.AST2CSTProcessor(token_index=cst_indexer,
                                set_semantic=not args.seq_semantic),
            p.CSTCollate()
        ]
    )

    try:
        lmdb_utils.feed_lmdb(args.output_lmdb_dir,
                             _stream_proto_objs(file_scan, pipeline,
                                                processed_files=processed_files)
                             )
    finally:
        if args.token_index:
            with open(args.token_index, "w") as o:
                cst_indexer.to_json(file_object=o)

        with open(lmdb_index_path, "w") as o:
            for file_path in processed_files:
                o.write(f"{file_path}\n")
