import os
import argparse
import logging
import json
import shutil

from tqdm import tqdm
from glob import glob

from urllib import request


from cst_transform.utils import svcomp, label_utils
from cst_transform.data import preprocessor as p
from cst_transform.data import vocab_utils, lmdb_utils


DEFAULT_DATASETS = {
    "bmc-ki": "https://www.sosy-lab.org/research/strategy-selection/supplements.tar.bz2",
    "algorithms": "https://www.sosy-lab.org/research/strategy-selection/supplements.tar.bz2",
    "sc": "https://www.sosy-lab.org/research/strategy-selection/supplements.tar.bz2",
    "tools": "https://sv-comp.sosy-lab.org/2018/results/results-verified/All-Raw.zip"
}

DEFAULT_TOOL_SELECTION = {
    "bmc-ki": "./cst_transform/resources/bmc-ki.json",
    "algorithms": "./cst_transform/resources/algorithms.json",
    "sc": "./cst_transform/resources/sc.json",
    "tools": "./cst_transform/resources/tools.json"
}

DEFAULT_PREFIX = {
    "bmc-ki": "../programs/benchmarks/",
    "algorithms": "../programs/benchmarks/",
    "sc": "../programs/benchmarks/",
    "tools": "../sv-benchmarks/c/"
}


class DownloadHook:

    def __call__(self, chunk_id, chunk_size, total_chunks):
        if not hasattr(self, "pbar"):
            total_size = total_chunks // chunk_size
            self.pbar = tqdm(total=total_size+1, desc="Download:")
            self.last_chunk = 0

        chunk_update = chunk_id - self.last_chunk
        self.pbar.update(chunk_update)

        if chunk_id == total_chunks:
            self.pbar.close()
        else:
            self.last_chunk = chunk_id

def download_dataset(benchmark_url, data_dir):

    file_name = benchmark_url.split("/")[-1]
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        logging.info("Benchmark is already downloaded. Skip.")
        return file_path

    request.urlretrieve(benchmark_url, file_path, reporthook=DownloadHook())

    return file_path


def _unpack_bz2(file_path):

    dir_name = os.path.dirname(file_path)
    target_dir = os.path.join(dir_name, "content")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        logging.info("File seems already unpacked. Skip.")
        return target_dir

    import tarfile
    tar = tarfile.open(file_path, "r:bz2")
    tar.extractall(target_dir)
    tar.close()

    return target_dir


def _unpack_zip(file_path):
    dir_name = os.path.dirname(file_path)
    target_dir = os.path.join(dir_name, "content")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        logging.info("File seems already unpacked. Skip.")
        return target_dir

    import zipfile
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    return target_dir


def unpack_file(file_path):

    if file_path.endswith(".tar.bz2"):
        return _unpack_bz2(file_path)

    if file_path.endswith(".zip"):
        return _unpack_zip(file_path)

    return file_path

def _process_set_file(file_path, tmp_dir):

    all_xml_files = []

    with open(file_path, "r") as i:
        files = i.readlines()

    for f in files:
        if not os.path.exists(f):
            file_path = download_dataset(f, tmp_dir)
            xml_files = find_xml_files(file_path, tmp_dir)
        else:
            xml_files = find_xml_files(f, tmp_dir)

        all_xml_files.extend(xml_files)

    return all_xml_files


def _scan_dir_for_xml(dir_path):
    R = glob(os.path.join(dir_path, "**", "*.xml"), recursive=True)
    if len(R) == 0:
        R = glob(os.path.join(dir_path, "**", "*.xml.bz2"), recursive=True)
    return R


def find_xml_files(file_path, tmp_dir):

    if os.path.isdir(file_path):
        return _scan_dir_for_xml(file_path)

    if file_path.endswith(".set"):
        return _process_set_file(file_path, tmp_dir)

    if file_path.endswith(".tar.bz2") or file_path.endswith(".zip"):
        dir_path = unpack_file(file_path)
        return _scan_dir_for_xml(dir_path)

    if file_path.endswith(".xml") or file_path.endswith(".xml.bz2"):
        return [file_path]


def _should_parse(file_path, tool_selection):
    if tool_selection is None:
        return True

    file_name = os.path.basename(file_path)
    tool = file_name.split(".")[0]
    return tool in tool_selection


def parse_benchmark_files(xml_files, tool_selection=None, prefix=""):

    if tool_selection:
        tool_selection = set(tool_selection)

    result = {}
    for xml_file in tqdm(xml_files):

        # Typically the tool name occurs in the file path
        if not _should_parse(xml_file, tool_selection):
            logging.info("Skip %s as it not in tool selection." % xml_file)
            continue

        file_name = os.path.basename(xml_file)
        tool = file_name.split(".")[0]

        if xml_file.endswith(".bz2"):
            svcomp.read_svcomp_bz2(xml_file, result, tool_name=tool, prefix=prefix)
        else:
            with open(xml_file, "r") as i:
                svcomp.read_svcomp_xml_str(i.read(), result, tool_name=tool, prefix=prefix)

    return result


def _collect_files(label_result, prefix, allowed_keys=None):
    label_result = label_result['reachability']

    for tool_results in label_result.values():
        for key, result in tool_results.items():

            if allowed_keys is not None and key not in allowed_keys:
                continue

            yield result['file'].replace(prefix, "")


def _scan_dir_for_c(file_path):
    return glob(os.path.join(file_path, "**", "*.[c|i]"), recursive=True)


def _load_split(split_path):
    with open(split_path, "r") as i:
        split = json.load(i)

    return split


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



def phase1_labels(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    if args.benchmark_url in DEFAULT_DATASETS:
        args.bench_prefix = DEFAULT_PREFIX[args.benchmark_url]
        args.tool_selection = DEFAULT_TOOL_SELECTION[args.benchmark_url]
        args.benchmark_url = DEFAULT_DATASETS[args.benchmark_url]

    # Test if benchmark is local file
    if not os.path.exists(args.benchmark_url):
        file_path = download_dataset(args.benchmark_url, args.tmp_dir)
        xml_files = find_xml_files(file_path, args.tmp_dir)
    else:
        xml_files = find_xml_files(args.benchmark_url, args.tmp_dir)

    logging.info("Compress all benchmark files into one single label file.")

    if args.tool_selection:
        with open(args.tool_selection, "r") as i:
            args.tool_selection = json.load(i)

    labels = parse_benchmark_files(xml_files, tool_selection=args.tool_selection,
                                    prefix=args.bench_prefix)

    allowed_files = set(
                        _collect_files(labels, args.bench_prefix,
                                       allowed_keys=allowed_keys)
                        )

    if args.rank_mode:
        logging.error("Rank mode is currently unsupported")
        if args.cleanup:
            logging.info("Cleanup tmp folder.")
            shutil.rmtree(args.tmp_dir)
        exit()
    else:
        labels = label_utils.result_to_correct_labels(labels, args.tool_selection)

    logging.info("Save gold labels...")
    with open(os.path.join(args.output_dir, "gold_labels.json"), "w") as o:
        json.dump(labels, o, indent=4)

    if args.prune_unsolvable:
        labels = label_utils.prune_unsolvable(labels)

    logging.info("Save labels...")
    with open(os.path.join(args.output_dir, "labels.json"), "w") as o:
        json.dump(labels, o, indent=4)

    if args.cleanup:
        logging.info("Cleanup tmp folder.")
        shutil.rmtree(args.tmp_dir)

    del labels
    del xml_files

    return allowed_files


def phase2_preprocess_code(args, allowed_files):


    found_files = set(
        [file_path for file_path in _scan_dir_for_c(args.benchmark_code_dir)
         if file_path.replace(args.benchmark_code_dir, "") in allowed_files]
    )

    vocab_file_path = os.path.join(
        args.output_dir, "vocabulary.json"
    )

    lmdb_dir_path = os.path.join(
        args.output_dir, "dataset"
    )

    lmdb_index_path = os.path.join(
        args.output_dir, "index.lst"
    )

    if args.split:
        lmdb_dir_path = os.path.join(
            args.tmp_dir, "dataset"
        )

    if os.path.exists(lmdb_dir_path):
        logging.info("Path %s seem to exists. Delete before proceeding. Skip" % lmdb_dir_path)
        return lmdb_dir_path

    cst_indexer = vocab_utils.MultiIndexer()
    if os.path.exists(vocab_file_path):
        with open(vocab_file_path, "r") as i:
            cst_indexer.from_json_io(i)

    processed_files = []
    stats = {}

    pipeline = p.ClangASTParser()
    print(found_files)


    try:
        for i, proto in _stream_proto_objs(found_files, pipeline,
                                        processed_files=processed_files):
            stats[proto.name] = len(proto.nodes)

    finally:
        with open("ast_stats.json", "w") as o:
            json.dump(stats, o, indent=4)



    return lmdb_dir_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark_url", required=True,
                        help="Url to a repository of benchmark results. Alternatively can point to a local file \
                                or a set file containing multiple entries. \
                                Shortcuts: [bmc-ki, algorithms, sc, tools] for standard datasets")

    parser.add_argument(
        "--benchmark_code_dir", required=True,
        help="Directory path to a clone of SV-Benchmark.\
               For reproducing results, clone from https://github.com/sosy-lab/sv-benchmarks/tree/svcomp18"
    )

    parser.add_argument("--output_dir", required=True,
                        help="The output directory to store the dataset")

    parser.add_argument(
        "--tmp_dir", help="A temporary directory for intermediate files. Default: The same as output directory."
    )

    parser.add_argument(
        "--split", help="Path to a file containing the dataset split. If not given, it will result in a monolithic dataset."
    )

    parser.add_argument(
        "--rank_mode", action="store_true",
        help="Whether preprocess the labels as a ranking."
    )

    parser.add_argument(
        "--isomorph_dedup", action="store_true",
        help="Whether to deduplicate isomorph CSTs in the dataset"
    )

    parser.add_argument(
        "--prune_unsolvable", action="store_true",
        help="Prune all instances which cannot be solved"
    )

    parser.add_argument(
        "--tool_selection",
        help="Process only a subset of tools for labeling. Default: All tools will be processed. If a standard dataset is used, this option will be set automatically."
    )

    parser.add_argument(
        "--bench_prefix",
        help="Benchmark files often include a system dependent prefix. Used to normalize data.",
        default="../sv-benchmarks/c/"
    )

    parser.add_argument("--seq_semantic", action="store_true",
                        help="If set, handle ast elements as a sequence instead of set")

    parser.add_argument(
        "--cleanup", action="store_true",
        help="Cleanup the tmp folder after every step. Only possible if output folder and tmp folder differ."
    )


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.tmp_dir:
        args.tmp_dir = args.output_dir

    if args.tmp_dir == args.output_dir and args.cleanup:
        logging.warning("Cannot cleanup as tmp folder and output folder match.")
        args.cleanup = False


    allowed_keys = None
    if args.split:
        splits = _load_split(args.split)
        allowed_keys = set.union(
            *[set(v) for v in splits.values()]
        )


    # Start processing
    logging.info("######## Phase 1: Downloading and preprocessing labels ###########")

    allowed_files = phase1_labels(args)

    logging.info("######## Phase 2: Generate CSTs from C programs ###########")

    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    phase2_preprocess_code(args, allowed_files)
