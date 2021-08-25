import argparse
import os
import bz2
import yaml
import json

from glob import glob
from tqdm import tqdm

import xml.etree.ElementTree as ET

def parse_file(file_path):
    with bz2.open(file_path, 'r') as f:
        root = ET.fromstring(f.read())
    
    tool = root.attrib['tool']

    for result in root.iter("run"):
        if result.attrib['properties'] != "unreach-call": continue
        name = result.attrib['name']
        status = None
        cputime = None

        for column in result.iter("column"):
            attr = column.attrib
            if attr['title'] == "status":
                status = attr['value']
            if attr['title'] == "cputime":
                cputime = float(attr['value'][:-1])

        if cputime > 900: status = "unknown"; cputime = 900
        
        yield {"tool": tool, "task_file": name, "status": status, "cputime": cputime}



def iterate_results(label_path):
    files = glob(os.path.join(label_path, "*.xml.bz2"), recursive=True)

    for file_path in tqdm(files):
        for result in parse_file(file_path):
            yield result


def ground_result(base_folder, input_file):
    file_path = os.path.join(base_folder, input_file)
    
    with open(file_path, 'r') as i:
        content = yaml.safe_load(i)

    input_file = content["input_files"]
    base_dir   = os.path.dirname(file_path)
    input_file = os.path.join(base_dir, input_file)

    verdict = None
    for prop in content["properties"]:
        if "unreach-call" in prop["property_file"]:
            verdict = prop["expected_verdict"]
    
    if verdict is None: raise ValueError("Given file is not a reachability task: %s" % input_file)

    return input_file, verdict


def compare_verdict(verdict, tool_output):
    if verdict: return "true" in tool_output
    return "false" in tool_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_labels")

    parser.add_argument("--svbench_path")

    args = parser.parse_args()

    svcomp_path = "."
    if args.svbench_path:
        svcomp_path = args.svbench_path
        assert os.path.exists(svcomp_path), "Directory does not exist: %s" % svcomp_path


    tmp_labels = {}

    for result in iterate_results(args.input_folder):
        task_id, tool = result["task_file"], result["tool"]

        if task_id not in tmp_labels: tmp_labels[task_id] = {}
        tmp_labels[task_id][tool] = {
            "status": result["status"],
            "cputime": result["cputime"]
        }

    with open(args.output_labels, "w") as output:
        for task_id, results in tmp_labels.items():
            input_file, verdict = ground_result(svcomp_path, task_id)
            input_file = input_file[len(svcomp_path):]

            output_result = {"input_file": input_file, "verdict": verdict, "results": {}}

            for tool, r in results.items():
                output_result["results"][tool] = {
                    "label": compare_verdict(verdict, r["status"]),
                    "cputime": r["cputime"]
                }
            
            output.write(json.dumps(output_result) + "\n")



if __name__ == '__main__':
    main()