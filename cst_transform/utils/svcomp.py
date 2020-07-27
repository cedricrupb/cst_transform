import argparse
import requests
import math
from tqdm import tqdm
import os
import bz2

from zipfile import ZipFile
from glob import glob
import xml.etree.ElementTree as ET

import subprocess as sp
import shutil

import json
import time


types_str = {
    'memory': ['valid-deref',
               'valid-free',
               'valid-memtrack',
               'valid-memcleanup',
               'valid-memsafety'],
    'reachability': ['unreach-call'],
    'overflow': ['no-overflow'],
    'termination': ['termination']
}

SV_BENCHMARKS_GIT = "https://github.com/sosy-lab/sv-benchmarks.git"


def _download_file(url, path):
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    print("Download url: %s >> %s" % (url, path))
    with open(path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size/block_size), unit='KB'):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size and os.path.getsize(path) != total_size:
        os.remove(file)
        raise ValueError("ERROR, something went wrong")


def create_folder(path):
    if len(os.path.splitext(path)[1]) > 0:
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        os.makedirs(path)


def read_svcomp_data(path):
    D = {}

    directory = os.path.dirname(path)
    tmp_dir = os.path.join(directory, "tmp")

    try:
        with ZipFile(path, 'r') as zipObj:
            zipObj.extractall(path=tmp_dir)

        for entry in tqdm(glob(os.path.join(tmp_dir, "*.bz2"))):
            read_svcomp_bz2(entry, D)

    finally:
        if os.path.exists(tmp_dir):
            os.system('rm -rf %s' % tmp_dir)

    return D


def detect_type(text):
    global types_str

    for k, V in types_str.items():
        for v in V:
            if v in text:
                return k

    raise ValueError("Unknown type for properties \"%s\"" % text)


def short_name(name, prefix="../sv-benchmarks/c/"):
    name = name.replace(prefix, "")
    name = name.replace("/", "_")
    name = name.replace(".", "_")
    return name


def ground_truth(file, type):
    global types_str

    for type_str in types_str[type]:
        if 'true-%s' % type_str in file:
            return True
        if 'false-%s' % type_str in file:
            return False

    raise ValueError("Cannot detect type %s in file \"%s\"." % (type, file))


def read_svcomp_xml_str(xml_content, result, tool_name=None, prefix="../sv-benchmarks/c/"):
    root = ET.fromstring(xml_content)
    # print(root)

    if tool_name is None:
        tool_name = root.attrib['benchmarkname']

    for run in root.iter('run'):
        attr = run.attrib
        file = attr['name']
        name = short_name(file, prefix)
        category = detect_type(attr['properties'])

        for column in run:
            title = column.attrib['title']

            if title == 'status':
                status = column.attrib['value']

            if title == 'cputime':
                cputime = float(column.attrib['value'][:-1])

            if title == 'memUsage':
                memUsage = int(column.attrib['value'])


        if category not in result:
            result[category] = {}
        if tool_name not in result[category]:
            result[category][tool_name] = {}
        result[category][tool_name][name] = {
            'file': file,
            'status': status,
            'ground_truth': ground_truth(file, category),
            'cputime': cputime,
            'memUsage': memUsage
        }


def read_svcomp_features(xml_content, result, prefix="../sv-benchmarks/c/"):
    root = ET.fromstring(xml_content)
    # print(root)

    columns = set(['Boolean', 'Loops', 'Composite types', 'Aliasing', 'Arrays', 'Floats'])

    tool_name = root.attrib['name']

    for run in root.iter('run'):
        attr = run.attrib
        file = attr['name']
        name = short_name(file, prefix)
        category = detect_type(attr['properties'])

        D = {}

        for column in run:
            title = column.attrib['title']

            if title in columns:
                D[title] = column.attrib['value']

        if category not in result:
            result[category] = {}
        if tool_name not in result[category]:
            result[category][tool_name] = {}
        result[category][tool_name][name] = {
            'file': file
        }
        result[category][tool_name][name].update(D)


def read_svcomp_bz2(path, result, tool_name=None, prefix="../sv-benchmarks/c/"):
    with bz2.open(path) as o:
        xml = o.read()
    read_svcomp_xml_str(xml, result, tool_name=tool_name, prefix=prefix)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_file")
    parser.add_argument("out")

    parser.add_argument("-a", "--attach", action="store_true")
    parser.add_argument("-p", "--prefix", default="../sv-benchmarks/c/")
    parser.add_argument("-f", "--features", action="store_true")

    args = parser.parse_args()

    base = {}
    if args.attach:
        with open(args.out, "r") as i:
            base = json.load(i)

    with open(args.xml_file, "r") as i:
        if args.features:
            read_svcomp_features(
                i.read(), base, prefix=args.prefix
            )
        else:
            read_svcomp_xml_str(
                i.read(), base, prefix=args.prefix
            )

    with open(args.out, "w") as o:
        json.dump(base, o, indent=4)
