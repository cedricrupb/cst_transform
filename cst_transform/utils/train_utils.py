import os
import json
import random
import numpy as np


def path_to_id(path):
    filename = os.path.basename(path)
    dirname  = os.path.dirname(path)
    
    prefix = []

    while len(dirname) > 0:
        dirname, current = os.path.split(dirname)
        prefix.append(current)
        if current == "sv-benchmarks": break
    full = prefix[::-1] + [filename]
    return os.path.join(*full)


def path_to_group(path):
    return os.path.basename(os.path.dirname(path))


def join_labels(labels):
    
    common_entries = list(set.intersection(*[set(l[0]) for l in labels]))
    join_labels = []

    for label_index, L in labels:
        index = {k: i for i, k in enumerate(label_index)}
        subset_index = [index[k] for k in common_entries]
        sublabels = L[subset_index]
        join_labels.append(sublabels)

    if len(join_labels) > 1:
        join_labels = np.vstack(join_labels).transpose()
    else:
        join_labels = join_labels[0]

    return common_entries, join_labels


class EmbeddingLoader:

    def __init__(self, embedding_file, labels_file):
        self.embedding_file = embedding_file
        self.labels_file = labels_file

        self._tools = None
        self._label_cache = {}
        self._verdict_cache = {}
        self._embed_cache = None

    def _init_label_cache(self, key):
        if key in self._label_cache: return

        self._label_cache[key] = {}

        tools = set()
        with open(self.labels_file, "r") as i:
            for line in i:
                content = json.loads(line)
                task_id = path_to_id(content["input_file"])

                if key in content["results"]:
                    self._label_cache[key][task_id] = content["results"][key]
                    self._verdict_cache[task_id] = content["verdict"]

                for tool in content["results"]: tools.add(tool)
        self._tools = tools

    @property
    def tools(self):
        if self._tools is None:
            tools = set()
            with open(self.labels_file, "r") as i:
                for line in i:
                    content = json.loads(line)
                    for tool in content["results"]: tools.add(tool)

            self._tools = tools

        return self._tools

    @property
    def embeddings(self):
        
        if self._embed_cache is None:
            self._embed_cache = {}
            with open(self.embedding_file, "r") as i:
                for line in i: 
                    content = json.loads(line)
                    self._embed_cache[path_to_id(content["path"])] = content["embedding"]

        return np.array(list(self._embed_cache.values()))

    
    def embedding(self, path):
        if self._embed_cache is None: self.embeddings
        key = path_to_id(path)

        if key not in self._embed_cache:
            raise ValueError("There does not exists a precomputed embedding for %s" % path)
        
        return np.array(self._embed_cache[key])


    def labels(self, key):
        
        if isinstance(key, tuple):
            labels = [self.labels(k) for k in key]
            return join_labels(labels)
        
        if self._tools is not None and key not in self._tools:
            raise ValueError("Tool %s is not supported, available tools are %s" % (key, str(self._tools)))

        if key not in self._label_cache: 
            self._init_label_cache(key)
            return self.labels(key)

        label_index = []
        labels = []

        for input_file, result in self._label_cache[key].items():
            label_index.append(input_file)
            labels.append(1 if result["label"] else 0)

        return label_index, np.array(labels)

    def __call__(self, *keys, return_groups = False):
        
        label_index, labels = self.labels(keys)
        label_subset = []
        embeddings   = []

        if self._embed_cache is None: self.embeddings

        for i, label_key in enumerate(label_index):
            if label_key in self._embed_cache:
                label_subset.append(i)
                embeddings.append(self._embed_cache[label_key])
        
        embeddings = np.array(embeddings)
        labels = labels[label_subset]

        output = (embeddings, labels,)

        if return_groups:
            group_index = [path_to_group(label_index[i]) for i in label_subset]
            group_keys  = {k: i for i, k in enumerate(set(group_index))}
            group_index = [group_keys[g] for g in group_index]

            output += (group_index,)


        return output


def group_stratified_split(groups, *arrays, ratio = 0.1):
    
    assert all(len(groups) == a.shape[0] for a in arrays), "All have to have same size at dim 0"

    train_indices, test_indices = [], []
    max_group = max(groups)

    for index in range(max_group + 1):
        group_index = [i for i, g in enumerate(groups) if g == index]
        test_size   = int(len(group_index) * ratio)
        test_index = random.sample(group_index, test_size)
        test_indices.extend(test_index)
        train_index = [i for i in group_index if i not in test_index]
        train_indices.extend(train_index)

    output = tuple()

    for array in arrays:
        output += (array[train_indices, :], array[test_indices, :])

    return output