import torch as th
import torch
import torch_geometric
from torch_geometric import data as th_data

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes

import lmdb
import random

try:
    from . import treedata_pb2 as proto
except ImportError:
    import treedata_pb2 as proto

from time import time

class TreeBatch(th_data.Data):
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = set.union(*keys)
        keys.remove('depth_count')
        keys = list(keys)
        depth = max(data.depth_count.shape[0] for data in data_list)
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {i: 0 for i in range(depth)}
        depth_count = th.zeros((depth,), dtype=th.long)
        batch.batch = []
        for i, data in enumerate(data_list):
            edges = data['edge_index']
            for d in range(1, depth):
                mask = data.depth_mask == d
                edges[mask] += cumsum[d - 1]
                cumsum[d - 1] += data.depth_count[d - 1].item()
            batch['edge_index'].append(edges)
            depth_count += data['depth_count']

            for key in data.keys:
                if key == 'edge_index' or key=='depth_count':
                    continue
                item = data[key]
                batch[key].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        batch.depth_count = depth_count
        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()


class TreeDataLoader(th.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                return TreeBatch.from_data_list(batch, follow_batch)
            elif isinstance(elem, torch.Tensor):
                return default_collate(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
                return type(elem)(*(collate(s) for s in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                return [collate(s) for s in zip(*batch)]

            raise TypeError('DataLoader found invalid type: {}'.format(
                type(elem)))

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda batch: collate(batch), **kwargs)



class SliceDataset(th_data.Dataset):

    def __init__(self, delegate, slice):
        self._slice = slice
        self._delegate = delegate

    def __len__(self):
        return len(self._slice)

    def __getitem__(self, index):
        if isinstance(index, slice):
            slice_ix = self._slice[index]
            return SliceDataset(self._delegate, slice_ix)

        assert index <= len(self)

        ix = self._slice[index]
        return self._delegate[ix]

    def __name__(self):
        return self._delegate.__name__()

    def __repr__(self):
        return "%s(%d)" % (self.__name__(), len(self))

class ContextLMDBDataset(th_data.Dataset):

    def __init__(self, root, base_db, shuffle=False, transform=None,
                 slice_index=None):
        self._env = lmdb.open(root,
                              max_readers=1,
                              readonly=True,
                              lock=False,
                              readahead=False,
                              meminit=False,
                              max_dbs=1)

        self._db = self._env.open_db(key=base_db.encode('utf-8'), create=False)
        with self._env.begin(write=False, db=self._db) as txn:
            self._len = txn.stat()['entries']
        self._indices = list(range(self._len))
        if slice_index is not None:
            self._indices = slice_index
        self._transform = transform
        if shuffle:
            random.shuffle(self._indices)

    def __name__(self):
        return 'ContextLMDBDataset'

    def __len__(self):
        return self._len

    def __getitem__(self, index):

        if isinstance(index, slice):
            slice_ix = self._indices[index]
            return SliceDataset(self, slice_ix)

        if th.is_tensor(index):
            ix = th.LongTensor(self._indices)
            slice_ix = ix[index].tolist()
            return SliceDataset(self, slice_ix)

        assert index < len(self), 'index range error for ix: %i' % index
        graph_key = '%d' % (self._indices[index])

        with self._env.begin(write=False, db=self._db) as txn:
            graph_bin = txn.get(graph_key.encode('utf-8'))

        if self._transform is not None:
            graph_bin = self._transform(graph_bin)

        return graph_bin

    def __repr__(self):
        return "%s(%d)" % (self.__name__(), len(self))


def to_proto(data):
    tree = proto.AnnotatedTree()
    tree.ParseFromString(data)
    return tree


def _add_node(index, buffer, data, n):
    if n in index:
        return index[n]
    depth = data.depth[n]
    while len(buffer) <= depth:
        buffer.append([[], [], [], []])
    cbuffer = buffer[depth]
    id = len(cbuffer[0])
    index[n] = id
    cbuffer[0].append(data.nodes[n])
    cbuffer[1].append(data.position[n])
    cbuffer[2].append(depth)
    return id


def prep_data(data):

    index = {}
    buffer = []

    for i in range(len(data.assignment)):

        f = data.from_node[i]
        t = data.assignment[i]
        add_depth = data.depth[f]
        f = _add_node(index, buffer, data, f)
        t = _add_node(index, buffer, data, t)
        buffer[add_depth][3].append(t)

    x = []
    position = []
    depth = []
    edge_index = []
    if len(buffer[0][3]) == 0:
        edge_index.append(0)
    depth_count = []

    for i in range(len(buffer)):
        c = buffer[i]
        x.extend(c[0])
        position.extend(c[1])
        depth.extend(c[2])
        depth_count.append(len(c[2]))
        edge_index.extend(c[3])

    return x, position, depth, depth_count, edge_index


def load_prepped(data):

    depth_count = [0]*4

    for d in data.depth:
        depth_count[d] += 1

    return data.nodes, data.position, data.depth, depth_count, data.assignment


def prepped_to_tensor(data):

    x, position, depth, depth_count, edge_index = data
    x = th.LongTensor(x)
    position = th.LongTensor(position)
    depth = th.LongTensor(depth)
    edge_index = th.LongTensor(edge_index)
    return th_data.Data(
        x=x, position=position, depth_mask=depth, depth_count=th.LongTensor(depth_count),
        edge_index=edge_index
    )
