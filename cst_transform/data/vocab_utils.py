import json

class Indexer(object):

    def __init__(self):
        self._buffer = {}
        self._counter = 0
        self._reverse = None
        self._close = False

    def _build_reverse(self):

        if self._reverse is None or len(self._reverse) < self._counter:
            self._reverse = [None]*self._counter
            for k, v in self._buffer.items():
                self._reverse[v] = k


    def _register(self, identifier):
        if identifier not in self._buffer:
            self._buffer[identifier] = self._counter
            self._counter += 1

    def index(self, identifier, add_index=True):
        if not self._close and add_index:
            self._register(identifier)
        try:
            return self._buffer[identifier]
        except KeyError:
            return -1

    def reverse(self, index):
        self._build_reverse()

        if index >= len(self._reverse):
            return None
        return self._reverse[index]

    def _to_dict_(self):
        D = {'_counter_': self._counter}
        D.update(self._buffer)
        return D

    def _from_dict_(self, D):
        self._counter = D['_counter_']
        self._buffer.update(D)
        del self._buffer['_counter_']
        return self

    def to_json(self, file_object=None):
        D = self._to_dict_()
        if file_object is None:
            return json.dumps(D)
        json.dump(D, file_object)

    def from_json(self, text):
        self._from_dict_(json.loads(text))

    def from_json_io(self, file_object):
        self._from_dict_(json.load(file_object))

    def close(self):
        self._close = True

    def __len__(self):
        return self._counter


class MultiIndexer(object):

    def __init__(self, names=[]):
        self._index = {}

        for n in names:
            self._index[n] = Indexer()

    def _register(self, name):
        if name not in self._index:
            self._index[name] = Indexer()

    def __len__(self):
        return sum([len(v) for v in self._index.values()])

    def index(self, name, identifier, add_index=True):
        if add_index:
            self._register(name)
        try:
            return self._index[name].index(identifier, add_index=add_index)
        except KeyError:
            return -1

    def reverse(self, name, index):
        try:
            return self._index[name].reverse(index)
        except KeyError:
            return None

    def _to_dict_(self):
        return {
            k: v._to_dict_() for k, v in self._index.items()
        }

    def _from_dict_(self, D):
        for k, V in D.items():
            self._index[k] = Indexer()._from_dict_(V)

    def to_json(self, file_object=None):
        D = self._to_dict_()
        if file_object is None:
            return json.dumps(D)
        json.dump(D, file_object)

    def from_json(self, text):
        self._from_dict_(json.loads(text))

    def from_json_io(self, file_object):
        self._from_dict_(json.load(file_object))

    def close(self):
        for indexer in self._index.values(): indexer.close()
