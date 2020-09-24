import torch
try:
    from . import treedata_pb2 as proto
except ImportError:
    import treedata_pb2 as proto

class NodeInterface:

    def __init__(self):
        self.label = "None"
        self.row = 0
        self.column = 0
        self.range = None

    def __str__(self):
        if self.range:
            pos = f"[{self.range[0]}:{self.range[1]}, {self.range[2]}:{self.range[3]}]"
        else:
            pos = f"[{self.row}:{self.column}]"

        return f"Node [label={self.label}, pos={pos}]"


def default_node_interface(node):
    node_interface = NodeInterface()
    node_interface.label = str(node)

    return node_interface


def to_dot(traversal, node_interface=default_node_interface):

    seen = set([])
    node_ser = ""
    edge_ser = ""

    for parent, pid, _, child, cid, _ in traversal:
        if pid not in seen:
            seen.add(pid)
            parent = node_interface(parent)
            node_ser += "N%d [label=\"%s\"];\n" % (pid, parent.label)
        if cid not in seen:
            seen.add(cid)
            child = node_interface(child)
            node_ser += "N%d [label=\"%s\"];\n" % (cid, child.label)
        edge_ser += "N%d -> N%d;\n" % (cid, pid)
    return "digraph Dummy {\n%s%s}" % (node_ser, edge_ser)


def to_proto(traversal, indexer, node_interface=default_node_interface):

    tree = proto.AnnotatedTree()
    trans = {'count': 0}

    for parent, pid, attr_name, child, cid, depth in traversal:
        if parent is None or child is None:
            continue

        parent_interface = node_interface(parent)
        child_interface = node_interface(child)

        reindex_child = not _match_line(parent_interface, child_interface)
        cid = _add_proto_node(tree, child_interface, cid, depth, trans, indexer, use_row=reindex_child)
        pid = _add_proto_node(tree, parent_interface, pid, depth - 1, trans, indexer)
        attr_id = indexer.index('attr_names', attr_name)

        tree.from_node.append(cid)
        tree.assignment.append(pid)
        tree.assign_attr.append(attr_id)

    return tree

def proto_to_traversal(pb, indexer):

    for i in range(len(pb.assignment)):
        cid = pb.from_node[i]
        pid = pb.assignment[i]
        parent = indexer.reverse('ast', pb.nodes[pid])
        child = indexer.reverse('ast', pb.nodes[cid])
        attr_name = "ast"
        if len(pb.assign_attr) > 0:
            attr_name = indexer.reverse('attr_name', pb.assign_attr[i])
        depth = pb.depth[cid]
        yield parent, pid, attr_name, child, cid, depth

def plain_proto_to_traversal(pb):

    for i in range(len(pb.assignment)):
        cid = pb.from_node[i]
        pid = pb.assignment[i]
        parent = str(pb.nodes[pid])
        child = str(pb.nodes[cid])
        attr_name = str(pb.assign_attr[i])
        depth = pb.depth[cid]
        yield parent, pid, attr_name, child, cid, depth


def depth_indexed_to_traversal(pb, indexer):

    depths = []

    for i in range(len(pb.nodes)):
        depth = pb.depth[i]

        while depth >= len(depths):
            depths.append([])

        depths[depth].append(i)

    for depth in range(len(depths)-1, 0, -1):
        parents = depths[depth-1]
        for cid in depths[depth]:
            edge_id = pb.assignment[cid]
            pid = parents[edge_id]

            parent = indexer.reverse('ast', pb.nodes[pid])
            child = indexer.reverse('ast', pb.nodes[cid])
            attr_name = "<?>"

            yield parent, pid, attr_name, child, cid, depth


def data_to_traversal(data, indexer):

    max_depth = data.depth_count.size(0)-1

    index = torch.range(0, data.x.size(0)-1)

    child_mask = data.depth_mask == max_depth
    buffer = data.x[child_mask]
    buffer_index = index[child_mask]

    for current_depth in range(max_depth - 1, -1, -1):

        parent_mask = data.depth_mask == current_depth
        parents = data.x[parent_mask]
        parents_index = index[parent_mask]
        edges = data.edge_index[child_mask]

        for id in range(buffer.size(0)):

            cid = buffer_index[id].item()
            pid = edges[id].item()
            rpid = parents_index[pid].item()

            child = indexer.reverse('ast', buffer[id].item())
            parent = indexer.reverse('ast', parents[pid].item())
            attr_name = "<?>"
            yield parent, rpid, attr_name, child, cid, current_depth


        buffer = parents
        buffer_index = parents_index
        child_mask = parent_mask


class Node:

    def __init__(self, id, **kwargs):
        self.id = id

        for key, value in kwargs.items():
            setattr(self, key, value)


class ReadGraph:

    EDGE_KEYS = ["from_node", "assignment", "assign_attr", "name"]

    def __init__(self, proto_t, reverser=None):
        self.name = proto_t.name
        self._proto = proto_t
        self._reverser = reverser
        self.root = -1
        self._ingoing = {}
        self._index_ingoing()

        node_length = len(proto_t.nodes)

        self._proto_fields = set([
            field.name for field in proto_t.DESCRIPTOR.fields
            if field.name not in ReadGraph.EDGE_KEYS and\
            len(getattr(proto_t, field.name)) == node_length
        ])


    def _index_ingoing(self):
        data = self._proto

        for i in range(len(data.assignment)):
            cid = data.from_node[i]
            pid = data.assignment[i]
            if pid not in self._ingoing:
                self._ingoing[pid] = []
            self._ingoing[pid].append(cid)
            if self.root == -1 or data.depth[pid] == 0:
                self.root = pid

    def in_edges(self, id):
        if id not in self._ingoing:
            return []
        return self._ingoing[id]

    def node(self, id):

        lix = self._proto.nodes[id]

        kwargs = {
            "id": id
        }

        if self._reverser:
            kwargs["label"] = self._reverser.reverse('ast', lix)
        else:
            kwargs["label_id"] = lix

        for field_name in self._proto_fields:
            kwargs[field_name] = getattr(self._proto, field_name)[id]

        return Node(
            **kwargs
        )

    def rewrite_label(self, id, label):
        if not self._reverser:
            raise ValueError("Invalid state. Require a reverser but reverser is None")
        self._proto.nodes[id] = self._reverser.index('ast', label)


class WriteGraph:

    def __init__(self, name, indexer=None):
        self.proto = proto.AnnotatedTree()
        self.proto.name = name
        self._fields = set([field.name for field in self.proto.DESCRIPTOR.fields])
        self._field_access_pattern = None
        self._indexer = indexer

    def _validate_access(self, kwargs):
        keys = [k for k in kwargs.keys() if k in self._fields]

        if not self._field_access_pattern:
            self._field_access_pattern = keys
            return

        wrong_keys = set([])
        missing_keys = set(self._field_access_pattern)

        for k in keys:
            if k not in self._field_access_pattern:
                wrong_keys.add(k)
                continue
            missing_keys.remove(k)

        ERROR_MSG = "The attribute of a node have to be consistent across all writes, but %s"

        assert len(wrong_keys) == 0, ERROR_MSG % ("received new keys: %s" % str(list(wrong_keys)))
        assert len(missing_keys) == 0, ERROR_MSG % ("the write misses the following attributes: %s" % str(list(missing_keys)))

    def _write_kwargs(self, kwargs, warn=True):

        if "node" in kwargs and isinstance(kwargs["node"], Node):
            for key, value in kwargs["node"].__dict__.items():
                if key not in kwargs:
                    kwargs[key] = value
            del kwargs["node"]
            warn = False

        self._validate_access(kwargs)

        for key, value in kwargs.items():
            if key not in self._fields:
                if warn:
                    print("Cannot assign %s=%s as %s is not a field of a node" % (str(key), str(value), str(key)))
                continue

            proto_obj = getattr(self.proto, key)

            try:
                proto_obj.append(value)
            except Exception:
                proto_obj.add().CopyFrom(value)


    def write_node(self, label, assign_to=-1, **kwargs):

        data = self.proto
        id = len(data.nodes)

        if self._indexer is not None:
            lix = self._indexer.index('ast', label)
        else:
            lix = label

        kwargs["nodes"] = lix

        depth = 0

        if assign_to != -1:
            if assign_to < len(data.depth):
                depth = data.depth[assign_to] + 1
            data.from_node.append(id)
            data.assignment.append(assign_to)

        if "depth" not in kwargs:
            kwargs["depth"] = depth

        self._write_kwargs(kwargs)

        return id

    def rewrite_label(self, id, label):
        self._proto.nodes[id] = self._indexer.index('ast', label)

    def read_graph(self):
        return ReadGraph(
            self.proto, self._indexer
        )



def _match_line(parent, child):
    return parent.row == child.row

def _add_token_position(tree, node):
    token_pos = tree.tokens.add()
    range = node.range

    token_pos.lineStart = range[0]
    token_pos.lineEnd = range[2]
    token_pos.charStart = range[1]
    token_pos.charEnd = range[3]


def _add_proto_node(tree, node, id, depth, trans, indexer, use_row=False):

    if id not in trans:

        trans[id] = trans['count']
        trans['count'] += 1

        label = node.label
        label_id = indexer.index('ast', label)
        tree.nodes.append(label_id)
        tree.depth.append(depth)
        position = node.column
        tree.position.append(position)

        if node.range:
            _add_token_position(tree, node)

    if use_row:
        tree.position[trans[id]] = node.row

    return trans[id]
