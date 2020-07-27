import hashlib
import functools



def wl_label(graph):
    """
        The graph is compressed to a single label bottom-up.
        Since we are working with trees this is a precise
        isomorphism check
        (for performance a hash function is used.
            Therefore it can happen that hash collison occur)
    """
    depth_sort = _depth_sort(graph)
    
    label_buffer = {
        cid: label for cid, label in enumerate(graph.proto.nodes)
    }

    for depth in range(len(depth_sort)-1, -1, -1):

        current_ids = depth_sort[depth]

        node_childs = map(lambda node_id: graph.in_edges(node_id), current_ids)
        node_mapper = functools.partial(_get_labels, buffer=label_buffer)

        node_childs = map(node_mapper, node_childs)

        wl_labels = map(_wl_label, current_ids, node_childs)

        label_buffer.update({
            current_ids[i]: label for i, label in enumerate(wl_labels)
        })

    return label_buffer[graph.root]




def _wl_label(node_id, child_ids):

    label = str(node_id)+"_("+"_".join([str(i) for i in child_ids])+")"
    label = hashlib.blake2b(label.encode("utf-8")).hexdigest()

    return str(label)

def _get_label(node_id, buffer):
    try:
        return buffer[node_id]
    except KeyError:
        return node_id

def _get_labels(node_ids, buffer):
    mapper = functools.partial(_get_label, buffer=buffer)
    return map(mapper, node_ids)



def _depth_sort(graph):

    depths = []

    if graph.is_depth_supported:
        for node_id in graph.ingoing.keys():
            depth = graph.proto.depth[node_id]

            while len(depths) <= depth:
                depths.append([])

            depths[depth].append(node_id)

        return depths

    depths.append([graph.root])
    Q = [(0, graph.root)]

    non_leaves = set(graph.ingoing.keys())

    while len(Q) > 0:
        depth, node_id = Q.pop(0)

        for in_node_id in graph.in_edges(node_id):
            if in_node_id == node_id:
                continue

            if in_node_id not in non_leaves:
                continue

            n_depth = depth + 1

            while len(depths) <= n_depth:
                depths.append([])

            depths[n_depth].append(in_node_id)
            Q.append((n_depth, in_node_id))

    return depths


class Node:

    def __init__(self, id, position):
        self.id = id
        self.position = position

class ReadGraph:

    def __init__(self, proto_t):
        self.name = proto_t.name
        self.proto = proto_t
        self.root = -1
        self.ingoing = {}
        self._index_ingoing()
        self.is_depth_supported = False

    def _index_ingoing(self):
        data = self.proto

        possible_roots = set(range(len(data.nodes)))

        for i in range(len(data.assignment)):
            cid = data.from_node[i]
            pid = data.assignment[i]

            if cid == pid:
                continue

            if pid not in self.ingoing:
                self.ingoing[pid] = []
            self.ingoing[pid].append(cid)

            if cid != pid and cid in possible_roots:
                possible_roots.remove(cid)

        assert len(possible_roots) == 1
        self.root = next(iter(possible_roots))



    def in_edges(self, id):
        if id not in self.ingoing:
            return []
        return self.ingoing[id]

    def node(self, id):
        pos = self.proto.position[id]
        lix = self.proto.nodes[id]
        return Node(
            id, position=pos
        )

class PReadGraph:

    def __init__(self, proto_t):
        self.name = proto_t.name
        self.proto = proto_t
        self.root = -1
        self.ingoing = {}
        self._index_ingoing()
        self.is_depth_supported = True
    def _index_ingoing(self):
        data = self.proto

        depth_index = {}

        for cid in range(len(data.assignment)):

            depth = data.depth[cid]
            if depth not in depth_index:
                depth_index[depth] = []
            depth_index[depth].append(cid)

            if depth == 0:
                self.root = cid
                continue

            pid = data.assignment[cid]
            pid = depth_index[depth - 1][pid]

            if pid not in self.ingoing:
                self.ingoing[pid] = []
            self.ingoing[pid].append(cid)

    def in_edges(self, id):
        if id not in self.ingoing:
            return []
        return self.ingoing[id]

    def node(self, id):
        pos = self.proto.position[id]
        lix = self.proto.nodes[id]
        return Node(
            id, position=pos
        )
