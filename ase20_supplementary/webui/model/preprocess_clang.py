import argparse

try:
    from . import treedata_pb2 as proto
    from . import utils
except (ValueError, ImportError):
    import treedata_pb2 as proto
    import utils
    import represent_ast as ra

class Node:

    def __init__(self, id, label, position):
        self.id = id
        self.label = label
        self.position = position


class ReadGraph:

    def __init__(self, proto_t, reverser):
        self.name = proto_t.name
        self._proto = proto_t
        self._reverser = reverser
        self.root = -1
        self._ingoing = {}
        self._index_ingoing()

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
        pos = self._proto.position[id]
        lix = self._proto.nodes[id]
        return Node(
            id, self._reverser.reverse('ast', lix), position=pos
        )

    def rewrite_label(self, id, label):
        self._proto.nodes[id] = self._reverser.index('ast', label)


class WriteGraph:

    def __init__(self, name, indexer):
        self.proto = proto.AnnotatedTree()
        self.proto.name = name
        self._indexer = indexer

    def write_node(self, label, position=0, assign_to=-1):
        data = self.proto
        id = len(data.nodes)
        lix = self._indexer.index('ast', label)
        data.nodes.append(lix)

        if assign_to != -1:
            depth = data.depth[assign_to] + 1
            data.from_node.append(id)
            data.assignment.append(assign_to)
        else:
            depth = 0

        data.depth.append(depth)
        data.position.append(position)
        return id

    def rewrite_label(self, id, label):
        self._proto.nodes[id] = self._indexer.index('ast', label)

    def read_graph(self):
        return ReadGraph(
            self.proto, self._indexer
        )


def find_statement_roots(graph, root):
    roots = []

    check = set(['CompoundStmt', 'IfStmt'])

    Q = graph.in_edges(root)
    while len(Q) > 0:
        id = Q.pop()
        node = graph.node(id)
        label = node.label

        if label == 'CompoundStmt':
            Q.extend(graph.in_edges(id))
        else:
            roots.append(id)
            for u in graph.in_edges(id):
                n2 = graph.node(u)
                if n2.label in check:
                    Q.append(u)

    return roots

def ast_to_seq(graph, root):
    sequence = []
    Q = [root]
    seen = set([root])

    while len(Q) > 0:
        id = Q[0]
        Q = Q[1:]

        if id == "[SEP]":
            sequence.append(id)
            continue

        label = graph.node(id).label

        if id != root and label == 'IfStmt':
            graph.rewrite_label(id, 'ElseIfStmt')
            continue

        if label == 'CompoundStmt':
            continue

        sequence.append(label)

        neigh = sorted([u for u in graph.in_edges(id) if u not in seen], key=lambda x: graph.node(x).position)
        seen = seen.union(set(neigh))
        Q = neigh + Q

    return sequence


def ast_to_set(graph, root):
    out = set([])

    Q = [root]

    while len(Q) > 0:
        id = Q.pop()

        if id == "[SEP]":
            sequence.append(id)
            continue

        label = graph.node(id).label

        if id != root and label == 'IfStmt':
            graph.rewrite_label(id, 'ElseIfStmt')
            continue

        if label == 'CompoundStmt':
            continue

        out.add(label)

        neigh = graph.in_edges(id)
        Q.extend(neigh)

    return out


def transform_state(graph, cgraph, attach_root, root, set_sem=False):


    if set_sem:
        sequence = list(ast_to_set(graph, root))
    else:
        sequence = ast_to_seq(graph, root)

    for pos in range(len(sequence)):
        ipos = 0 if set_sem else pos
        cgraph.write_node(sequence[pos], position=ipos, assign_to=attach_root)


def stmt_label(label):
    return "Stmt_%s" % label



def transform_func(graph, cgraph, attach_func, root, set_sem=False):

    if attach_func == 1:
        state_roots = [root]
    else:
        state_roots = find_statement_roots(graph, root)

    state_roots = sorted(state_roots, key=lambda id: graph.node(id).position)
    attach_roots = []

    for pos, id in enumerate(state_roots):
        node = graph.node(id)
        nid = cgraph.write_node(node.label, position=pos, assign_to=attach_func)
        attach_roots.append(nid)

    for i in range(len(state_roots)):
        transform_state(graph, cgraph, attach_roots[i], state_roots[i],
                        set_sem=set_sem)



def attach_noop(graph):
    Q = [0]

    rgraph = graph.read_graph()

    while len(Q) > 0:
        current = Q.pop()
        edges = rgraph.in_edges(current)

        if len(edges) == 0:
            continue

        graph.write_node("[NOOP]", position=0, assign_to=current)

        Q.extend(edges)

def rm_empty_func(graph):
    rm = []

    for u in graph.in_edges("N0"):
        if len(graph.in_edges(u)) == 0:
            rm.append(u)

    for r in rm:
        graph.remove_node(r)



def transform_graph(graph, indexer, set_sem=False):

    cgraph = WriteGraph(graph.name, indexer)
    program_id = cgraph.write_node("PROGRAM", position=0)
    init_id = cgraph.write_node("InitFunctionDecl", position=0, assign_to=program_id)

    for u in graph.in_edges(graph.root):
        if u == graph.root:
            continue
        node = graph.node(u)
        root = init_id
        if 'FunctionDecl' in node.label:
            root = cgraph.write_node(node.label, position=0, assign_to=program_id)
        transform_func(graph, cgraph, root, u, set_sem=set_sem)

    attach_noop(cgraph)
    return cgraph


def preprocess(id, data, old_index, new_indexer, set_sem=False):

    graph = ReadGraph(data, old_index)

    out_graph = transform_graph(graph, new_indexer, set_sem=set_sem)

    out = out_graph.proto


    return id, out


def bounded_stream(stream, bound):

    for i, D in enumerate(stream):
        yield D
        if i >= bound:
            break
