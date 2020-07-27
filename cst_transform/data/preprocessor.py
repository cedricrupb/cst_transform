import os
import inspect
import traceback

try:
    import clang as CLANG
    import ast_utils as au
    import vocab_utils
except ImportError:
    from . import clang as CLANG
    from . import ast_utils as au
    from . import vocab_utils

try:
    from . import treedata_pb2 as proto
except ImportError:
    import treedata_pb2 as proto


class ClangASTParser:
    """
        Uses Clang compiler front end to parse a C file.
        file_path --> ast_proto
    """
    def __init__(self, token_index=None):
        self.token_index = token_index

        if self.token_index is None:
            self.token_index = vocab_utils.MultiIndexer()

    def __call__(self, file_path):
        try:
            print("Run CLANG for %s" % file_path)
            traversal = CLANG.clang_to_traversal(file_path, truncate_level=2)
            ni = CLANG._clang_node_interface

            psplit = os.path.normpath(file_path).split(os.path.sep)
            name = os.path.join(psplit[-2], psplit[-1])

            print("To PB2 %s" % file_path)
            tree = au.to_proto(traversal, self.token_index, node_interface=ni)
            tree.name = name
            return tree
        except Exception:
             traceback.print_exc()


class AST2CSTProcessor:
    """
        Given an AST proto, the processor generates a CST representation.
        ast_proto --> ast_proto
    """

    def __init__(self, set_semantic=True, attach_dummy=True, token_index=None):
        self.token_index = token_index
        self.set_semantic = set_semantic
        self.attach_dummy = attach_dummy

        if self.token_index is None:
            self.token_index = vocab_utils.MultiIndexer()

    def __call__(self, ast_proto, token_index):

        assert token_index is not None

        if ast_proto is None:
            return None

        graph = au.ReadGraph(ast_proto, token_index)

        out_graph = _transform_ast_to_cst(graph,
                                          self.token_index,
                                          set_sem=self.set_semantic,
                                          attach_dummy=self.attach_dummy)

        return out_graph.proto

class CSTCollate:
    """
        Caution: This function destroys the standard storage format.
        However, it is necessary as a preprocessing step for training.

        Before: Nodes are numbered consecutive. Therefore, the exists
        no two nodes with the same identifier.

        ID:          0 1 2 3 4 5 6 7 8 9 10
        node_types:  3 3 2 1 3 6 5 4 7 3  2
        from_node:   0 1 1 0 ...
        to_node:     1 2 3 2 ...

        After: Nodes are indexed depth-wise. Therefore, nodes are only
        distinguishable on the same depth. E.g. every first node at every
        depth is assigned with id 0.

        Depth:       0 1 1 2 2 2 2 3 3 3 3
        ID:          0 0 1 0 1 2 3 0 1 2 3
        node_types:  3 3 2 1 3 6 5 4 7 3 2
        from_node:   0 1 1 0 ...
        to_node:     1 2 3 2 ...

        ast_proto --> ast_proto
    """
    def __init__(self):
        pass

    def __call__(self, ast_proto):

        if ast_proto is None:
            return None

        index = {}
        buffer = []

        for i in range(len(ast_proto.assignment)):
            # For every edge in the graph
            f = ast_proto.from_node[i]
            t = ast_proto.assignment[i]
            add_depth = ast_proto.depth[f]

            # Assigns a new depth based index
            f = _add_node(index, buffer, ast_proto, f)
            t = _add_node(index, buffer, ast_proto, t)

            # Saves the edge into the buffer
            buffer[add_depth][1].append(t)

        if len(buffer[0][1]) == 0:
            buffer[0][1].append(0)


        # Unroll buffer
        read_graph = au.ReadGraph(ast_proto)
        name = ast_proto.name.replace("/", "_").replace(".", "_")
        write_graph = au.WriteGraph(name)

        for i in range(len(buffer)):
            cbuffer = buffer[i]
            for node_ix in range(len(cbuffer[0])):

                node_id = cbuffer[0][node_ix]
                node = read_graph.node(node_id)
                assign = cbuffer[1][node_ix]

                write_graph.write_node(
                    node.label_id, assign_to=assign, node=node,
                    depth=i
                )

        return write_graph.proto


class Pipeline:

    INPUT_TOKEN = "__input__"

    def __init__(self, processors):
        self.processors = processors

    def _forward(self, func, kwargs):
        sig = inspect.signature(func.__call__)

        input_kwargs = {}
        zero_key = next(iter(sig.parameters.keys()))

        for key, value in kwargs.items():
            if key == Pipeline.INPUT_TOKEN:
                key = zero_key
            if key in sig.parameters:
                input_kwargs[key] = value

        return func(**input_kwargs)


    def __call__(self, input_obj):

        kwargs = {Pipeline.INPUT_TOKEN: input_obj}

        for func in self.processors:

            output_obj = self._forward(func, kwargs)

            kwargs = {Pipeline.INPUT_TOKEN: output_obj}
            kwargs.update(func.__dict__)

        return kwargs[Pipeline.INPUT_TOKEN]




def _transform_ast_to_cst(graph, indexer, set_sem=False, attach_dummy=True):

    cgraph = au.WriteGraph(graph.name, indexer)

    root_node = graph.node(graph.root)

    # Program is the root of a CST
    program_id = cgraph.write_node("PROGRAM", position=0, node=root_node)

    # Dummy function for global declarations
    init_id = cgraph.write_node("InitFunctionDecl", position=0, assign_to=program_id,
                                node=_dummy_node(root_node))

    for u in graph.in_edges(graph.root):
        if u == graph.root:
            continue
        node = graph.node(u)

        # On default the root of all statements is the dummy function
        # If we encounter a function decl, then the follow up statements
        # are rooted by this function
        root = init_id
        if 'FunctionDecl' in node.label:
            root = cgraph.write_node(node.label, position=0, assign_to=program_id, node=node)

        _transform_function_node(graph, cgraph, root, u, set_sem=set_sem)

    if attach_dummy:
        # Attaches dummy nodes to the graph
        # Should not change the semantic of the CST
        _attach_noop(cgraph)

    return cgraph


def _find_statement_roots(graph, root):
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

def _transform_function_node(graph, cgraph, attach_func, root, set_sem=False):

    # If attach_func == 1, means we use the dummy init function
    if attach_func == 1:
        state_roots = [root]
    else:
        state_roots = _find_statement_roots(graph, root)

    state_roots = sorted(state_roots, key=lambda id: graph.node(id).position)
    attach_roots = []

    for pos, id in enumerate(state_roots):
        node = graph.node(id)
        label = node.label
        nid = cgraph.write_node(label, position=pos+1, assign_to=attach_func, node=node)
        attach_roots.append(nid)

    for i in range(len(state_roots)):
        _transform_state_node(graph, cgraph, attach_roots[i], state_roots[i],
                                set_sem=set_sem)


def _transform_state_node(graph, cgraph, attach_root, root, set_sem=False):

    if set_sem:
        sequence = list(_ast_to_set(graph, root).values())
    else:
        sequence = _ast_to_seq(graph, root)

    for pos in range(len(sequence)):
        ipos = 0 if set_sem else pos
        node = sequence[pos]
        label = node.label
        cgraph.write_node(label, position=ipos+1, assign_to=attach_root, node=node)


def _ast_to_seq(graph, root):
    sequence = []
    Q = [root]
    seen = set([root])

    while len(Q) > 0:
        id = Q[0]
        Q = Q[1:]

        if id == "[SEP]":
            sequence.append(id)
            continue

        node = graph.node(id)
        label = node.label

        if id != root and label == 'IfStmt':
            graph.rewrite_label(id, 'ElseIfStmt')
            continue

        if label == 'CompoundStmt':
            continue

        sequence.append(node)

        neigh = sorted([u for u in graph.in_edges(id) if u not in seen], key=lambda x: graph.node(x).position)
        seen = seen.union(set(neigh))
        Q = neigh + Q

    return sequence


def _ast_to_set(graph, root):
    out = {}

    Q = [root]

    while len(Q) > 0:
        id = Q.pop()

        if id == "[SEP]":
            sequence.append(id)
            continue

        node = graph.node(id)
        label = node.label

        if id != root and label == 'IfStmt':
            graph.rewrite_label(id, 'ElseIfStmt')
            continue

        if label == 'CompoundStmt':
            continue

        if label not in out:
            out[label] = node

        neigh = graph.in_edges(id)
        Q.extend(neigh)

    return out


def _attach_noop(graph):
    Q = [0]

    rgraph = graph.read_graph()

    while len(Q) > 0:
        current = Q.pop()
        edges = rgraph.in_edges(current)

        if len(edges) == 0:
            continue

        graph.write_node("[NOOP]", position=0, assign_to=current,
                         node=_dummy_node(rgraph.node(current)))

        Q.extend(edges)


def _add_node(index, buffer, data, n):
    if n in index:
        return index[n]
    depth = data.depth[n]
    while len(buffer) <= depth:
        buffer.append([[], []])
    cbuffer = buffer[depth]
    id = len(cbuffer[0])
    index[n] = id
    cbuffer[0].append(n)
    return id


def _dummy_node(node):
    dummy_node = au.Node(-1, label="Dummy")

    for key, value in node.__dict__.items():
        if isinstance(value, int):
            dummy = 0
        elif isinstance(value, proto.TokenPosition):
            dummy = proto.TokenPosition()
            dummy.lineStart = 0
            dummy.lineEnd = 0
            dummy.charStart = 0
            dummy.charEnd = 0
        setattr(dummy_node, key, dummy)

    return dummy_node
