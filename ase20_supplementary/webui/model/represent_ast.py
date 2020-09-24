import argparse
import json
from tqdm import tqdm
from pycparser import preprocess_file, c_ast
from pycparserext.ext_c_parser import GnuCParser as CParser
import re

try:
    from . import treedata_pb2 as proto
except ImportError:
    import treedata_pb2 as proto

cparser = CParser()

class NodeInterface():

    def __init__(self):
        self.label = "None"
        self.row = 0
        self.column = 0


def successor(node):
    if not isinstance(node, c_ast.Node):
        return []

    out = []
    out.extend(node.children())

    if node.attr_names:
        for attr in node.attr_names:
            out.append((attr, getattr(node, attr)))

    return out


def traverse_ast(ast):

    count = 1

    sequence = [[0, 'root', ast, 0, successor(ast), 0]]

    while len(sequence) > 0:
        id, name, current, depth, succesor, it = sequence[-1]

        if it >= len(succesor):
            parent = None
            if len(sequence) > 1:
                pid, _, parent, _, _, _ = sequence[-2]
            yield parent, pid, name, current, id, depth
            sequence.pop()
        else:
            succ = succesor[it]
            sequence[-1][-1] += 1
            id = count
            count += 1
            sequence.append(
                [id, succ[0], succ[1], depth + 1, successor(succ[1]), 0]
            )


def node_to_str(node, showcoord=False):

    if isinstance(node, c_ast.Node):
        txt = str(type(node))
        txt = txt.replace("<class \'pycparser.c_ast.", "").replace("\'>", "").replace("<class \'pycparserext.ext_c_parser.", "")

        if showcoord and node.coord:
            txt = txt + ":%d" % node.coord.line
            if node.coord.column:
                txt = txt + ":%d" % node.coord.column
        return txt

    if isinstance(node, list):
        if len(node) == 0:
            return "EMPTY"
        if len(node) == 1:
            return str(node[0])
    return str(node)


def _create_node_interface(node):
    inter = NodeInterface()
    inter.label = node_to_str(node)
    if isinstance(node, c_ast.Node) and node.coord:
        inter.row = node.coord.line
        if node.coord.column:
            inter.column = node.coord.column
    else:
        inter.column = 0
        inter.row = 0
    return inter


def to_dot(traversal, node_interface=_create_node_interface):

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




def _add_proto_node(tree, node, id, depth, trans, indexer, use_row=False,
                    node_interface=_create_node_interface):
    node = node_interface(node)
    if id not in trans:
        trans[id] = trans['count']
        trans['count'] += 1
        label = node.label
        label_id = indexer.index('ast', label)
        tree.nodes.append(label_id)
        tree.depth.append(depth)
        position = node.column
        tree.position.append(position)

    if use_row:
        tree.position[trans[id]] = node.row
    return trans[id]


def match_line(parent, child):

    if hasattr(parent, 'coord') and parent.coord and hasattr(child, 'coord') and child.coord:
        return parent.coord.line == child.coord.line

    return False


def to_proto(traversal, indexer, node_interface=_create_node_interface):

    tree = proto.AnnotatedTree()
    trans = {'count': 0}

    for parent, pid, attr_name, child, cid, depth in traversal:
        if parent is None or child is None or node_to_str(child) == 'EMPTY':
            continue
        reindex_child = not match_line(parent, child)
        cid = _add_proto_node(tree, child, cid, depth, trans, indexer, use_row=reindex_child, node_interface=node_interface)
        pid = _add_proto_node(tree, parent, pid, depth - 1, trans, indexer, node_interface=node_interface)
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


def proto_to_traversal_0(pb):

    for i in range(len(pb.assignment)):
        cid = pb.from_node[i]
        pid = pb.assignment[i]
        parent = str(pb.nodes[pid])
        child = str(pb.nodes[cid])
        attr_name = str(pb.assign_attr[i])
        depth = pb.depth[cid]
        yield parent, pid, attr_name, child, cid, depth



def _line_filter(line):
    if not line:
        return False
    return True


def _clean_line(line):
    line = line.strip()
    line = re.sub('\s+', ' ', line)
    return line


def clean_code(code):

    m = re.compile('(\/\*([\r\n]+\*[^\/]*)*[^\/]*\/|(\/\/|\#).*)')
    code = m.sub("", code)

    return ' '.join(
        [
            cl for cl in (_clean_line(ll) for ll in code.splitlines())
            if _line_filter(cl)
        ]
    )

def prepare_code(code):
    m = re.compile('extern void \*?\(\*__volatile[^\)]*\)[^;]*;')
    code = m.sub("", code)

    code = code.replace("__signed__", "").replace("const typeof", "typeof")

    return code


def parse_ast(file_path, clean=False):
    global cparser

    text = preprocess_file(file_path)

    text = prepare_code(text)

    if clean:
        text = clean_code(text)

    return cparser.parse(text, file_path)
