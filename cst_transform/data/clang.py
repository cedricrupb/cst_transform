import argparse
import os
import shutil
import subprocess as sp

import re

try:
    import ast_utils as au
except Exception:
    from . import ast_utils as au

from time import time
from tqdm import tqdm


builtin = set([
    'int', 'char', 'long', 'short', 'float', 'double'
])

line_pattern = re.compile("[\|\s`]+\-")
token_pattern = re.compile("[^\s'<>]+|'[^']*'|<?<[^<>]+>>?")
pos_pattern = re.compile("[^\s<>,]+")

def _refine_int(obj):
    val = obj.tokens[1]
    while val[-1].lower() in ['u', 'l']:
        val = val[:-1]
    val = int(val)
    pos_sign = val >= 0
    val = abs(val)

    out = obj.type
    if val <= 1:
        out = 'BitLiteral'
    elif val < 128:
        out = 'ByteLiteral'
    elif val < 32768:
        out = 'ShortLiteral'
    elif val >= 2147483648:
        out = 'LongLiteral'
    obj.type = ("N_" if not pos_sign else "") + out


def _test_op(name):

    for test in ['struct', '[', 'enum', '_t', '_T', '(', 'union']:
        if test in name:
            return True

    spli = name.split("_")[1:-1]

    if spli[0] == 'void *':
        return False

    for sp in spli:
        if '*' in sp:
            return True

    return False

def _reduce_op_name(name):

    if _test_op(name):
        sp = name.split("_")
        name = '_'.join([sp[0], sp[-1]])
    sp = name.split('_')
    if len(sp) > 2:
        if sp[-2] == 'valu':
            name = '_'.join([sp[0], sp[-3], sp[-2], sp[-1]])
        else:
            name = '_'.join([sp[0], sp[-2], sp[-1]])

    else:
        name = '_'.join([sp[0], sp[-1]])

    return name


def _refine_op(obj):
    obj.type = '_'.join([obj.type]+[t[1:-1] for t in obj.tokens if t not in ['prefix', 'postfix', 'cannot', 'overflow']])
    obj.type = _reduce_op_name(obj.type)

def _refine_cast(obj):
    obj.type = obj.tokens[1]

def _refine_ref(obj):
    if obj.tokens[0] == 'Function':
        obj.type = "FunctionRefExpr"
    elif obj.tokens[0] == 'lvalue':
        type = obj.tokens[2][1:-1]
        pointer = type.endswith(" *")
        if pointer:
            type = type[:-2]
        ctype = type.split(" ")
        if ctype[0] != 'struct' and ctype[-1] in builtin:
            type = ''.join([(t[0].upper() + t[1:]) for t in ctype])
            obj.type = type + ("Pointer" if pointer else "") + "RefExpr"


def _refine_func_decl(obj):
    if obj.tokens[0] == 'main':
        obj.type = "MainFunctionDecl"

refine_switch = {
    'IntegerLiteral': _refine_int,
    'UnaryOperator': _refine_op,
    'BinaryOperator': _refine_op,
    'TernaryOperator': _refine_op,
    'ImplicitCastExpr': _refine_cast,
    'DeclRefExpr': _refine_ref,
    'FunctionDecl': _refine_func_decl
}


def _check_clang():
    if shutil.which("clang") is None:
        raise ValueError("Clang compiler is not installed.")


def read_stdout(process):
    with process as proc:
        while True:
            line = proc.stdout.readline().rstrip()
            if not line:
                break
            yield line


def _run_clang_ast(path_to_file):
    _check_clang()

    cmd = [
        'clang', '-cc1', '-Wno-everything', '-ast-dump', path_to_file
    ]

    p = sp.Popen(cmd, stdout=sp.PIPE,  universal_newlines=True)

    return read_stdout(p)


def _parse_line(line):

    match = line_pattern.search(line)

    if match is None:
        return 0, 0, 0, line

    match_end = match.span()[1]
    cmd = 1 if line[match_end - 2] == '`' else 0
    pipe = match_end - 2
    start_pos = match_end

    return pipe, cmd, start_pos, line[start_pos:]


class RangeObj:

    def __repr__(self):
        return "[%d:%d, %d:%d]" % (self.from_line, self.from_col, self.to_line, self.to_col)


class ASTObject:

    def __init__(self, type, memory_addr, range=None, tokens=[]):
        self.type = type
        self.memory_addr = memory_addr
        self.range = range
        self.tokens = tokens

    def __repr__(self):
        return "Type: %s, Memory: %s, Range: %s, Tokens: %s" % (str(self.type), str(self.memory_addr),
                                                                str(self.range), str(self.tokens))


def parse_loc(txt):

    if not txt.startswith("<"):
        return None

    obj = RangeObj()

    if 'invalid sloc' in txt:
        obj.from_line = -1
        obj.from_col = -1
        obj.to_line = -1
        obj.to_col = -1
        return obj

    F = pos_pattern.findall(txt)
    if len(F) == 0:
        return None
    if len(F) == 1:
        F.append(F[0])

    pos = []
    for f in F:
        sp = [int(s) for s in f.split(':')[1:]]
        if len(sp) == 1:
            pos.append((-1, sp[0]))
        else:
            pos.append((sp[0], sp[1]))

    obj.from_line = pos[0][0]
    obj.from_col = pos[0][1]
    obj.to_line = pos[1][0]
    obj.to_col = pos[1][1]

    return obj


def _tokenize(line):

    return token_pattern.findall(line)



def _parse_line_to_obj(line):

    tokens = _tokenize(line)

    if len(tokens) < 3:
        return ASTObject(type='_'.join(tokens),
                         memory_addr=None,
                         range=None,
                         tokens=[])

    range = parse_loc(tokens[2])
    six = 2 if range is None else 3

    args = {
        'type': tokens[0],
        'memory_addr': tokens[1],
        'range': range,
        'tokens': tokens[six:]
    }

    return ASTObject(**args)


def _remove_memory_tokens(obj):

    if obj.type.endswith("Decl"):
        if obj.tokens[0] == 'prev':
            obj.tokens = [obj.tokens[0]] + obj.tokens[4:]
        else:
            obj.tokens = obj.tokens[1:]
    elif obj.type == 'DeclRefExpr':
        if obj.tokens[1] == 'Function':
            obj.tokens = [obj.tokens[1]] + obj.tokens[3:]
        elif obj.tokens[2].endswith('Var'):
            obj.tokens = obj.tokens[1:2] + obj.tokens[4:]

    return obj


def _refine_ast_types(obj):

    if obj.type in refine_switch:
        refine_switch[obj.type](obj)

    return obj


def _remove_ordinary_nodes_create():

    D = set([])

    def f(pid, cid, rid, obj):
        if pid in D:
            D.add(cid)
            return None
        if pid == 0 and (obj.range is None or obj.range.from_line == 0)\
            and len(obj.tokens) > 0 and obj.tokens[0] == 'implicit':
            D.add(cid)
            return None
        if rid > 0 and rid in D:
            D.remove(rid)
        return obj

    return f


def _range_propagate():

    D = {}

    def f(pid, cid, rid, obj):
        D[cid] = obj.range
        range = D[pid]
        if range is not None and obj.range is not None:
            obj.range.from_line = max(obj.range.from_line, range.from_line, 0)
            obj.range.to_line = max(obj.range.to_line, obj.range.from_line, 0)
            obj.range.from_col = max(obj.range.from_col, 0)
            obj.range.to_col = max(obj.range.to_col, 0)
        if rid > 0:
            del D[rid]
        return obj
    return f


def _parse_output(clang_out):

    id_count = 0
    pipes = {}

    for pos, line in enumerate(tqdm(clang_out)):

        pipe, cmd, pos, line = _parse_line(line)
        if line == '<<<NULL>>>':
            continue
        id = id_count
        id_count += 1
        pipes[pos] = id
        pid = pipes[pipe]
        if cmd:
            yield pid, id, pid, _parse_line_to_obj(line)
            del pipes[pipe]
        else:
            yield pid, id, -1, _parse_line_to_obj(line)


def _preprocess(pid, cid, rid, obj, preprocessor):

    for prep in preprocessor:
        try:
            obj = prep(pid, cid, rid, obj)
        except Exception:
            obj = prep(obj)
        if obj is None:
            return None
    return obj


def _traverse_clang_out(cout, truncate_level=0):

    preprocessor = [_range_propagate()]
    if truncate_level >= 1:
        preprocessor.append(_remove_memory_tokens)
    if truncate_level >= 2:
        preprocessor.append(_remove_ordinary_nodes_create())
        preprocessor.append(_refine_ast_types)

    buffer = {}

    for pid, cid, rid, ast_obj in _parse_output(cout):
        ast_obj = _preprocess(pid, cid, rid, ast_obj, preprocessor)
        if ast_obj is None:
            continue
        buffer[cid] = [ast_obj, -1]
        parent, depth = buffer[pid]
        child = buffer[cid]
        child[1] = depth + 1
        if rid > 0:
            del buffer[rid]
        yield parent, pid, 'ast', child[0], cid, child[1]


def _clang_node_interface(node):
    inter = au.NodeInterface()

    if not isinstance(node, ASTObject):
        inter.label = str(node)
        return inter

    inter.label = node.type
    if node.range is not None:
        inter.row = node.range.from_line
        inter.column = node.range.from_col
        inter.range = (node.range.from_line,
                       node.range.from_col,
                       node.range.to_line,
                       node.range.to_col)

    return inter



def clang_to_traversal(file_path, truncate_level=0):

    start_time = time()
    cout = _run_clang_ast(file_path)

    print("CLANG Runtime: %d" % (time() - start_time ))

    return _traverse_clang_out(cout, truncate_level)
