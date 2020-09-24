import argparse
from flask import Flask, abort, request
from flask_restful import Resource, Api
import os
import uuid
import traceback
import threading
import json
from glob import glob

from pycparser import c_ast
from pycparserext import ext_c_generator
from pycparser import preprocess_file
from pycparserext.ext_c_parser import GnuCParser, FuncDeclExt
from model.run_predict import run_predict
import re

def stripComments(code):
    code = str(code)
    return re.sub('^ *//.*\n?', '', code)

app = Flask(__name__, static_url_path="/static/")
api = Api(app)

allowed_dirs = set(['js', 'css', 'semantic'])
threads = {}

checkpoints = {
    'bmc-ki': ['../checkpoints/bmc', '../labels/tool_order_bmc-ki.json'],
    'sc': ['../checkpoints/compositions', '../labels/tool_order_sc.json'],
    'algorithms': ['../checkpoints/algorithms', '../labels/tool_order_algorithms.json'],
    'tools': ['../checkpoints/tools', '../labels/tool_order_tools.json']
}


def preprocess(path):
    text = preprocess_file(path)
    cparser = GnuCParser()
    ast = cparser.parse(text, path)
    generator = ext_c_generator.GnuCGenerator()
    with open(path, "w") as o:
        o.write(generator.visit(ast))
    return ast


def get_funcs(ast):
    functions = []

    has_init = False

    for decl in ast.ext:
        if isinstance(decl, c_ast.FuncDef):
            func_def = decl.decl.name
            functions.append(func_def)
        elif isinstance(decl, c_ast.Decl) and isinstance(decl.type, FuncDeclExt):
            func_def = decl.name
            functions.append(func_def)
        else:
            has_init = True

    #if has_init:
    functions = ['__init__'] + functions

    return functions


def index_main(keys, strict=False):
    for i, k in enumerate(keys):
        if (strict and k == 'main') or (not strict and 'main' in k.lower()):
            return i
    return 1



def locate_funcs(keys, funcs):

    out = {}

    #align init
    if funcs[0] == '__init__':
        out[keys[0]] = funcs[0]
        keys = keys[1:]
        funcs = funcs[1:]

    kmain = index_main(keys)
    fmain = index_main(funcs, strict=True)

    #align
    left = fmain
    right = len(funcs) - fmain

    p = 0
    for i in range(max(0, kmain-left), min(len(keys), kmain+right)):
        out[keys[i]] = funcs[p]
        p += 1

    return out





def match_att(att_file, ast):
    with open(att_file, "r") as i:
        attention = json.load(i)

    output = {}
    funcs = get_funcs(ast)

    attention = {
        k: v for k , v in attention.items() if 'NOOP' not in k and\
                                        'prediction' not in k
    }

    func_map = locate_funcs(list(attention.keys()), funcs)

    for k, func_name in func_map.items():
        output[func_name] = attention[k]

    with open(att_file, "w") as o:
        json.dump(output, o, indent=4)

def run_clf_predict(pred_dir, req_file):

    base = os.path.abspath(__file__)
    base = os.path.dirname(base)

    check_cfg = checkpoints[checkpoint]

    path = os.path.join(base, check_cfg[0])
    att_file = os.path.join(pred_dir, 'attention.json')

    ast = preprocess(req_file)

    index = os.path.join(base, "../resources/token_clang.json")
    tool_index = os.path.join(base, check_cfg[1])

    P = run_predict(path, req_file, att_file, indexer_path=index,
                    tools=tool_index)

    P = [X[0] for X in sorted(P.items(), key=lambda X: X[1], reverse=True)]
    P = [P[0]]

    match_att(att_file, ast)

    with open(os.path.join(pred_dir, "prediction.json"), "w") as o:
        json.dump(P, o)
    return P, ast

def to_func_ast(ast):
    return c_ast.FileAST(ast)


def to_func(ast):
    functions = {}

    for decl in ast.ext:
        func_def = "__init__"
        if isinstance(decl, c_ast.FuncDef):
            func_def = decl.decl.name
        if func_def not in functions:
            functions[func_def] = []
        functions[func_def].append(decl)

    for k in list(functions):
        functions[k] = to_func_ast(functions[k])
    return functions


def start_prediction(id, pred_dir, req_file):
    P, ast = run_clf_predict(pred_dir, req_file)

    F = to_func(ast)

    generator = ext_c_generator.GnuCGenerator()
    for name, ast in F.items():
        path = os.path.join(pred_dir, name+".c")
        with open(path, "w") as o:
            out = generator.visit(ast)
            o.write(out)



def request_predict(form):
    id = str(uuid.uuid4())

    path = os.path.join(".", "process", id)
    os.makedirs(path)
    file_path = os.path.join(path, "file.c")

    text = form['data']
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    text = stripComments(text)

    with open(file_path, "w") as o:
        o.write(text)

    thread = threading.Thread(
        target=start_prediction,
        args=(id, path, file_path)
    )
    thread.start()

    threads[id] = thread
    return id


@app.route("/")
def index():
    return app.send_static_file('index.html')


@app.route("/<string:type>/<path:path>")
def send_static(type, path):
    if type not in allowed_dirs:
        return abort(404)
    path = os.path.join(type, path)
    path = os.path.normpath(path)

    if not path.startswith(type):
        return abort(404)

    return app.send_static_file(path)


class PredictionTask(Resource):

    def get(self, predict_id):
        path = os.path.join(".", "process", predict_id)
        if not os.path.exists(path):
            return abort(404)

        exc = os.path.join(path, 'exception')
        if os.path.exists(exc):
            with open(exc, "r") as i:
                return {'exception': i.read(), 'finish': True}

        path = os.path.join(".", "process", predict_id)

        state = {
            'attention': os.path.isfile(os.path.join(path, 'attention.json')),
            'pred': os.path.isfile(os.path.join(path, 'prediction.json'))
        }

        finished = True
        for v in state.values():
            finished = finished and v
        state['finish'] = finished

        if not finished and predict_id not in threads:
            return {'exception': "Seem to be an old request!",
                    'finished': True}

        return state

    def put(self):
        return {'request_id': request_predict(
            request.form
        )}


class CFileResource(Resource):

    def get(self, id, func_name=None):
        path = os.path.join(".", "process", id)
        if not os.path.exists(path):
            return abort(404)

        if func_name is None:

            func_names = []

            for p in glob(os.path.join(path, "*.c")):
                b = os.path.basename(p)
                if b == 'file.c':
                    continue
                func_names.append(
                    b.replace(".c", "")
                )
            return {'functions': func_names}
        path = os.path.join(".", "process", id, func_name+".c")
        if not os.path.exists(path):
            return abort(404)

        with open(path, "r") as i:
            return i.read()


class AttentionResource(Resource):
    def get(self, id):
        path = os.path.join(".", "process", id, "attention.json")

        if not os.path.isfile(path):
            return abort(404)

        with open(path, "r") as i:
            return json.load(i)


class PredictionResource(Resource):
    def get(self, id):
        path = os.path.join(".", "process", id, "prediction.json")

        if not os.path.isfile(path):
            return abort(404)

        with open(path, "r") as i:
            return json.load(i)



api.add_resource(PredictionTask, '/api/task/',
                 '/api/task/<string:predict_id>/')
api.add_resource(CFileResource, '/api/cfile/<string:id>/',
                 '/api/cfile/<string:id>/<string:func_name>/')
api.add_resource(AttentionResource, "/api/attention/<string:id>/")
api.add_resource(PredictionResource, "/api/prediction/<string:id>/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Choose a checkpoint from [bmc-ki, algorithms, sc, tools].")

    args = parser.parse_args()
    checkpoint = args.checkpoint

    if checkpoint not in checkpoints:
        print("Checkpoint does not exists: %s" % checkpoint)
        print("Choose a checkpoint from [bmc-ki, algorithms, sc, tools].")
        exit()

    app.run(debug=True)
