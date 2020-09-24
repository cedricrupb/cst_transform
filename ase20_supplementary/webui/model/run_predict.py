import argparse
import os
import json

import torch
import torch as th

try:
    from experiments import Experiment, unprep_transform_wo_labels
    from models import create_model
    import clang as CLANG
    import represent_ast as ra
    import preprocess_clang as pc
    import utils
except Exception:
    from .experiments import Experiment, unprep_transform_wo_labels
    from .models import create_model
    from . import clang as CLANG
    from . import represent_ast as ra
    from . import preprocess_clang as pc
    from . import utils

num_features = 366
num_classes = 3

checkpoints = {
    'bmc-ki': ['../../checkpoints/bmc', '../../labels/tool_order_bmc-ki.json'],
    'sc': ['../../checkpoints/compositions', '../../labels/tool_order_sc.json'],
    'algorithms': ['../../checkpoints/algorithms', '../../labels/tool_order_algorithms.json'],
    'tools': ['../../checkpoints/tools', '../../labels/tool_order_tools.json']
}


def _prepare_model(args_path, config_path, num_classes):
    with open(args_path) as i:
        args = json.load(i)

    with open(config_path) as i:
        model_cfg = json.load(i)

    experiment = Experiment(
        None, num_features, num_classes, None
    )

    model = create_model(
        args['model_type'], experiment, model_cfg
    )

    return model



def load_model(model_dir, num_classes):
    if not os.path.exists(model_dir):
        raise ValueError("Model path %s does not exists." % model_dir)

    model_path = os.path.join(model_dir, 'model.pt')
    args_path = os.path.join(model_dir, 'args.json')
    config_path = os.path.join(model_dir, 'config.json')

    model = _prepare_model(args_path, config_path, num_classes)
    param = torch.load(model_path, map_location=torch.device("cpu") )
    model.load_state_dict(param)
    model.eval()

    model.return_attach = True

    return model


def parse_example(path, tokenIndex):

    traversal = CLANG.clang_to_traversal(path, truncate_level=2)
    node_interface = CLANG._clang_node_interface

    tmpIndexer = utils.MultiIndexer()

    proto = ra.to_proto(
        traversal, tmpIndexer, node_interface=node_interface
    )

    _, proto = pc.preprocess(
        -1, proto, tmpIndexer, tokenIndex, set_sem=True
    )

    return unprep_transform_wo_labels(proto), proto


def assign_class(pred, classes, ret_full=False):
    x = th.nn.Softmax(dim=1)(pred)
    _, index = x.topk(1)

    index = index.squeeze().item()

    if ret_full:
        x = x.squeeze()
        R = {}
        for i, c in enumerate(classes):
            R[c] = x[i].item()
            print(
                "%s: %f" % (c, x[i].item())
            )
        return R

    return [classes[index]]


def predict(model, data, classes):

    pred = model(
        data.x, data.depth_mask, data.edge_index, data.position
    )

    att = None
    if isinstance(pred, tuple):
        pred, att = pred

    pred = assign_class(pred, classes, ret_full=True)
    return pred, att


def _simplify_assign(D):
    D = D['children']

    for functions in D.values():
        del functions['label']
        del functions['pos']
        if 'children' in functions:
            for k, v in functions['children'].items():

                v['ast'] = {}

                if 'children' in v:
                    for _k, _v in v['children'].items():
                        v['ast'][_k] = _v['attention']

                    del v['children']

                functions[k] = v
            del functions['children']

    return D


def assign_attention(data, indexer, attention):

    output = {}
    index = {}

    def ix(depth, p):

        if depth == 0:
            return output

        return index[(depth, p)]

    for d in range(1, 4):

        m = data.depth_mask == d
        x = data.x[m]
        e = data.edge_index[m]
        pos = data.position[m]
        A = attention[3 - d].squeeze()

        for p in range(x.size(0)):
            cx = x[p].item()
            clabel = indexer.reverse('ast' ,cx)
            ce = e[p].item()
            ca = A[p].item()
            cp = pos[p].item()

            parent = ix(d - 1, ce)
            if 'children' not in parent:
                parent['children'] = {}
            eix = "%s:%d:%d" % (clabel, p, cp)
            parent['children'][eix] = {
                'label': clabel,
                'pos': cp,
                'attention': ca
            }
            index[(d, p)] = parent['children'][eix]

    return _simplify_assign(output)






def run_predict(model_dir, in_file, out_file=None, ret_ast=False,
                embed_file=None,
                indexer_path="token_clang.json",
                tools="./labels/tool_order.json"):

    indexer = utils.MultiIndexer()
    with open(indexer_path, "r") as i:
        indexer.from_json_io(i)

    with open(tools, "r") as i:
        classes = json.load(i)

    model = load_model(model_dir, len(classes))
    data, ast = parse_example(in_file, indexer)

    P, att = predict(model, data, classes)

    if out_file is not None:
        ratt = assign_attention(
            data, indexer, att
        )

        ratt['prediction'] = P

        with open(out_file, "w") as o:
            json.dump(ratt, o, indent=4)


    if embed_file is not None:
        embedding = att[-1].squeeze().detach().cpu().numpy()

        embedding = {
            str(i): float(embedding[i]) for i in range(embedding.shape[0])
        }

        with open(embed_file, "w") as o:
            json.dump(embedding, o, indent=4)

    if ret_ast:
        return P, ast

    return P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("in_file")

    parser.add_argument("--embed_file", type=str)

    args = parser.parse_args()
    checkpoint = args.checkpoint

    if checkpoint not in checkpoints:
        print("Checkpoint does not exists: %s" % checkpoint)
        print("Choose a checkpoint from [bmc-ki, algorithms, sc, tools].")
        exit()

    base = os.path.abspath(__file__)
    base = os.path.dirname(base)

    check_cfg = checkpoints[checkpoint]

    path = os.path.join(base, check_cfg[0])
    index = os.path.join(base, "../../resources/token_clang.json")
    tool_index = os.path.join(base, check_cfg[1])

    print(
        run_predict(
            path, args.in_file, None,
            indexer_path=index,
            embed_file=args.embed_file,
            tools=tool_index
        )
    )
