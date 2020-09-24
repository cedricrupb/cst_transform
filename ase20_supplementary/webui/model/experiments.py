import torch as th
from torch_geometric import datasets as ds
from torch_geometric import data
from torch_geometric import transforms as trf

import os
import shutil
import random
import json
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score

try:
    from . import data as dt
except ImportError:
    import data as dt



def element_accuracy(prob, labels):

    _, index = prob.topk(1)
    index = index.squeeze()

    s = 0
    for i in range(index.size(0)):
        r = 1 if labels[i, index[i]].item() >= 0.5 else 0
        s += r
    s /= index.size(0)
    return s


def oracle_accuracy(prob, labels):

    _, index = prob.topk(1)
    index = index.squeeze()

    s = 0
    c = 0
    for i in range(index.size(0)):
        r = 1 if labels[i, index[i]].item() >= 0.5 else 0
        s += r
        c += 1 if labels[i].max().item() >= 0.5 else 0
    s /= c
    return s

class OracleMetric:

    def __init__(self):
        self.reset()

    def __call__(self, prob, labels):
        _, index = prob.topk(1)
        index = index.squeeze()

        s = 0
        c = 0
        for i in range(index.size(0)):
            r = 1 if labels[i, index[i]].item() >= 0.5 else 0
            s += r
            c += 1 if labels[i].max().item() >= 0.5 else 0
        self.score += s
        self.nb_eval += c

    def eval(self):
        if self.nb_eval == 0:
            return 0.0
        return self.score / self.nb_eval

    def reset(self):
        self.score = 0
        self.nb_eval = 0



def pairwise_auc(prob, labels):
    prob = prob.numpy()
    labels = labels.numpy()

    s = 0
    for i in range(labels.shape[0]):
        label = labels[i]
        pro = prob[i]
        pos_ix = np.where(label >= 0.5)[0]
        neg_ix = np.where(label < 0.5)[0]

        norm = pos_ix.shape[0] * neg_ix.shape[0]

        if norm == 0:
            s += 1
            continue

        v = 0
        for p in range(pos_ix.shape[0]):
            for n in range(neg_ix.shape[0]):
                if pro[pos_ix[p]] > pro[neg_ix[n]]:
                    v += 1
        s += v/norm
    s /= labels.shape[0]
    return s




def auc_curve(prob, labels):
    prob = prob.numpy()
    labels = labels.numpy()

    try:
        return roc_auc_score(
            labels, prob
        )
    except Exception:
        return 1.0




def create_pw_to_prob(num_classes):
    N = num_classes
    A = th.zeros((int(N*(N-1)/2), num_classes), dtype=th.float)
    B = th.zeros((int(N*(N-1)/2), num_classes), dtype=th.float)
    pw = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            A[pw, i] = 1
            B[pw, j] = 1
            pw += 1

    def pw_to_prob(prob, labels):
        out = th.matmul(prob, A) + th.matmul(1 - prob, B)
        return out, labels
    return pw_to_prob

def ml_transform(logits, labels):

    x = th.nn.Sigmoid()(logits)
    x = x.detach().cpu()
    labels = labels.detach().cpu()

    return x, labels

def mc_transform(logits, labels):

    x = th.nn.Softmax(dim=-1)(logits)
    x = x.detach().cpu()
    labels = labels.detach().cpu()

    return x, labels


class MLCrossEntropy(th.nn.Module):

    def __init__(self, binary_loss=False, weight=None):
        super().__init__()
        self.norm = th.nn.LogSoftmax(dim=-1)
        self.binary_loss = binary_loss
        self.weight = weight
        self.use_weight = weight is not None

    def forward(self, logit, y):

        if self.binary_loss:
            y = y / y.sum(dim=-1, keepdim=True)
            y[y != y] = 0

        x = self.norm(logit)
        w = 1
        if self.use_weight:
            w = self.weight.to(logit.device)

        nll = (-w * y * x).sum(dim=-1)
        nll = nll.mean(dim=0)

        return nll




class Experiment():

    def __init__(self, cache_path, num_features, num_classes, loss,
                 num_edge_features=-1):
        self.cache_path = cache_path
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_edge_features = num_edge_features
        self.loss = loss
        self.eval_metrics = {}


class MeanMetric:

    def __init__(self, eval_func):
        self.eval_func = eval_func
        self.reset()

    def __call__(self, logits, labels):
        self.score += self.eval_func(logits, labels)
        self.nb_eval += 1

    def eval(self):
        if self.nb_eval == 0:
            return 0.0
        return self.score / self.nb_eval

    def reset(self):
        self.score = 0
        self.nb_eval = 0


class ConstantMetric:

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, logits, labels):
        pass

    def eval(self):
        return self.constant

    def reset(self):
        pass


class TransformMetric:

    def __init__(self, metric, transforms=None):
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = []

        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.metric = metric

    def __call__(self, logits, labels):

        for transform in self.transforms:
            logits, labels = transform(logits, labels)

        return self.metric(logits, labels)

    def eval(self):
        return self.metric.eval()

    def reset(self):
        self.metric.reset()



class TP_FP_FN_Metric:

    def __init__(self, eval_func):
        self.eval_func = eval_func
        self.reset()

    def __call__(self, logits, labels):
        tp, fp, fn = self.eval_func(logits, labels)
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


class PrecisionMetric(TP_FP_FN_Metric):

    def eval_(tp, fp):
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def eval(self):
        return PrecisionMetric.eval_(self.tp, self.fp)



class RecallMetric(TP_FP_FN_Metric):

    def eval_(tp, fn):
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def eval(self):
        return RecallMetric.eval_(self.tp, self.fn)


class F1Metric(TP_FP_FN_Metric):

    def eval_(tp, fp, fn):
        precision = PrecisionMetric.eval_(tp, fp)
        recall = RecallMetric.eval_(tp, fn)
        if precision + recall == 0:
            return 0.0
        return 2*(precision * recall) / (precision + recall)

    def eval(self):
        return F1Metric.eval_(self.tp, self.fp, self.fn)


class F1PrecRecMetric(TP_FP_FN_Metric):

    def eval_(tp, fp, fn):
        precision = PrecisionMetric.eval_(tp, fp)
        recall = RecallMetric.eval_(tp, fn)
        if precision + recall == 0:
            return 0.0
        return 2*(precision * recall) / (precision + recall), precision, recall

    def eval(self):
        return F1PrecRecMetric.eval_(self.tp, self.fp, self.fn)


class ReferenceMetric:

    def __init__(self, metric, select_ix=-1):
        self.metric = metric
        self.ix = select_ix

    def __call__(self, logits, labels):
        pass

    def eval(self):
        res = self.metric.eval()

        if self.ix >= 0:
            return res[self.ix]
        return res

    def reset(self):
        self.metric.reset()


def logit_accuracy(logits, labels):

    output = th.nn.functional.log_softmax(logits, dim=-1)

    labels = labels.detach().cpu()
    output = output.detach().cpu()

    pred = output.max(dim=1)[1]
    return pred.eq(labels).sum().item() / labels.size(0)


def logit_ml_accuracy(logits, labels):

    output = th.sigmoid(logits)

    labels = labels.detach().cpu()
    output = output.detach().cpu()

    pred = output.round()
    return pred.eq(labels).sum().item() / (labels.size(0) * labels.size(1))


def reduce(L, reduction):

    if reduction == 'mean':
        return L.mean()
    if reduction == 'sum':
        return L.sum()
    return L


class HingeLoss(th.nn.Module):

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.margin = margin

    def forward(self, p, y):

        mask = y != 0

        y = y.float()

        p = th.nn.Tanh()(p)
        L = th.nn.functional.relu(self.margin - y * p)
        L = L[mask]
        return reduce(L, self.reduction)


def load_labels(name):

    pattern = "./labels/labels_%s.json" % name

    with open(pattern, "r") as i:
        return json.load(i)


def attach_transform(num_classes, labels, gold_labels):

    def f(data):
        data.name = data.name.replace("/", "_").replace(".", "_")

        label = [0.0]*num_classes
        if data.name in labels:
            label = labels[data.name]

        gold = [0.0]*num_classes
        if data.name in gold_labels:
            gold = gold_labels[data.name]

        return data, label, gold

    return f

def prep_transform(data):
    data, y, gold = data

    data = dt.load_prepped(data)
    data = dt.prepped_to_tensor(data)
    data.y = th.tensor(y).unsqueeze(0).float()
    data.gold = th.tensor(gold).unsqueeze(0).float()

    return data

def prep_transform_wo_labels(data):

    data = dt.load_prepped(data)
    data = dt.prepped_to_tensor(data)
    return data

def unprep_transform_wo_labels(data):

    data = dt.prep_data(data)
    data = dt.prepped_to_tensor(data)
    return data


def unprep_transform(data):

    data, y, gold = data

    name = data.name
    data = dt.prep_data(data)
    data = dt.prepped_to_tensor(data)
    data.name = name
    data.y = th.tensor(y).unsqueeze(0).float()
    data.gold = th.tensor(gold).unsqueeze(0).float()
    return data


class SVExperiment(Experiment):

    def __init__(self, cache_path, name, cv_number=0,
                 max_degree=-1, embed=False, edge_type=False,
                 prepared=True):

        label_key = name[3:]

        self.labels = load_labels(label_key)
        self.rank_mode = 'rank' in name

        num_classes = len(self.labels['classes'])
        num_features = 366

        super().__init__(
            cache_path, num_features, num_classes,
            th.nn.BCEWithLogitsLoss(), 0)

        with open('./labels/labels_correct_prob.json', 'r') as i:
            self.gold_labels = json.load(i)

        data_transform = attach_transform(num_classes, self.labels, self.gold_labels)

        def transform(data):
            graph = dt.to_proto(data)
            graph = data_transform(graph)
            if prepared:
                graph = prep_transform(graph)
            else:
                graph = unprep_transform(graph)
            return graph

        self.dataset = dt.ContextLMDBDataset(self.cache_path, "cpg-data",
                                             transform=transform)
        self._init_datasets(cv_number)
        self._init_metrics()

    def _init_datasets(self, cv_number):
        dataset = self.dataset

        test_length = len(dataset) // 10
        start_idx = cv_number * test_length
        end_idx = min([(cv_number + 1) * test_length, len(dataset)])
        idx_range = th.tensor(list(range(start_idx, end_idx, 1)))
        nidx_range = th.tensor([i for i in range(len(dataset)) if i < start_idx or i >= end_idx])

        self.test_dataset = dataset[idx_range]
        self.train_dataset = dataset[nidx_range]

        dataset = self.train_dataset
        self.val_dataset = dataset[:len(dataset) // 10]
        self.train_dataset = dataset[len(dataset) // 10:]

    def _init_metrics(self):

        transforms = [ml_transform]

        if self.rank_mode:
            pw = create_pw_to_prob(self.num_classes)
            transforms.append(pw)

        metric = MeanMetric(oracle_accuracy)
        metric = TransformMetric(metric, transforms=transforms)
        metric.require_gold = True

        self.eval_metrics = {
            'accuracy': metric,
            'pw_acc': MeanMetric(logit_ml_accuracy),
            'loss': ConstantMetric(0.0)
        }


def prepare_weight(labels):

    num_classes = len(labels['classes'])

    pos = [0]*num_classes
    neg = [0]*num_classes

    for k, V in labels.items():
        if k == 'classes':
            continue
        for i, v in enumerate(V):
            if v >= 0.5:
                pos[i] += 1
            else:
                neg[i] += 1
    weight = [neg[i]/pos[i] for i in range(num_classes)]

    print("Positives: %s" % pos)
    print("Negatives: %s " % neg)
    print(weight)

    return th.tensor(weight)




class SVPreparedExperiment(Experiment):

    def __init__(self, cache_path, name, cv_number=0,
                 max_degree=-1, embed=False, edge_type=False,
                 class_weighted=False,
                 prepared=True):

        label_key = name[4:]
        self.mc = name[0] == 'm'

        self.labels = load_labels(label_key)
        self.rank_mode = 'rank' in name

        num_classes = len(self.labels['classes'])
        num_features = 366
        weight = None

        if class_weighted:
            weight = prepare_weight(self.labels)

        if self.mc:
            loss = MLCrossEntropy(
                binary_loss=True, weight=weight
            )
        else:
            loss = th.nn.BinaryCrossEntropy(
                weight=weight
            )

        super().__init__(
            cache_path, num_features, num_classes,
            loss, 0)

        gold_path = "./labels/labels_%s.json" % (
            label_key.replace("prob", "correct_prob")
                      .replace("_non", "")
        )

        with open(gold_path, 'r') as i:
            self.gold_labels = json.load(i)

        data_transform = attach_transform(num_classes, self.labels, self.gold_labels)

        def transform(data):
            graph = dt.to_proto(data)
            graph = data_transform(graph)
            if prepared:
                graph = prep_transform(graph)
            else:
                graph = unprep_transform(graph)
            return graph


        self._init_datasets(transform, non=label_key.endswith("non"))
        self._init_metrics()

    def _init_datasets(self, transform, non=False):
        train_path = os.path.join(self.cache_path, 'train_non' if non else 'train')
        val_path = os.path.join(self.cache_path, "validate")
        test_path = os.path.join(self.cache_path, "test")
        self.dataset = dt.ContextLMDBDataset(train_path, "cpg-data",
                                             transform=transform,
                                             shuffle=True)
        self.train_dataset = self.dataset
        self.val_dataset = dt.ContextLMDBDataset(val_path, "cpg-data",
                                                 transform=transform)
        self.test_dataset = dt.ContextLMDBDataset(test_path, "cpg-data",
                                                  transform=transform)


    def _init_metrics(self):

        transforms = [mc_transform if self.mc else ml_transform]

        if self.rank_mode:
            pw = create_pw_to_prob(self.num_classes)
            transforms.append(pw)

        metric = OracleMetric()
        metric = TransformMetric(metric, transforms=transforms)
        metric.require_gold = True

        roc_metric = MeanMetric(auc_curve)
        roc_metric = TransformMetric(roc_metric, transforms=transforms)
        roc_metric.require_gold = True

        pw_metric = MeanMetric(pairwise_auc)
        pw_metric = TransformMetric(pw_metric, transforms=transforms)
        pw_metric.require_gold = True

        self.eval_metrics = {
            'accuracy': metric,
            'roc_auc': roc_metric,
            'pw_auc': pw_metric,
            'pw_acc': MeanMetric(logit_ml_accuracy),
            'loss': ConstantMetric(0.0)
        }



def get_experiment(name, cache_path, embed=False, edge_types=False):

    cv_number = 0

    if '_' in name:
        cv_name, cv_number = name.rsplit("_", 1)
        try:
            cv_number = int(cv_number)
            name = cv_name
        except ValueError:
            cv_number = 0

    if name.startswith('sv-'):
        return SVExperiment(
            cache_path, name, cv_number=cv_number
        )

    if name.startswith('psv-') or name.startswith('msv-'):
        return SVPreparedExperiment(
            cache_path, name, cv_number=cv_number
        )

    raise Exception("Unknown experiment %s" % name)
