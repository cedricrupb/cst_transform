import torch as th
import numpy as np
from sklearn.metrics import roc_auc_score

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
