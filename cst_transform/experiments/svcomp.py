import os
import torch as th
import json

try:
    from metrics import MeanMetric, TransformMetric, ConstantMetric, OracleMetric
    from metrics import logit_ml_accuracy, auc_curve, pairwise_auc
    from loss import MLCrossEntropy
except ImportError:
    from .metrics import MeanMetric, TransformMetric, ConstantMetric, OracleMetric
    from .metrics import logit_ml_accuracy, auc_curve, pairwise_auc
    from .loss import MLCrossEntropy


# We like to import the dataset from a sibling package
if __package__ is not None:

    # We try a relative import
    from .. import data as dt

else:
    # Now we have to explicity import the module
    import sys

    # Append parent directory to system path
    sys.path.insert(0,
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    import data as dt


class Experiment():

    def __init__(self, cache_path, num_features, num_classes, loss):
        self.cache_path = cache_path
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss
        self.eval_metrics = {}


class SVExperiment(Experiment):
    """
        This experiment describes the main setup
        for SVComp experiments.

        To enable user-friendly experiment setups,
        we assume the following file structure for data_path:

        data_path/
           |------- train/ -> LMDB Train dataset
           |------- validate/ -> LMDB Validation dataset
           |------- test/ -> LMDB Test dataset
           |------- vocabulary.json -> Vocabulary used for parsing
           |------- labels.json -> Training labels
            `------ gold_labels.json -> Gold labels (might be the same as labels.json)

        Options:
         data_path - Path to the dataset
         train_mode - Mode to train the dataset selected from [binary, multi, rank] (Default: multi)
         class_weighted - Whether to weight labels with respect to dataset statistics (Default: False)
         prepared - Whether CSTCollate transform was used during dataset creation (Default: True)

        Hint: Rank mode requires a different label format then standard.
              If trained in rank mode, the dataset is handled as a multiple binary
              classification problems. However, during testing we transform
              the binary problems into a ranking.

    """
    def __init__(self,
                 data_path,
                 train_mode = "multi",
                 class_weighted = False,
                 prepared = True):

        self._init_paths(data_path)
        self._validate_data_convention()

        self._load_labels()

        num_classes = len(self.labels['classes'])
        num_features = self._num_features()

        weight = None
        if class_weighted:
            weight = prepare_weight(self.labels)

        if train_mode == 'multi':
            loss = MLCrossEntropy(
                binary_loss=True, weight=weight
            )
        else:
            loss = th.nn.BinaryCrossEntropy(
                weight=weight
            )

        self.train_mode = train_mode
        self.rank_mode = train_mode == 'rank'

        super().__init__(
            data_path, num_features, num_classes,
            loss)

        data_transform = attach_transform(num_classes, self.labels, self.gold_labels)

        def transform(data):
            graph = dt.to_proto(data)
            graph = data_transform(graph)
            if prepared:
                graph = prep_transform(graph)
            else:
                graph = unprep_transform(graph)
            return graph

        self._init_datasets(transform)
        self._init_metrics()

    def _init_paths(self, data_path):
        self.train_path = os.path.join(data_path, "train")
        self.validate_path = os.path.join(data_path, "validate")
        self.test_path = os.path.join(data_path, "test")
        self.vocabulary_path = os.path.join(data_path, "vocabulary.json")
        self.label_path = os.path.join(data_path, "labels.json")
        self.gold_label_path = os.path.join(data_path, "gold_labels.json")

    def _validate_data_convention(self):

        assert os.path.exists(self.train_path) and os.path.isdir(self.train_path),\
                "Missing training path at %s" % self.train_path
        assert os.path.exists(self.validate_path) and os.path.isdir(self.validate_path),\
                "Missing validate path at %s" % self.validate_path
        assert os.path.exists(self.test_path) and os.path.isdir(self.test_path),\
                "Missing testing path at %s" % self.test_path

        assert os.path.exists(self.vocabulary_path), "Missing vocabulary at %s" % self.vocabulary_path
        assert os.path.exists(self.label_path), "Missing labels at %s" % self.label_path

        if not os.path.exists(self.gold_label_path):
            self.gold_label_path = self.label_path

    def _load_labels(self):
        with open(self.label_path, "r") as i:
            self.labels = json.load(i)

        with open(self.gold_label_path, "r") as i:
            self.gold_labels = json.load(i)

    def _num_features(self):
        with open(self.vocabulary_path, "r") as i:
            vocab = json.load(i)
        return vocab['ast']['_counter_']


    def _init_datasets(self, transform):
        self.dataset = dt.ContextLMDBDataset(self.train_path, "cpg-data",
                                             transform=transform,
                                             shuffle=True)
        self.train_dataset = self.dataset
        self.val_dataset = dt.ContextLMDBDataset(self.validate_path,
                                                 "cpg-data",
                                                 transform=transform)
        self.test_dataset = dt.ContextLMDBDataset(self.test_path, "cpg-data",
                                                  transform=transform)


    def _init_metrics(self):

        transforms = [mc_transform if self.train_mode == "multi" else ml_transform]

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
