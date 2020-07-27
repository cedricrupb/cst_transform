
from .metrics import (
    MeanMetric,
    ConstantMetric,
    OracleMetric,
    TransformMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    F1PrecRecMetric,
    ReferenceMetric
)

from .svcomp import (
    SVExperiment
)


def get_experiment(exp_type, data_path):
    """
    Configures experiment by name

    E.g. sv-bench_mc_cw
          A sv-bench experiment in multi-class setting and
          class weighting in loss.

    """

    options = exp_type.split("_")

    exp_name = options[0]

    if exp_name == 'sv-bench':

        kwargs = {
            'data_path': data_path
        }

        for option in options:
            if option == 'mc':
                kwargs['train_mode'] = 'multi'
            elif option == 'b':
                kwargs['train_mode'] = 'binary'
            elif option == 'r':
                kwargs['train_mode'] = 'rank'
            elif option == 'cw':
                kwargs['class_weighted'] = True

        return SVExperiment(**kwargs)

    raise ValueError("Unkown experiment %s" % exp_type)
