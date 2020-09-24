import os
import argparse
import json
import optuna


def convert_reduced(config):

    out = {
        'exp_type': config['exp_type'],
        'model_type': config['model_type'],
        'output_dir': config['output_dir'],
        'temp_dir': config['temp_dir'],
        'lmdb': config['lmdb'],
        'batch_size': config['batch_size'],
        'num_epochs': config['epochs'],
        'weight_decay': config['weight_decay'],
        'learning_rate': config['lr'],
        'warmup_epochs': int(config['warmup_factor'] * config['epochs'])
    }

    if config['hidden'] > 32:
        config['intermediate_ratio'] = 1

    model_config = {
        'dropout': 0.1,
        'config':{
            'hidden_size': config['hidden'],
            'max_depth': config['max_depth'],
            'num_attention_heads': config['num_heads'],
            'intermediate_size': config['intermediate_size'],
            'attention_dropout_ratio': config['dropout'],
            'residual_dropout_ratio': config['dropout']
        }
    }

    return out, model_config


def run_experiment(config):

    config, model_config = convert_reduced(config)

    defaults = {
        'log': 'WARN',
        'logging_epoch': 1,
        'wandb': True,
        'do_train': True,
        'do_eval': True,
        'model_type': 'hierarchical_encoder',
        'warmup_epochs': 10,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'early_stopping': 20,
        'num_epochs': 1000,
        'save_epoch': 1,
        'eval_batch_size': 16
    }

    defaults.update(config)

    defaults = {k: v for k, v in defaults.items() if v is not False}

    if not os.path.exists(defaults['output_dir']):
        os.makedirs(defaults['output_dir'])

    cfg_path = os.path.join(defaults['output_dir'], 'config.json')

    with open(cfg_path, "w") as o:
        json.dump(model_config, o)

    defaults['model_cfg'] = cfg_path

    postfix = ' '.join(
        [
            ("--%s" % k) if v is True else "--%s %s" % (k, str(v))
            for k, v in defaults.items()
        ]
    )

    return os.system(
        "python run_experiment.py %s" % postfix
    )

def make_run(args):
    def run_trial(trial):

        sample = {
            'batch_size': 16,
            'lr': trial.suggest_loguniform("lr", 1e-5, 1e-2),
            'epochs': 30,
            'hidden': 64,
            'max_depth': 3,
            'num_heads': 1,
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
            'warmup_factor': 0.1,
            'transform_values': 1,
            'intermediate_size': 64,
            'dropout': 0.1,
        }

        for attr in ['exp_type', 'model_type', 'output_dir', 'lmdb', 'temp_dir', 'seed']:
            if attr not in sample:
                sample[attr] = getattr(args, attr)

        sample['output_dir'] = os.path.join(sample['output_dir'], "BayesSample%d" % trial.number)

        test_path = os.path.join(
            sample['output_dir'], sample['exp_type'], 'transformer_tree', 'test.json'
        )

        print(sample)
        run_experiment(sample)

        with open(test_path, "r") as i:
            test = json.load(i)

        return 1 - test['val_accuracy']


    return run_trial


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", required=True)
    parser.add_argument("--model_type", default="hierarchical_encoder")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--temp_dir")
    parser.add_argument("--num_samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if not args.temp_dir:
        args.temp_dir = args.output_dir

    opt = make_run(args)
    study = optuna.create_study()

    study.optimize(opt, n_trials=args.num_samples, catch=(Exception,))
