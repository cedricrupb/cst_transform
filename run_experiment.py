import os
import logging
import argparse
import random
import numpy as np
import json

import torch


from torch_geometric.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm, trange
import inspect
import wandb
import shutil

from cst_transform.data import TreeDataLoader
from cst_transform.experiments import get_experiment, ConstantMetric
from cst_transform.model import create_model


logger = logging.getLogger(__name__)


class EarlyStopping():

    def __init__(self, args):
        self.stop = args.early_stopping
        self.best_eval = 0
        self.best_step = 0
        self.cooldown = self.stop

    def should_stop(self, global_step, val_loss):
        if self.stop < 0:
            self.best_step = global_step
            return False
        if self.is_improve(val_loss):
            self.cooldown = self.stop
            self.best_step = global_step
            self.best_eval = val_loss
            return False
        else:
            if self.best_eval == val_loss:
                self.cooldown = self.stop
            self.cooldown = self.cooldown - 1
        return self.cooldown <= 0

    def is_improve(self, val_loss):
        if self.stop < 0:
            return False
        return val_loss > self.best_eval


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def reduce_by_signature(clazz, mapping):

    sig = inspect.signature(clazz)
    res = {}

    for k, V in mapping.items():
        if k in sig.parameters:
            res[k] = V
    return res


def generate_wandb_desc(args):

    desc = {}

    desc['Name'] = args.name
    desc['Device'] = args.device
    desc['#GPU'] = args.n_gpu
    desc['FP16'] = args.fp16
    desc['#Epochs'] = args.num_epochs
    desc['Batch Size'] = args.batch_size
    desc['LR'] = args.learning_rate
    desc['Warmup'] = args.warmup_steps


    return '\n'.join(["%s: %s" % (k, str(v)) for k, v in desc.items()])


def save_checkpoint(args, model, global_step):

    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_%s" % global_step)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    out = {k: str(v) for k, v in vars(args).items()}

    with open(os.path.join(checkpoint_dir, "args.json"), "w") as o:
        json.dump(out, o, indent=4)

def load_checkpoint(model, global_step):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_%s" % global_step)

    if not os.path.exists(checkpoint_dir):
        return

    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


def forward_batch(model, batch, exclude=None):

    keys = set(batch.keys)

    if exclude is not None:
        keys = set.difference(keys, exclude)

    inputs = {k: batch[k] for k in keys}

    inputs = reduce_by_signature(model.forward, inputs)
    return model(**inputs)


def train(args, model, experiment, cb=None):

    stopping = EarlyStopping(args)

    train_dataset = experiment.train_dataset
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = TreeDataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8
    )

    if args.num_steps > 0:
        t_total = args.num_steps
        args.num_epochs = t_total // len(train_dataloader) + 1
    else:
        t_total = (len(train_dataset) // args.batch_size + 1) * args.num_epochs

    if args.save_epoch > 0:
        args.save_step = (len(train_dataset) // args.batch_size + 1) * args.save_epoch

    if args.logging_epoch > 0:
        args.logging_step = (len(train_dataset) // args.batch_size + 1) * args.logging_epoch

    if args.warmup_epochs > 0:
        args.warmup_steps = (len(train_dataset) // args.batch_size + 1) * args.warmup_epochs

    no_decay = ["bias", "BatchNorm1d.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_epochs)
    logger.info("  Total train batch size = %d",
                args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    epoch = 0
    stop = False
    train_iterator = trange(int(args.num_epochs), desc="Epoch")
    set_seed(args)

    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = batch.to(args.device)
            optimizer.zero_grad()
            y = batch.y

            output = forward_batch(
                model, batch, exclude=set(['y'])
            )
            loss = experiment.loss(output, y)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            improve = False

            if args.logging_step > 0 and global_step % args.logging_step == 0:
                inter_loss = (tr_loss - logging_loss) / args.logging_step
                logging_loss = tr_loss
                logging.info("Epoch %d Step %d Train Loss %f" % (epoch, step, inter_loss))
                val_eval = evaluate(args, model, experiment, mode="val")
                val_eval['train_loss'] = inter_loss
                if args.wandb:
                    wandb.log(val_eval)
                last_best = stopping.best_step
                improve = stopping.is_improve(val_eval['val_accuracy'])
                stop = stopping.should_stop(global_step, val_eval['val_accuracy'])
                if cb:
                    stop = cb(val_eval)
                if stop:
                    logging.info("Early stopping after %d nonimprovements" % stopping.stop)

            if improve or (args.save_step > 0 and global_step % args.save_step == 0):
                save_checkpoint(args, model, global_step)
                if args.save_step <= 0 and last_best > 0:
                    shutil.rmtree(os.path.join(args.output_dir, "checkpoint_%s" % last_best))

            if global_step > t_total or stop:
                epoch_iterator.close()
                break
        if global_step > t_total or stop:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, stopping.best_step

class ConfusionMatrix:

    def __init__(self, tools):
        self.tools = tools
        self.m = [0]*(len(self.tools)**2)

    def __call__(self, prob, labels):
        _, index = prob.topk(1)
        index = index.squeeze()

        pp = self.m
        for i in range(index.size(0)):
            c = 1 if labels[i].max().item() >= 0.5 else 0
            p = labels[i].argmax().item()
            r = 1 if labels[i, index[i]].item() >= 0.5 else 0
            if c == 1 and r == 0:
                pp[p*len(self.tools)+index[i]] += 1


    def print(self):
        pp = self.m
        print("Confusion")
        for i in range(len(pp)):
            if pp[i] == 0:
                continue
            b = i // len(self.tools)
            k = i % len(self.tools)
            print("\t%s : %d (%s)" % (self.tools[k], pp[i], self.tools[b]))


def evaluate(args, model, experiment, mode="test", prefix=""):

    eval_dataset = experiment.test_dataset if mode == 'test' else experiment.val_dataset
    dataloader = TreeDataLoader(
        eval_dataset, batch_size=args.eval_batch_size, num_workers=8
    )

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    metrics = experiment.eval_metrics

    take_loss = 'loss' in metrics
    cm = ConfusionMatrix(experiment.gold_labels['classes'])

    for m in metrics.values():
        m.reset()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(args.device)

            y = batch.y
            gold = batch.gold

            outputs = forward_batch(
                model, batch, exclude=set(['y', 'gold'])
            )

            if take_loss:
                loss = experiment.loss(outputs, y)
                eval_loss += loss.item()
                nb_eval_steps += 1

            logits = outputs.detach().cpu()
            x = torch.nn.Softmax(dim=-1)(logits)
            cm(x, gold)

            for ev in metrics.values():
                if hasattr(ev, 'require_gold') and ev.require_gold:
                    ev(logits, gold)
                else:
                    ev(logits, y)

    if take_loss:
        metrics['loss'] = ConstantMetric(eval_loss / nb_eval_steps)
    out = {}
    for m, V in list(metrics.items()):
        out['_'.join([mode, m])] = V.eval()
    metrics = out

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key, str(metrics[key]))

    cm.print()

    return metrics


def prepare_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, required=True,
                        help="name of the model")
    parser.add_argument("--exp_type", type=str, required=True,
                        help="name of the experiment to perform")
    parser.add_argument("--model_cfg", type=str,
                        help="Model configuration")
    parser.add_argument("--lmdb", type=str,
                        help="Path to train database")

    parser.add_argument(
        "--do_train", action="store_true",
        help="whether to train"
    )
    parser.add_argument(
        "--do_eval", action="store_true",
        help="whether to evaluate"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="batch size to train with"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32,
        help="batch size to evaluate with"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100,
        help="number of epochs"
    )
    parser.add_argument(
        "--num_steps", type=int, default=0,
        help="number of train steps (Overrides number of epochs)."
    )

    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_epochs", default=0, type=int,
                        help="Linear warmup over warmup_epochs.")

    parser.add_argument(
        "--early_stopping", default=-1, type=int,
        help="Number of epochs to wait for improvement. -1 deactivate"
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory to write to"
    )

    parser.add_argument(
        "--temp_dir", type=str,
        help="Tempory directory. If not specified the same as output_dir"
    )

    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for logging experiment stats.")
    parser.add_argument("--logging_step", type=int, default=0,
                        help="Logging at step X.")
    parser.add_argument("--logging_epoch", type=int, default=0,
                        help="Logging at epoch X. Will overwrite logging step.")
    parser.add_argument("--save_step", type=int, default=0,
                        help="Save progress at step X.")
    parser.add_argument("--save_epoch", type=int, default=0,
                        help="Save progress at epoch X. Overrides save_step.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="avoid using an gpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Log level")
    parser.add_argument("--checkpoint", type=int,
                        help="Checkpoint-Id to warmstart model weights")
    return parser


def run_experiment(args):
    if args.no_cuda:
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()

    args.device = device

    set_seed(args)

    model = args.model_type

    exp_type = args.exp_type

    name = "%s - %s" % (model, exp_type)
    args.name = name

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=getattr(logging, args.log.upper()))
    logger.warning("Name: %s, Device: %s, n_gpu: %s, 16-bits training: %s",
                    name, device, args.n_gpu, args.fp16)

    args.output_dir = os.path.join(
        args.output_dir, args.exp_type, model
    )
    output_dir = args.output_dir

    if not args.temp_dir:
        args.temp_dir = output_dir

    temp_dir = args.temp_dir

    if not args.lmdb:
        args.lmdb = temp_dir

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_cfg = None

    if args.model_cfg:
        with open(args.model_cfg, "r") as i:
            model_cfg = json.load(i)


    experiment = get_experiment(args.exp_type, args.lmdb)
    model = create_model(args.model_type, experiment, model_cfg)

    print(model)

    if args.wandb:
        wandb.init(project="attend_code", name=name,
                    notes=generate_wandb_desc(args),
                    config=model_cfg)
        wandb.watch(model)

    model.to(args.device)
    experiment.loss.to(args.device)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    if args.do_train:
        global_step, tr_loss, best_step = train(
            args, model, experiment
        )
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_step)
        save_checkpoint(args, model, global_step)

        if best_step != global_step:
            load_checkpoint(model, best_step)


    if args.do_eval:
        val_eval = evaluate(args, model, experiment, mode="val")
        test_eval = evaluate(args, model, experiment)
        test_eval.update(val_eval)
        test_output = os.path.join(output_dir, "test.json")
        with open(test_output, "w") as o:
            json.dump(test_eval, o, indent=4)

        if args.wandb:
            for k, v in test_eval.items():
                wandb.run.summary[k] = v


if __name__ == '__main__':
    parser = prepare_argparse()

    args = parser.parse_args()

    run_experiment(args)
