# Attend and Represent: A Novel View on Algorithm Selection for Software Verification
By Cedric Richter and Heike Wehrheim [[paper](https://ieeexplore.ieee.org/document/9286080)]

### **Attn + Represent: Architecture**
![architecture]

Today, a plethora of different software verification tools exist. When
having a concrete verification task at hand, software developers
thus face the problem of algorithm selection. Existing algorithm
selectors for software verification typically use handpicked program
features together with (1) either manually designed selection
heuristics or (2) machine learned strategies. While the first approach
suffers from not being transferable to other selection problems, the
second approach lacks interpretability, i.e., insights into reasons for
choosing particular tools.
In this paper, we propose a novel approach to algorithm selection
for software verification. Our approach employs representation
learning together with an attention mechanism. Representation
learning circumvents feature engineering, i.e., avoids the handpicking
of program features. Attention permits a form of interpretability
of the learned selectors. We have implemented our approach and
have experimentally evaluated and compared it with existing approaches.
The evaluation shows that representation learning does
not only outperform manual feature engineering, but also enables
transferability of the learning model to other selection tasks.

The repository contains our implementation and supplementary material.
The supplementary material contains pre-trained models based on four selection tasks
and web interface which can be used to test these models.


[architecture]: https://github.com/cedricrupb/cst_transform/blob/master/architecture.PNG

## Installation
Dependencies: Python 3.8.0, PyTorch 1.8.0, PyTorch Geometric 1.7.0, Clang 12.0
```bash
$ pip install torch==1.8.0
$ pip install -r requirements.txt
$ pip install -e "."
```

## Dataset generation
Clone the official [sv-benchmark repository](https://github.com/sosy-lab/sv-benchmarks):
```bash
$ git clone https://github.com/sosy-lab/sv-benchmarks/tree/svcomp18
```
To generate the dataset from source code:
```bash
$ python run_dataset_generation.py --benchmark_url [benchmark] --benchmark_code_dir [sv-bench path] --output_dir [path to dataset lmdb]
```

## Inference
The CST Transformer can be used for algorithm selection by running:
```bash
$ python run_predict.py --checkpoint [checkpoint] [file.c]
```
The command line preprocesses the given C file and then performs
a algorithm selection based on the given checkpoint. 
The checkpoint argument is optional and we load the "Tools" checkpoint on 
default. Possible choices are [bmc-ki, sc, algorithms, tools] or a path to an
existing checkpoint. 

## Embedding
Besides algorithm selection, a pretrained CST Transformer can represent
a C program in form of a vector representation. Embedding programs is currently
handled by `run_predict.py`, which allows us to not only embed programs but also
run inference with a precomputed embedding:
```bash
$ python run_predict.py --checkpoint [checkpoint] --embed --embed_file [file.json] [file.c]
```
This command prints out an program embedding instead of performing algorithm selection.
If the option `--embed_file` is specified the embedding is additionally saved to
a JSON file.


Inference on an existing embedding file `file.json` can be run by leaving out the `--embed` option. In this case, `file.c` will be ignored.

Note however that we expect that the precomputed embedding is produced by the same checkpoint which should be used for algorithm. The behavior is undefined in all other cases.