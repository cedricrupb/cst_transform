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

## Training a custom selector with scikit-learn and SVComp'21
As shown before, the CST Transformer learns an expressive feature representation of source code, usable for algorithm selection of verification algorithm. In fact, we have shown that the extracted features are competitive with state-of-the-art methods, while maintaining a smaller computational footprint.

Now, we show how you can train a custom algorithm selector using the CST Transformer as a feature extractor and scikit-learn model as the selection model.

### Extracting training data from SVComp'21
Training data for an algorithm selector can be obtained from the annual software verification competition. Here, we choose the latest competition from 2021. Verification tasks and benchmarks of verifiers can be obtained from Zenodo [[results](https://zenodo.org/record/4458215)][[benchmark](https://zenodo.org/record/4459126)]:
```
wget https://zenodo.org/record/4458215/files/svcomp21-results.zip
wget https://zenodo.org/record/4459126/files/sv-benchmarks-svcomp21.zip
```
After extracting all files, we are mostly interested in the following two folders:
- `svcomp21-results/results-verified` (Folder containing benchmarks for verifier run at SVCOMP'21)
- `sv-benchmarks` (The set of verification tasks)

### Embedding SV-Comp tasks as feature vectors
As the first step, we apply a pretrained CST Transformer to map verification tasks to feature vectors. In the following, we employ the `run_batch_embed.py` script, which is more confient to embed a large number of programs than aforementioned methods.
To embed all SV-Comp tasks of the reachability categore, run the following command:
```bash
python run_batch_embed.py sv-benchmarks/c/ sv-comp-embed.jsonl
 --file_index sv-benchmarks/c/ReachSafety-* sv-benchmarks/c/SoftwareSystems-*
```
The script will produce a jsonl file with all embeddings (which takes around 3 - 5 hours depending on the CPU.)

Note: The CST Transformer is only pretrained for reachability tasks. While it is possible to embed other verification task types, the quality of representation is untested. 
If you wish to select another checkpoint for embedding, please refer to the Embedding section.

### Extracting labels from SVComp results
The necessary labels for training and testing models can be extracted with the provided script `scripts/parse_svcomp_results.py`. Therefore, we can parse the labels by executing:
```bash
python scripts/parse_svcomp_results.py svcomp21-results/results-verified labels.json --svbench_path sv-benchmarks/ 
```
Since labels are generated by comparing with the tasks verdict, it is important to provide the path to `sv-benchmark`. The script will then load the task file from the benchmark and extract both verdict and the respective C file.

## Training a logistic hardness model for CPAchecker
Training a single hardness model can be easily done with the provided utilities. In the following, we will import both embeddings and labels to numpy for training a logistic regression classifier.
The goal of the classifier is to provide a score for how likely CPAchecker will solve a specific given task.
```python
from cst_transform.utils import EmbeddingLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data_loader = EmbeddingLoader("sv-comp-embed.jsonl", "labels.jsonl")

X, y = data_loader("CPAchecker")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Train logistic regression classifier
model = LogisticRegression(C = C, max_iter = 10_000)
model.fit(X_train, y_train)

print("The hardness model achieved an accuracy of %f%%" % (100 * model.score(X_test, y_test)))
```
For training a complete selector usable with `run_selector.py` please refer to the Jupyter notebook `train_selector.ipynb`. There we provide more examples to train hardness models and demonstrate further utilities provided by this library. Most importantly, the notebook can be used to train a complete selector from scratch.