Attend and Represent: A Novel View on Algorithm Selection for Software Verification
##############
Supplementary Material
Authors: **Blind**

This package contains the supplementary material for the paper including WebUI, splits and labelling.
Because of double-blind submission, we are not publishing the full code source.
A replication package will be published after the double-blind phase.
Disclaimer: The WebUI is only prototypical. Most functions are available. However, bugs can occur.

Content
###############

checkpoints/ - The four checkpoints obtained by our experiments in RQ1 and RQ2
labels/ - The employed labels for every task. Files ending with "non" refer to the deduplicated dataset.
resources/ - Includes splits, label index and Proto2 definitions
webui/ - Root of our webui
webui/model/ - Includes code used in our training.


WebUI - Installation
###########

0. Virtual Environment
We propose to install all dependencies of this project in an empty
virtual environment.
The implementation was tested on Python 3.6.1 on WSL.

1. Installation
The framework Torch 1.2.0 has to be installed at first:
``pip install torch==1.2.0``

Afterwards the remaining dependencies can be installed by:
``pip install -r requirements.txt``

WebUI - Usage
####################

2. Change the folder to webui/

3. Run the following command:
``python serving.py [bmc-ki | algorithms | sc | tools]``
The script will load one of the mentioned checkpoints.
Then, it locally starts a web server running on "localhost:5000".

4. Start your browser (Tested on Chromer, but other browser should work, too)
  and visit the URL: "localhost:5000".

INFO: To use the WebUI at its fullest, an active internet connection is required.
The webpage will download additional CSS and JavaScript code from CDNs.


Standalone Algorithm Selector
################

The Algorithm Selector can be used without the WebUI.
Change the directory to webui/model/.
Afterwards, run the following command:
``python run_predict.py [bmc-ki | algorithms | sc | tools] [file.c]``
The argument ``[file.c]`` can be replaced by an arbitrary C file.


Feature Vector for a program
#######################

Additionally, the standalone selector can be used to generate a featur representation
for a given C program.
In the directory webui/model/, run the following command:
``python run_predict.py [checkpoint] [file.c] --embed_file [out.json]``

The script will map the given C program to a feature vector saved in the Json file ``[out.json]``


WebUI - Known bugs
#####################
Some known bugs for the Web UI. They will be fixed at release.
Bugs only relate to the web interface, but not to the backend.
The backend can be run idependently as described above.

- If the C program contains a label, the textual highlights will be moved upwards.
- Cancelling the prediction via UI does not cancel the prediction.
  To cancel the prediction, refresh the web page.
- If the error message occur "This seems to be an old request."
  Refresh the web page.
- Sometimes for an unknown reason, the UI fails to sync with the backend.
  If there does not occur a result after 4min, refresh the web page and try again.
  Alternatively, the browser console can be consolidated.
- The mapping of attention to code line is sometimes buggy.
  Therefore, the attention score is sometimes shown for a line above.
  This will be fixed in the refactored code base.
