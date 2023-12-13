# CS152-SSL

Download data from the CS152 server: /cs/cs152/shared/SSL/CS152-SSL/LabeledData and /cs/cs152/shared/SSL/CS152-SSL/UnlabeledData 

Run ``scripts.sh`` to re-run all of our experiments

Run ``analysis.py`` to obtain our plots

Important files:
* ``mixmatch_training.py``: this file is responsible for training a ResNet18 model using the MixMatch algorithm
* ``mixmatch.py``: implement core of the MixMatch algorithm
* ``model.py``: implements and creates ResNet18 with extended layers
* ``custom_utils.py``: climbing hold extracting to create labeled and unlabeld datasets, implement custom dataset classes
* ``analysis.py``: implements plotting of results and confusion matrix




