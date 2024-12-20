# SpecRnet application to ASVspoof2019
Code for the Audio Processing & Indexing course project. Applying SpecRnet to ASVspoof2019 LA dataset and analyzing results.

Inspired by the paper by Kawa et al. : https://arxiv.org/abs/2210.06105

Data can be downloaded from asvspoof.org : https://datashare.ed.ac.uk/handle/10283/3336

train_asvspoof.py trains the model once

run_multiple.py runs specrnet 5 times and evaluates 5 times on different random seeds.

analysis.py uses the best model from run_multiple.py to do more in-depth analysis of the results.