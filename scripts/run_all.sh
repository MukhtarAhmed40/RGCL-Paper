#!/bin/bash
python preprocess.py
python train.py
python evaluate.py
python attack_eval.py
python reproduce_results.py
