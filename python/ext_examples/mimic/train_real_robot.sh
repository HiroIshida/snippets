#!/bin/bash
python3 -m mimic.scripts.train_auto_encoder -pn real_robot -n 5000 -bottleneck 60
#python3 -m mimic.scripts.train_propagator -pn real_robot -n 4000 -model LSTM
#python3 -m mimic.scripts.train_propagator -pn real_robot -n 4000 -model LSTM
#python3 -m mimic.scripts.train_propagator -pn real_robot -n 4000 -model LSTM
#python3 -m mimic.scripts.train_propagator -pn real_robot -n 4000 -model LSTM
