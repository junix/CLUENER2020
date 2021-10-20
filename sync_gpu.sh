#!/usr/bin/env bash
tar cvf - --exclude=pytorch_version/outputs/  --exclude=pytorch_version/venv/ --exclude=pytorch_version/prev_trained_model   pytorch_version  | ssh root@gpu "cd /home/wanglijun/CLUENER2020 ; tar xvf -"
