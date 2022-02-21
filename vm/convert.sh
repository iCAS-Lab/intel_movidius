#!/bin/sh
PATH_TO_OPENVINO=/home/brendan/intel/openvino_2021
CURRENT_DIR=$(pwd)
mkdir openvino_$1
cd openvino_$1
echo $CURRENT_DIR/$1
python3 $PATH_TO_OPENVINO/deployment_tools/model_optimizer/mo.py --input_model $CURRENT_DIR/$1 --input_shape=$2
cd ..