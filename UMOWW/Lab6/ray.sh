#!/bin/bash

MODEL="Qwen/Qwen3-4B-Thinking-2507"
HEAD_NODE_IP="172.20.83.210"
WORKER_IPS=("172.20.83.211" "172.20.83.212" "172.20.83.213")
RAY_PORT="6379"
MAX_MODEL_LEN="8192"
GPU_MEM_UTIL="0.9"
TP_SIZE="4"

MODE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --single) MODE="single" ;;
        --cluster) MODE="cluster" ;;
        --init) MODE="init" ;;
        --clean) MODE="clean" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODE" ]; then
    echo "Error: You must specify a mode. Use --single, --cluster, --init, or --clean"
    exit 1
fi

if [ "$MODE" == "single" ]; then
    vllm serve $MODEL --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL
elif [ "$MODE" == "cluster" ]; then
    vllm serve $MODEL --tensor-parallel-size $TP_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --distributed-executor-backend ray
elif [ "$MODE" == "init" ]; then
    SETUP_ENV="conda info --envs | grep -qw ray || (conda create -n ray python=3.12 -y && conda activate ray && pip install ray vllm transformers); conda activate ray"
    
    echo "Starting Ray head node..."
    ray start --head --port=$RAY_PORT || { echo "Error: Failed to start Ray head node. Aborting."; exit 1; }
    
    echo "Starting Ray worker nodes..."
    for WORKER_IP in "${WORKER_IPS[@]}"; do
        echo "Connecting to $WORKER_IP..."
        ssh -t "$WORKER_IP" "bash -ic '$SETUP_ENV && ray start --address=$HEAD_NODE_IP:$RAY_PORT'"
    done
elif [ "$MODE" == "clean" ]; then
    echo "Stopping Ray and cleaning conda environments on worker nodes..."
    for WORKER_IP in "${WORKER_IPS[@]}"; do
        echo "Connecting to $WORKER_IP..."
        ssh -t "$WORKER_IP" "bash -ic 'conda activate ray && ray stop && conda deactivate && conda env remove -n ray -y'"
    done
    
    echo "Stopping Ray on head node..."
    ray stop
fi