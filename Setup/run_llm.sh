#!/bin/bash 

NET='llm_network'

LLM_IMAGE=llm_inference_image
LLM_CONTAINER=llm_inference_container

LOG_DIR=$(pwd)/$(date '+%y%m%d')_llm_log
CPU_SERVER_IP='163.152.48.208' # change this to your CPU server IP

MODEL_DIR="/home/cloud_cxl/CXL_2nd_year_mid_GPU/Data/Llama3-ChatQA-1.5-8B" #<- specify your model directory here

# make network if not exits
docker network inspect $NET > /dev/null 2>&1 || docker network create $NET

echo "Loading LLM Container..."

docker run -d \
  --name $LLM_CONTAINER \
  --network $NET \
  -p 5000:5000 \
  -v $MODEL_DIR:/app/model \
  -v $LOG_DIR:/app/log \
  --gpus '"device=0"' \
  $LLM_IMAGE python3 /app/llm_inference_server.py --cpu-server-ip=$CPU_SERVER_IP
