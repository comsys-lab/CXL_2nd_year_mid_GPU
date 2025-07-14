#!/bin/bash 

NET='llm_network'

LLM_IMAGE=llm_inference_image
LLM_CONTAINER=llm_inference_container

LOG_DIR=$(pwd)/$(date '+%y%m%d')_llm_log
CPU_SERVER_IP='163.152.48.208' # change this to your CPU server IP

# make network if not exits
docker network inspect $NET > /dev/null 2>&1 || docker network create $NET

echo "Loading LLM Container..."

docker run -d \
  --name $LLM_CONTAINER \
  --network $NET \
  -p 5000:5000 \
  -v $(pwd)/huggingface_cache:/app/huggingface_cache \
  -v $LOG_DIR:/app/log \
  --gpus '"device=0"' \
  $LLM_IMAGE python3 /app/llm_inference_server.py --cpu-server-ip=$CPU_SERVER_IP
