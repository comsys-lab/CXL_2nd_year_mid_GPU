#!/bin/bash 

NET='llm_network'

EMBEDDING_IMAGE=embedding_image
EMBEDDING_CONTAINER=embedding_container_async

LOG_DIR=$(pwd)/$(date '+%y%m%d')_embedding_log
CPU_SERVER_IP='163.152.48.208' # change this to your CPU server IP

MODEL_DIR="/home/cloud_cxl/CXL_2nd_year_mid_GPU/Data/all-MiniLM-L6-v2" #<- specify your model directory here

# make network if not exits
docker network inspect $NET > /dev/null 2>&1 || docker network create $NET

echo "Loading Embedding Container..."

docker run -d \
    --name $EMBEDDING_CONTAINER \
    --network $NET \
    -p 5003:5003 \
    -v $MODEL_DIR:/app/model \
    -v $LOG_DIR:/app/log \
    --gpus '"device=1"' \
    $EMBEDDING_IMAGE python3 /app/embedding_server_async.py --cpu-server-ip=$CPU_SERVER_IP
