## Experiment Setup for GPU Server

### HW configuration

| **CPU** | 2× Intel Xeon 4410Y (Sapphire Rapids) @2.5 GHz, 12 cores |
| --- | --- |
|  | 30 MiB LLC per CPU, Hyper-Threading Enabled |
| **Memory** | Socket 0: 2× 64 GB DDR5-4800 MT/s, Total 128 GB |
|  | Socket 1: 2× 64 GB DDR5-4800 MT/s, Total 128 GB |
| **GPU** | 2x NVIDIA GeForce RTX 4090 |

### SW configuration

- OS: Ubuntu 22.04.4 LTS (Jammy Jellyfish)
- Kernel: Linux 6.12.0
- Python: 3.11.7
- CUDA: 12.2
- Docker: 27.5.1

### Build Docker Images

```bash
docker build -f Dockerfile.embedding -t embedding_image .
docker build -f Dockerfile.llm -t llm_inference_image .
```

### Run LLM and Embedding model server

- Run embedding and LLM in GPU server
    
    ```bash
    bash run_embedding.sh
    bash run_llm.sh
    ```