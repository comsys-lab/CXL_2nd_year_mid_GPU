import asyncio
import uuid
from aiohttp import web
from qdrant_client import QdrantClient, models  # qdrant
from sentence_transformers import SentenceTransformer
import time
import aiohttp
import os
import logging
import sys
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--cpu-server-ip", type=str, default="163.152.48.208", help="CPUserver IP address")
argparser.add_argument("--gpu-server-ip", type=str, default="163.152.48.205", help="GPU server IP address")
args = argparser.parse_args()

cpu_server_ip = args.cpu_server_ip
gpu_server_ip = args.gpu_server_ip

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1
request_queue = None


# Qdrant Initialization
logger.info("Initializing Qdrant...")  # qdrant
client = QdrantClient(host=cpu_server_ip, port=6333, timeout=30.0)
collection_name = "wiki_passages"  # qdrant

logger.info("Qdrant initialized.")  # qdrant

# SentenceTransformer 모델 로드
model = SentenceTransformer("/app/model", device="cuda")  # Load from mounted directory

request_timings = {}

def create_queue():
    global request_queue
    if request_queue is None:
        request_queue = asyncio.Queue()

def get_next_log_file(base_log_dir, base_name="log"):
    """
    주어진 폴더에 log1, log2, ... 형식으로 로그 파일 이름을 생성합니다.
    """
    index = 1
    while True:
        log_file_path = os.path.join(base_log_dir, f"{base_name}{index}.log")
        if not os.path.exists(log_file_path):
            return log_file_path
        index += 1

def print_timings():
    base_log_dir = "/app/log"  # 최상위 로그 디렉터리
    os.makedirs(base_log_dir, exist_ok=True)  # 최상위 디렉터리 생성 (없으면)

    # 새 로그 파일 생성
    log_file_path = get_next_log_file(base_log_dir)

    with open(log_file_path, "w") as log_file:  # 새 파일에 쓰기 모드로 열기
        log_file.write("\nRequest Timings Summary:\n")

        total_queuing_delay = 0.0
        total_embedding_time = 0.0
        total_vectordb_time = 0.0
        query_count = len(request_timings)

        for query_id, timings in request_timings.items():
            queuing_delay = timings.get('queuing_delay', 0.0)
            embedding_time = timings.get('embedding_time', 0.0)
            vectordb_time = timings.get('vectordb_time', 0.0)

            # 누적 시간 계산
            total_queuing_delay += queuing_delay
            total_embedding_time += embedding_time
            total_vectordb_time += vectordb_time

            log_file.write(f"Query ID {query_id}:\n")
            log_file.write(f"  Queuing delay: {queuing_delay:.4f} seconds\n")
            log_file.write(f"  Embedding Time: {embedding_time:.4f} seconds\n")
            log_file.write(f"  VectorDB Time: {vectordb_time:.4f} seconds\n")

        # 평균 계산
        if query_count > 0:
            avg_queuing_delay = total_queuing_delay / query_count
            avg_embedding_time = total_embedding_time / query_count
            avg_vectordb_time = total_vectordb_time / query_count

            log_file.write("\nAverage Times Across All Queries:\n")
            log_file.write(f"  Average Queuing delay: {avg_queuing_delay:.4f} seconds\n")
            log_file.write(f"  Average Embedding Time: {avg_embedding_time:.4f} seconds\n")
            log_file.write(f"  Average VectorDB Time: {avg_vectordb_time:.4f} seconds\n")

        log_file.write("\nEnd of Timings Summary.\n")
    
    logger.info(f"Log saved in: {log_file_path}")
    request_timings.clear()

async def process_batch(batch_data):
    task_id = uuid.uuid4()
    logger.info(f"[Task {task_id}] Processing batch start.")

    queries = [item['query'] for item in batch_data]
    query_ids = [item['query_id'] for item in batch_data]
    messages = [item['message'] for item in batch_data]
    start_times = [item['start_time'] for item in batch_data]

    try:
        embed_time = time.time()
        embedded_queries = model.encode(queries, batch_size=BATCH_SIZE).tolist()
        end_time = time.time()

        search_queries = [
            models.QueryRequest(
                query=embedded_queries[i],
                params={"hnsw_ef": 768},
                limit=5,
                with_payload=True
            ) for i in range(len(embedded_queries))
        ]

        results = client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries
        )
        vector_time = time.time()

        async with aiohttp.ClientSession() as session:
            for i, result in enumerate(results):
                queuing_delay = embed_time - start_times[i]
                embedding_time = end_time - embed_time
                vectordb_time = vector_time - end_time

                logger.info(f"[Task {task_id}] Queuing delay for Query ID {query_ids[i]}: {queuing_delay:.4f} seconds")
                logger.info(f"[Task {task_id}] Embedding time for Query ID {query_ids[i]}: {embedding_time:.4f} seconds")
                logger.info(f"[Task {task_id}] VectorDB time for Query ID {query_ids[i]}: {vectordb_time:.4f} seconds")

                if query_ids[i] not in request_timings:
                    request_timings[query_ids[i]] = {}
                request_timings[query_ids[i]]['queuing_delay'] = queuing_delay
                request_timings[query_ids[i]]['embedding_time'] = embedding_time
                request_timings[query_ids[i]]['vectordb_time'] = vectordb_time

                points = result.points
                context = " ".join([point.payload.get("document", "") for point in points])

                try:
                    async with session.post(
                        "http://" + gpu_server_ip + ":5000/generate",
                        json={
                            "message": messages[i],
                            "query_id": query_ids[i],
                            "question": queries[i],
                            "context": context
                        }
                    ) as response:
                        if response.status == 200:
                            logger.info(f"[Task {task_id}] Response success for Query ID {query_ids[i]}")
                        else:
                            logger.warning(f"[Task {task_id}] Response failed for Query ID {query_ids[i]} with status code {response.status}")
                except Exception as e:
                    logger.error(f"[Task {task_id}] Request failed for Query ID {query_ids[i]}: {e}")

    except Exception as e:
        logger.error(f"[Task {task_id}] Batch Query failed: {e}")

    logger.info(f"[Task {task_id}] Processing batch complete.")

async def worker():
    logger.info("[Worker] Launched")
    batch_data = []

    while True:
        try:
            logger.info("[Worker] Waiting for data in queue...")
            req_data = await request_queue.get()
            is_first_request = req_data.get('message') == 'first'

            if is_first_request:
                request_timings.clear()
            logger.info(f"[Worker] Retrieved request: {req_data}")
            batch_data.append(req_data)

            is_last_request = req_data.get('message') == 'end'

            logger.info(f"[Worker] Batch size: {len(batch_data)}, Is last request: {is_last_request}")

            if len(batch_data) >= BATCH_SIZE or is_last_request:
                await process_batch(batch_data)
                batch_data = []

                if is_last_request:
                    logger.info("[Worker] Last request received, exiting.")
                    print_timings()

        except Exception as e:
            logger.error(f"[Worker] Unexpected error: {e}")

async def handle_retrieve(request):
    try:
        data = await request.json()
        data['start_time'] = time.time()
        await request_queue.put(data)  # 큐에 추가
        logger.info(f"Data added to queue: {data}")
        return web.json_response({'status': 'Request received'})
    except Exception as e:
        logger.error(f"Error handling /retrieve: {e}")
        return web.json_response({'status': 'Error', 'message': str(e)}, status=500)

async def handle_status(request):
    logger.info(f"Status check from: {request}")
    return web.json_response({'status': 'OK'})

async def start_background_tasks(app):
    create_queue()  # 적절한 이벤트 루프에서 큐 생성
    app['worker'] = asyncio.create_task(worker())

async def cleanup_background_tasks(app):
    app['worker'].cancel()
    await app['worker']

app = web.Application()
app.router.add_post('/retrieve', handle_retrieve)
app.router.add_get('/status', handle_status)
app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # 이벤트 루프 정책 설정
    logger.info("Retrieval container is ready.")
    web.run_app(app, host="0.0.0.0", port=5003)
