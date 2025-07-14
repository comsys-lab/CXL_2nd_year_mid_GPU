from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import time
import logging
from queue import Queue
import requests
from threading import Thread
import os
import argparse

# Flask 앱 생성
app = Flask(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--cpu-server-ip", type=str, default="163.152.48.208", help="CPUserver IP address")
args = argparser.parse_args()

cpu_server_ip = args.cpu_server_ip

# NVIDIA ChatQA 모델 및 토크나이저 로드
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
# REQ_GEN_URL = "http://163.152.48.208:6000"
REQ_GEN_URL = "http://" + cpu_server_ip + ":6000"

# 전역 변수
request_queue = Queue()
response_store = {}
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128

request_timings = {}
total_latencies = []
ttft_latencies = []
rps_start_time = None
rps_end_time = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # PAD 토큰 및 ID 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<pad>")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
except Exception as e:
    raise RuntimeError(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")

# 포맷팅 함수 정의
def get_formatted_input(text, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in text:
        if item['role'] == "user":
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in text]) + "\n\nAssistant:"
    return system + "\n\n" + context + "\n\n" + conversation

# Custom StoppingCriteria 클래스 정의
class TimingStoppingCriteria(StoppingCriteria):
    def __init__(self, max_new_tokens, batch_ids, callback_url=None):
        self.max_new_tokens = max_new_tokens
        self.batch_ids = batch_ids
        self.token_count = 0
        self.token_times = []
        self.first_token_time = None
        self.callback_url = callback_url

    def send_callback(self, data, callback_type):
        if self.callback_url:
            try:
                response = requests.post(f"{self.callback_url}/{callback_type}", json=data)
                logging.info(f"Callback response status: {response.status_code}")
            except Exception as e:
                logging.error(f"Error sending callback request: {e}")

    def __call__(self, input_ids, scores, **kwargs):
        current_time = time.time()

        # 첫 번째 토큰 생성 시간 기록 및 콜백 전송
        if self.token_count == 1 and self.first_token_time is None:
            self.first_token_time = current_time
            if self.callback_url:
                for query_id in self.batch_ids:
                    data = {
                        "query_id": query_id,
                        "event": "first_token_generated",
                        "time": self.first_token_time
                    }
                    self.send_callback(data, "TTFT")

        # 토큰 간 시간 기록
        if self.token_count > 1:
            self.token_times.append(current_time - self.last_token_time)

        self.last_token_time = current_time
        self.token_count += 1

        # return self.token_count >= self.max_new_tokens
        return self.token_count >= 3

# Logger 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_statistics():
    base_log_dir = "/app/log"  # 최상위 로그 디렉터리
    os.makedirs(base_log_dir, exist_ok=True)  # 최상위 디렉터리 생성 (없으면)

    # 새 로그 파일 생성
    log_file_path = f"{base_log_dir}/LLM_summary_B{BATCH_SIZE}_{time.strftime('%Y%m%d_%H_%M')}"


    with open(log_file_path, "w") as log_file:  # 새 파일에 쓰기 모드로 열기
        log_file.write("\nLLM Summary:\n")

        total_queuing_delay = 0.0
        total_ttft = 0.0
        total_latency = 0.0
        query_count = 0

        for query_id, timings in request_timings.items():
            queueing_delay = timings.get('queueing_delay')
            ttft = timings.get('ttft')
            latency = timings.get('latency')

            # 누적 시간 계산
            total_queuing_delay += queueing_delay
            total_ttft += ttft
            total_latency += latency
            query_count += 1

            # request_timings[query_id]['queueing_delay'] = queueing_delay
            # request_timings[query_id]['ttft'] = ttft
            # request_timings[query_id]['total_latency'] = total_latency

            log_file.write(f"Query ID {query_id}:\n")
            log_file.write(f"  Queueing delay of LLM: {queueing_delay:.4f} seconds\n")
            log_file.write(f"  TTFT of LLM: {ttft:.4f} seconds\n")
            log_file.write(f"  Latency of LLM: {latency:.4f} seconds\n")

        # 평균 계산
        if query_count > 0:
            avg_queuing_delay = total_queuing_delay / query_count
            avg_ttft = total_ttft / query_count
            avg_latency = total_latency / query_count

            log_file.write("\nAverage Times Across All Queries:\n")
            log_file.write(f"  Average Queueing delay of LLM: {avg_queuing_delay:.4f} seconds\n")
            log_file.write(f"  Average TTFT of LLM: {avg_ttft:.4f} seconds\n")
            log_file.write(f"  Average Latency of LLM: {avg_latency:.4f} seconds\n")
        
            throughput = query_count/(rps_end_time - rps_start_time)
            log_file.write("\nThroughput of LLM\n")
            log_file.write(f"  Throughput of LLM: {throughput:.4f} requests/sec\n")

        log_file.write("\nEnd of Timings Summary.\n")
        
def batch_inference_worker():
    global total_latencies, ttft_latencies, rps_start_time, rps_end_time, request_timings

    logger.info("Batch inference worker started.")
    while True:
        batch = []
        batch_ids = []
        batch_start_times = []
        messages = []

        while len(batch) < BATCH_SIZE:
            message, query_id, formatted_input, start_time = request_queue.get()
            if message == "start":
                request_timings = {}
                rps_start_time = time.time()

            batch.append(formatted_input)
            batch_ids.append(query_id)
            batch_start_times.append(start_time)
            messages.append(message)
            if message == "end":
                break

        # if not batch:
        #     time.sleep(0.1)  # 요청이 없을 때 대기
        #     continue

        logger.info(f"Processing batch of size {len(batch)}.")

        try:
            queueing_delay = time.time() - start_time
            tokenized_prompts = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # Stopping Criteria with Timing
            stopping_criteria = TimingStoppingCriteria(max_new_tokens=MAX_NEW_TOKENS, batch_ids=batch_ids, callback_url=REQ_GEN_URL)

            outputs = model.generate(
                input_ids=tokenized_prompts["input_ids"],
                attention_mask=tokenized_prompts["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stopping_criteria])
            )

            responses = [
                tokenizer.decode(output[tokenized_prompts["input_ids"].shape[-1]:], skip_special_tokens=True)
                for output in outputs
            ]
            end_time = time.time()


            for input_id, query_id, response, start_time, message in zip(batch, batch_ids, responses, batch_start_times, messages):
                ttft = stopping_criteria.first_token_time - start_time if stopping_criteria.first_token_time else None
                avg_tpot = (sum(stopping_criteria.token_times) / len(stopping_criteria.token_times)) if stopping_criteria.token_times else None
                latency = end_time - start_time

                logger.info(f"Message: {message}, Query ID: {query_id}, Latency: {latency:.4f}, Queueing delay: {queueing_delay:.4f}, Prefill: {(ttft-queueing_delay):.4f} seconds, Decode: {(latency-ttft):.4f}, Avg TPOT: {avg_tpot:.4f} seconds")
                # logger.info(f"Query ID: {query_id}\n Input: {input_id}\n Response: {response}")
                logger.info(f"Query ID: {query_id}\n Response: {response}")

                if query_id not in request_timings:
                    request_timings[query_id] = {}
                request_timings[query_id]['queueing_delay'] = queueing_delay
                request_timings[query_id]['ttft'] = ttft
                request_timings[query_id]['latency'] = latency

                data = {
                    "query_id": query_id,
                    "response": response
                }
                stopping_criteria.send_callback(data, "complete")
                if message == "end":
                    stopping_criteria.send_callback(data, "notify_completion")
                    rps_end_time = time.time()
                    calculate_statistics()

        except Exception as e:
            for query_id in batch_ids:
                logger.error(f"Query ID: {query_id}, inference error: {e}")
        finally:
            for _ in batch_ids:
                request_queue.task_done()

# 백그라운드에서 Worker 실행
worker_thread = Thread(target=batch_inference_worker, daemon=True)
worker_thread.start()

@app.route("/generate", methods=["POST"])
def generate():
    t_1 = time.time()
    data = request.json
    query_id = data.get("query_id")
    logger.info(f"Request received. Query ID: {query_id}")
    message = data.get("message")
    start_time = time.time()  # 요청 수신 시간 기록
    text = [{"role": "user", "content": data.get("question")}]  # 입력 데이터 처리
    formatted_input = get_formatted_input(text, data.get("context"))

    # 요청을 큐에 추가
    t_2 = time.time()
    # logger.info(f"Request processed. Put request to queue: {query_id}, {t_2 - t_1} sec elapsed.")
    request_queue.put((message, query_id, formatted_input, start_time))

    # 즉시 응답 반환
    t_3 = time.time()
    # logger.info(f"Put Done. Sending response: {query_id}, {t_3 - t_2} sec elapsed.")
    return jsonify({"status": "received", "query_id": query_id})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
