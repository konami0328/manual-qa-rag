"""
Load test for the Tesla Model Y RAG QA system.

Target endpoint: POST /ask_benchmark
Test data:       data/qa_pairs/test.jsonl (questions sampled at startup)

Metrics collected per request:
    - end-to-end response time (Locust built-in)
    - retrieve_ms, rerank_ms, generate_ms (from response JSON)

Custom timings are reported via locust.events.request so they appear
as separate entries in the Locust stats table.

Usage:
    locust -f app/load_test/locustfile.py \
           --host http://localhost:8001 \
           --users 1 --spawn-rate 1 --run-time 3m

    locust -f app/load_test/locustfile.py \
           --host http://localhost:8001 \
           --users 5 --spawn-rate 1 --run-time 3m

    locust -f app/load_test/locustfile.py \
           --host http://localhost:8001 \
           --users 10 --spawn-rate 1 --run-time 3m
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Load questions once at module level (shared across all users)
# ---------------------------------------------------------------------------

_TEST_PATH = os.path.join(
    os.path.dirname(__file__),   # app/load_test/
    "..", "..",                   # project root
    "data", "qa_pairs", "test.jsonl",
)
_TEST_PATH = os.path.normpath(_TEST_PATH)

_questions: list[str] = []

def _load_questions(path: str) -> list[str]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("question", "").strip()
            if q:
                questions.append(q)
    return questions


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global _questions
    _questions = _load_questions(_TEST_PATH)
    print(f"[locust] Loaded {len(_questions)} questions from {_TEST_PATH}")


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class RAGUser(HttpUser):
    """
    Simulates a user sending questions to /ask_benchmark.

    wait_time: 1-3 seconds between requests per user, to avoid
               pure hammering and better simulate real usage.
    """
    wait_time = between(1, 3)

    @task
    def ask(self):
        question = random.choice(_questions)

        t_start = time.perf_counter()
        with self.client.post(
            "/ask_benchmark",
            json         = {"question": question},
            catch_response = True,
        ) as resp:
            t_end = time.perf_counter()

            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"JSON parse error: {e}")
                return

            resp.success()

            # --- report per-layer timings as separate Locust stats entries ---
            timings = data.get("timings", {})
            for layer in ("retrieve_ms", "rerank_ms", "generate_ms"):
                ms = timings.get(layer)
                if ms is None:
                    continue
                events.request.fire(
                    request_type = "LAYER",
                    name         = layer,
                    response_time = ms,
                    response_length = 0,
                    exception    = None,
                    context      = {},
                )