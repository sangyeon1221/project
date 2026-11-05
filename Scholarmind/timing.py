"""
timing.py
---------
질문 1회를 단계별로 재고(화면 표시용 표), 매번 logs/timings.jsonl에 한 줄씩 남긴다.
코드 버전은 소스 파일 내용의 해시라서, 코드를 고친 전후를 eval/compare_timings.py로
바로 비교할 수 있다. 화면 녹화를 프레임 단위로 보며 시간을 재던 일을 대신한다.
"""

import glob
import hashlib
import json
import os
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "logs", "timings.jsonl")


def code_version() -> str:
    """루트의 .py 파일 내용 해시 앞 8자리 — 한 글자라도 바뀌면 다른 버전이 된다."""
    h = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(ROOT, "*.py"))):
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:8]


class StageTimer:
    """lap(이름)을 부를 때마다 직전 lap 이후 걸린 시간을 그 단계의 시간으로 기록한다."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.start = self._last = time.perf_counter()
        self.laps: list = []

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps.append((name, now - self._last))
        self._last = now

    @property
    def total(self) -> float:
        return self._last - self.start

    def markdown(self) -> str:
        rows = "\n".join(f"| {n} | {s:.1f}초 |" for n, s in self.laps)
        return f"| 단계 | 시간 |\n|---|---:|\n{rows}\n| **합계** | **{self.total:.1f}초** |"

    def save(self, question: str) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "version": code_version(),
            "gpu": os.getenv("OLLAMA_NUM_GPU", "") != "0",
            "kind": self.kind,
            "question": question,
            "laps": {n: round(s, 2) for n, s in self.laps},
            "total": round(self.total, 2),
        }
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 기록 실패가 답변을 막으면 안 된다
