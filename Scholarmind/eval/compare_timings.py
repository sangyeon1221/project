"""
eval/compare_timings.py
-----------------------
logs/timings.jsonl을 코드 버전별로 묶어 단계별 중앙값을 비교한다.

    python eval/compare_timings.py            # 새 검색 기록만
    python eval/compare_timings.py --all      # 후속 질문·다시 답하기 포함
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timing import LOG_PATH  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if not os.path.exists(LOG_PATH):
    sys.exit("기록이 없습니다 — 앱에서 질문을 한 번 해 보세요.")

with open(LOG_PATH, encoding="utf-8") as fh:
    records = [json.loads(line) for line in fh if line.strip()]
if "--all" not in sys.argv:
    records = [r for r in records if r["kind"] == "새 검색"]

versions, stages = {}, []
for r in records:  # 처음 나온 순서대로 버전과 단계 이름을 모은다
    versions.setdefault(r["version"], []).append(r)
    stages += [s for s in r["laps"] if s not in stages]

head = ["단계"] + [f"{v} (n={len(rs)})" for v, rs in versions.items()]
print(" | ".join(head))
for name in stages + ["합계"]:
    cells = []
    for rs in versions.values():
        vals = [r["total"] if name == "합계" else r["laps"].get(name) for r in rs]
        vals = [v for v in vals if v is not None]
        cells.append(f"{statistics.median(vals):.1f}초" if vals else "-")
    print(" | ".join([name] + cells))
