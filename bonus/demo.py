"""Demo: 5 queries against HybridMemoryAgent, printing assembled context each time.

Maps to the 5 scenarios in BONUS-CHALLENGE.md:
  1. simple (vector hit)        4. paraphrase (vector wins)
  2. needs profile context      5. mixed (hybrid + profile)
  3. needs fresh activity

Run:  python bonus/demo.py     (exits 0 on success)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402


SEED_MEMORIES = [
    "Tôi đã đọc một bài về Kubernetes auto-scaling, cách Pod tự mở rộng theo lưu lượng và quản lý vòng đời container trong production.",
    "Ghi chú: serverless Lambda giúp tối ưu chi phí khi traffic thất thường, chỉ trả tiền cho thời gian chạy thực tế.",
    "Bài viết về zero-trust security và OAuth JWT: xác thực hai yếu tố cho mọi request, không tin tưởng mặc định mạng nội bộ.",
    "Tôi tìm hiểu cách tự động mở rộng hạ tầng theo số người dùng tăng đột biến mà không cần can thiệp thủ công.",
    "Note về mã hoá dữ liệu nhạy cảm: encryption at rest cho database và TLS trên đường truyền cho mọi API.",
    "Đọc về CDN edge caching để giảm độ trễ cho video streaming ở nhiều khu vực địa lý khác nhau.",
]

QUERIES = [
    ("1. simple (vector hit)",        "Tôi đã đọc gì về Kubernetes?"),
    ("2. needs profile context",      "Recommend đọc gì tiếp cho tôi"),
    ("3. needs fresh activity",       "Tôi đang quan tâm gì gần đây?"),
    ("4. paraphrase (vector wins)",   "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. mixed (hybrid + profile)",   "Cho tôi summary cloud security"),
]


def main() -> int:
    print("Bonus demo — HybridMemoryAgent (episodic vector memory + Feast profile)")
    print("=" * 72)
    agent = HybridMemoryAgent()

    user_id = "u_001"
    for m in SEED_MEMORIES:
        agent.remember(m, user_id=user_id)
    print(f"Seeded {len(agent.memories)} memory chunks for {user_id}\n")

    for label, q in QUERIES:
        print(f"── Query {label}")
        print(agent.recall(q, user_id=user_id))
        print()

    print("=" * 72)
    print("PASS — 5 queries answered, context assembled for each.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
