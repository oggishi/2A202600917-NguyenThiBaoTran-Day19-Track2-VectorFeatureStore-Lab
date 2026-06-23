# Reflection — Lab 19

**Tên:** _< Nguyen Thi Bao Tran>_
**MSSV:** _< A20-K1 / A20-K2 / 2A202600917>_
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Kết quả lite (P@10): kw 77.8% · sem 73.2% · **hyb 78.6%**.

- **`exact` (96.7% kw = hyb > 88.7% sem):** từ kỹ thuật xuất hiện verbatim
  trong corpus nên BM25 lexical match đủ mạnh; vector thêm noise. Hybrid hoà
  với BM25 vì keyword signal đã đè RRF.
- **`paraphrase` (kw 33% > hyb 32% > sem 24%):** bất ngờ — semantic *thấp
  nhất*. `bge-small-en-v1.5` là model English-trained, recall yếu trên
  paraphrase thuần Việt; BM25 còn vớ được vài token trùng. Đổi sang `bge-m3`
  (Docker path) sẽ đảo ngược, semantic thắng.
- **`mixed` (hyb 100% > sem 98.5% > kw 97%):** hybrid thắng rõ vì RRF fuse
  exact-term hit (BM25) với conceptual hit (vector) — đúng hình dạng query thật.

Hybrid thắng *trung bình* nhờ robust, không bao giờ tệ nhất.

**Không dùng hybrid khi:** (1) lookup mã/ID chính xác (SKU, error code) — BM25
một mình nhanh và ngang điểm; (2) đường latency-critical — chạy 2 retriever +
fusion gấp đôi chi phí cho lợi ích biên; (3) embedding model không hợp ngôn ngữ
(ở đây English model trên tiếng Việt) — vector đóng góp ít, dùng BM25 thuần đến
khi nâng cấp model.

---

## Điều ngạc nhiên nhất khi làm lab này

Semantic lại *kém nhất* trên paraphrase tiếng Việt — ngược với slogan "vector
beats keyword". Bài học: language coverage của embedding model quan trọng hơn
việc "có dùng vector hay không". Một model sai ngôn ngữ còn tệ hơn BM25.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/` — `ARCHITECTURE.md`, `agent.py`, `demo.py`)
- [ ] Pair work với: _<solo>_
