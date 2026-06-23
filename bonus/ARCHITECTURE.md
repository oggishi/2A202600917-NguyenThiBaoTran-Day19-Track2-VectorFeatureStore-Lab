# Bonus — Hybrid Memory cho Trợ lý AI cá nhân (người dùng VN)

**Contributors:** solo submission.
**POC:** `agent.py` (`HybridMemoryAgent.remember()` / `.recall()`) + `demo.py` (5 queries).
**Tie-in lab 19:** Vector Store + RRF (§1–§3), Feast feature views + TTL + PIT join (§4/§6).

Mục tiêu: một trợ lý "nhớ" được 3 loại thông tin với 3 vòng đời rất khác nhau —
**episodic memory** (cuộc hội thoại / tài liệu đã đọc, tăng liên tục), **stable
profile** (ngôn ngữ, tốc độ đọc, lĩnh vực quan tâm — đổi chậm), và **recent
activity** (query 1 giờ qua — đổi từng giây). Sai lầm phổ biến là nhét cả 3 vào
một store; bài này tách theo *đặc tính truy cập + vòng đời dữ liệu*.

## Sơ đồ kiến trúc

```
                         ┌──────────────────────────────────────┐
   user query  ─────────▶│        HybridMemoryAgent.recall()     │
   (vi/en mix)           └───────────────┬──────────────────────┘
                                         │
          ┌──────────────────────────────┼───────────────────────────────┐
          ▼                              ▼                                ▼
 ┌──────────────────┐        ┌────────────────────────┐      ┌────────────────────────┐
 │  EPISODIC MEMORY │        │   STABLE PROFILE        │      │   RECENT ACTIVITY       │
 │  Qdrant (vector) │        │   Feast online store    │      │   Feast streaming view  │
 │  + BM25 (sparse) │        │   user_profile_features │      │   query_velocity_*      │
 │  RRF k=60        │        │   TTL 30d · daily batch │      │   TTL 1h · push/stream  │
 │  filter:user_id  │        │   reading_speed_wpm,    │      │   queries_last_hour,    │
 │                  │        │   preferred_language,   │      │   distinct_topics_24h   │
 │                  │        │   topic_affinity        │      │                         │
 └────────┬─────────┘        └───────────┬────────────┘      └───────────┬────────────┘
          │ top-3 memories               │ tabular features              │ velocity
          └──────────────────────────────┴───────────────────────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │  assemble context string      │  ──▶  (LLM prompt)
                          │  "User likes <affinity>,      │
                          │   reads <wpm>wpm, recent:      │
                          │   <queries_last_hour>q/h.      │
                          │   Top memories: …"             │
                          └──────────────────────────────┘
```

**Write path** (`remember()`): text → chunk → embed (fastembed bge-small) →
upsert vào Qdrant với `payload.user_id`. Đây là đường ghi *đồng bộ* — recall
phản ánh ngay (sub-second), không qua materialize.
**Profile/activity path:** offline (Parquet/warehouse) → `feast materialize`
→ online store; recall đọc bằng `get_online_features()`. Training-time đọc bằng
`get_historical_features()` (PIT join) để tránh leakage.

## 3 quyết định kiến trúc (tradeoff explicit)

### 1. Chunking strategy — câu-aware ~60 token, **không** per-conversation, **không** semantic-break
Episodic memory chunk thế nào quyết định *retrieval quality vs storage cost vs context window*.
- **Per-message** (1 vector / message): recall granularity cao nhưng vỡ ngữ cảnh
  liên câu và phình số vector.
- **Per-conversation** (1 vector / cả hội thoại): storage rẻ, ít vector — nhưng
  một embedding 384-chiều **không** biểu diễn nổi 5 chủ đề trong 1 hội thoại
  2000-token; cosine signal bị pha loãng → recall tụt.
- **Semantic-break** (cắt theo khoảng cách embedding): chất lượng tốt nhất nhưng
  cần *thêm một lượt embed mỗi lần ingest* + threshold phải tune → latency + phức tạp.

**Chọn:** cắt theo ranh giới câu, gộp tới ~60 token/chunk. Mỗi chunk gần như đơn-chủ-đề
(cosine sạch), storage bị chặn (~1 vector / 2–3 câu), và recall nhét được nhiều
chunk vào context window 8k. **VN-context:** *không* split theo whitespace để
đếm token vì từ tiếng Việt đa âm tiết ("cơ sở dữ liệu" = 3 token whitespace nhưng
1 khái niệm); cắt theo dấu câu (`. ! ? \n`) — script-agnostic, không gãy giữa từ.

### 2. Feature schema — tabular features, **không** embedding-of-history
`user_profile_features` (entity `user`, TTL 30d, source Parquet→warehouse):
`reading_speed_wpm:Int64`, `preferred_language:String`, `topic_affinity:String`.
`query_velocity_features` (entity `user`, TTL 1h, streaming): `queries_last_hour:Int64`,
`distinct_topics_24h:Int64`.
- **Embedding feature** (1 vector học từ lịch sử đọc): bắt được sở thích *latent*,
  cho phép personalization theo similarity — nhưng là black box, cần training
  pipeline + re-embed khi drift, và **không trả lời được** "vì sao gợi ý cái này?"
  (không debug nổi từ vector 384-chiều).
- **Tabular**: interpretable, debug được, rẻ, materialize thẳng — nhưng không bắt
  được nuance.

**Chọn tabular** vì với trợ lý cá nhân, *trust + explainability* là tính năng:
user hỏi "trợ lý biết gì về tôi?" thì phải trả lời được bằng chữ. Để ngỏ cửa
thêm 1 embedding feature view sau khi có đủ data.

### 3. Freshness strategy — 3 tier theo use case (gắn với TTL)
| Use case | Cơ chế | Độ trễ | Lý do |
|---|---|---|---|
| "Trợ lý nhớ tài liệu tôi vừa đọc?" | ghi đồng bộ Qdrant (write path) | sub-second | recall phải thấy ngay; không qua materialize |
| Recent activity / phát hiện mệt mỏi (query đêm dài hơn) | streaming push → `query_velocity` | giây | velocity cũ = mất tín hiệu real-time; TTL 1h |
| Profile (tốc độ đọc, affinity, ngôn ngữ) | daily batch | ngày | đổi chậm; sub-second là lãng phí compute; TTL 30d |

**Tradeoff:** streaming-cho-tất-cả = mental model đơn giản nhưng infra đắt (Kafka+push)
và vô nghĩa với attr ổn định; daily-cho-tất-cả = rẻ nhưng `query_velocity` thành vô dụng.
Vì vậy chọn *freshness theo từng feature view*, đúng bài học TTL của lab.

## Lựa chọn đã loại bỏ (rejected alternative)

> **Tôi xem xét lưu episodic memory *trong* feature store** (dưới dạng embedding
> feature view) **nhưng tách riêng sang Qdrant vì:** (a) cadence refresh khác hẳn —
> memory mới mỗi vài phút vs profile mỗi tuần, mô hình `materialize` của Feast giả
> định batch các entity-keyed row, không hợp với text tự do tăng liên tục; (b) ANN
> search trên memory bất kỳ cần HNSW index — online store của Feast (SQLite/Redis KV)
> chỉ là key→value lookup, không làm được similarity search.

Lựa chọn loại bỏ thứ 2: **per-user Qdrant collection** (vs single collection +
`user_id` payload filter) — bỏ cho POC vì hàng nghìn collection thêm overhead quản
lý, còn payload filter trên field keyword đã index thì đủ nhanh. *Sẽ* đáng đổi sang
per-user/per-tenant collection khi cần **hard isolation** để xoá dữ liệu đảm bảo theo
Nghị định 13/2023 (PDPD).

## Vietnamese-context considerations
- **Code-switching:** query thật trộn vi/en ("Cho tôi summary cloud security").
  bge-small là English-trained → recall trên paraphrase thuần Việt yếu (lab NB2 cho
  thấy ~24–32%). Đó chính là lý do **hybrid quan trọng *hơn* cho tiếng Việt**: nhánh
  BM25 bắt được token kỹ thuật tiếng Anh khi nhánh vector hụt.
- **Tokenizer:** whitespace over-split từ ghép VN ("cơ sở dữ liệu" → 3 token) làm
  loãng BM25 precision; production VN nên dùng `underthesea`/`pyvi` word-segmentation
  (đổi lấy thêm dependency + latency — chấp nhận ở quy mô lớn).
- **Privacy (Nghị định 13/2023):** thói quen đọc + lịch sử query là personal data,
  phải xoá được → ủng hộ thiết kế filter-by-user (hard-delete được) và **không** nướng
  lịch sử user vào một embedding model dùng chung.

## What this POC doesn't handle yet (honest limitations)
- Không encryption at rest; `user_id` là input tin cậy (spoof được) — chưa có auth.
- Chưa có CRUD/edit/delete memory, chưa multi-device sync.
- Chưa memory decay / consolidation (gộp memory cũ thành summary).
- BM25 rebuild mỗi lần recall — O(N), ổn cho POC, không ổn với 10k+ memory (cần
  index sparse bền vững, ví dụ Qdrant sparse vectors).
- Feast online lookup có fallback synthetic nếu chưa `materialize` (để demo luôn
  exit 0); production thì không fallback mà phải fail loud.

## Vibe-coding log (~100 từ)
- **Prompt hiệu quả nhất:** đưa *công thức RRF tường minh* + payload schema +
  nhấn "rank 1-based" → fusion loop đúng ngay lần đầu. Spec-in beats vibe-out.
- **Prompt fail:** "thiết kế feature schema cho tôi" mà không nói entity/TTL/freshness
  → ra cột generic, không có lý do TTL. Phải tự nghĩ business semantics (cái nào
  đổi nhanh, cái nào chậm) rồi mới giao phần boilerplate. Đúng cảnh báo của lab:
  TTL/timestamp là *think-hard*, không phải delegate.
