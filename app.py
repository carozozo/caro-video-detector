# ==============================================================================
# IMPORTS
# ==============================================================================
import streamlit as st
from typing import List, Optional, Tuple, Any
import cv2
import os
import time
import numpy as np
from openai import OpenAI
from PIL import Image
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
MODEL_NAME = "google/siglip2-so400m-patch14-384"
IMAGE_SIZE = 384
TABLE_NAME = "_caro_video_clips_2"


# ==============================================================================
# DATABASE & CACHE INITIALIZATION
# ==============================================================================
def get_pg_conn():
    """Create and return a PostgreSQL connection."""
    dsn = st.secrets.get("PG_DSN", None)
    if not dsn:
        st.error("尚未設定 PG_DSN")
        st.stop()

    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        register_vector(conn)
        return conn
    except Exception as e:
        st.error(f"PostgreSQL 連接失敗: {e}")
        st.stop()


def ensure_table(vector_size: int = 768) -> str:
    conn = get_pg_conn()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id BIGSERIAL PRIMARY KEY,
                source_id TEXT UNIQUE,
                path TEXT,
                video_name TEXT,
                timestamp INTEGER,
                embedding vector({vector_size})
            );
            """
        )
        # Create index for faster clip_score search
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
            ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_video_time_idx
            ON {TABLE_NAME} (video_name, timestamp);
            """
        )
    return TABLE_NAME


@st.cache_resource
def load_siglip_embedder() -> Optional[object]:
    from transformers import GemmaTokenizerFast, SiglipImageProcessor, SiglipModel
    import torch
    from typing import List
    from PIL import Image

    tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_NAME)
    image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipModel.from_pretrained(MODEL_NAME)

    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    class SiglipWrapper:
        def __init__(self, model, tokenizer, image_processor, device: str):
            self.model = model
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.device = device

        def encode_text(self, texts: List[str]) -> List[List[float]]:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

            return outputs.cpu().numpy().tolist()

        def encode(self, images: List[Image.Image]) -> List[List[float]]:
            # 這裡我們強制執行比例保持與補邊
            inputs = self.image_processor(
                images=images,
                return_tensors="pt",
                size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},  # 容器大小
                do_resize=True,
                resample=3,  # Bicubic 確保邊緣平滑
                do_rescale=True,  # 確保數值歸一化
                do_normalize=True,
                do_center_crop=False,  # 關閉裁剪，改用縮放
                do_pad=True,  # 補黑邊，防止拉伸變形
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # 執行 L2 歸一化，這在大規模檢索中對餘弦相似度至關重要
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return outputs.cpu().numpy().tolist()

    wrapper = SiglipWrapper(model, tokenizer, image_processor, device)
    wrapper.encode_text(["boy", "car", "tree"])  # warm up

    return wrapper


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    return OpenAI(api_key=api_key)


# ==============================================================================
# EMBEDDING UTILITIES
# ==============================================================================
def normalize_vector(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return arr.tolist()
    return (arr / norm).tolist()


def embed_frame(frame: np.ndarray, embedder) -> List[float]:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vectors = embedder.encode([img])
    vec = list(map(float, vectors[0]))
    return normalize_vector(vec)


# ==============================================================================
# FILE & DATA MANAGEMENT
# ==============================================================================
def save_uploaded_files_to_tmp(uploaded_files, tmp_dir: str = "tmp") -> List[str]:
    os.makedirs(tmp_dir, exist_ok=True)

    saved_paths: List[str] = []
    for uploaded in uploaded_files:
        if getattr(uploaded, "name", None) is None:
            raise ValueError("uploaded file has no name")
        safe_name = "".join([c if c.isalnum() else "_" for c in uploaded.name])
        temp_path = os.path.join(tmp_dir, f"{safe_name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        saved_paths.append(temp_path)

    return saved_paths


def clear_database() -> None:
    conn = get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE_NAME};")
    st.success("資料庫內容已清空")


def view_database_data() -> List[dict]:
    conn = get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, source_id, path, video_name, timestamp
            FROM {TABLE_NAME}
            ORDER BY video_name, timestamp
            LIMIT 1000;
            """
        )
        rows = cur.fetchall()
    conn.close()

    records: List[dict] = []
    for row in rows:
        record = {
            "id": row[0],
            "source_id": row[1],
            "path": row[2],
            "video_name": row[3],
            "timestamp_seconds": int(row[4]),
            "timestamp_hms": sec_to_hms(row[4]),
        }
        records.append(record)
    return records


# ==============================================================================
# VIDEO PROCESSING
# ==============================================================================
def analyze_videos(files: List[str]) -> None:
    for file in files:
        st.write(f"影片 {file}")

        embedder = load_siglip_embedder()
        motion_threshold = st.session_state.get("motion_threshold", 0.03)

        start_total = time.time()
        frames_data = extract_and_embed_frames(
            file, embedder, sample_fps=1, motion_threshold=motion_threshold
        )
        vector_size = len(frames_data[0]["vector"]) if frames_data else 768
        table_name = ensure_table(vector_size)
        save_to_database(file, frames_data, output_root="frames")

        total_elapsed = time.time() - start_total
        st.success(f"耗時 {total_elapsed:.2f} 秒")


def is_static_frame(
    frame: np.ndarray, prev_gray: Optional[np.ndarray], threshold: float
) -> Tuple[bool, Optional[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        frame_delta = cv2.absdiff(prev_gray, gray)
        changed_ratio = float(np.mean(frame_delta > 15))

        if changed_ratio < float(threshold):
            return True, gray

    return False, gray


def extract_and_embed_frames(
    video_path: str,
    embedder: Any,
    sample_fps: int = 1,
    motion_threshold: float = 0.03,
    batch_size: int = 8,
) -> List[dict]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / max(1, sample_fps))))

    frames_data = []
    frame_idx = 0
    progress = st.progress(0)

    enable_motion_filter = float(motion_threshold) > 0.0
    prev_gray = None

    # 批次暫存區
    batch_images = []
    batch_metas = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        timestamp = int(round(frame_idx / max(1.0, src_fps)))

        if enable_motion_filter:
            is_static, prev_gray = is_static_frame(frame, prev_gray, motion_threshold)
            if is_static:
                frame_idx += 1
                continue

        # 將 OpenCV BGR 轉為 PIL Image (SigLIP 預期格式)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 暫存到批次
        batch_images.append(pil_img)
        batch_metas.append({"frame": frame, "timestamp": timestamp})

        # 當達到 batch_size 或影片結束，進行一次性 Embedding
        if len(batch_images) == batch_size:
            vectors = embedder.encode(batch_images)  # 一次餵 8 張
            for i, vec in enumerate(vectors):
                frames_data.append(
                    {
                        "frame": batch_metas[i]["frame"],
                        "timestamp": batch_metas[i]["timestamp"],
                        "vector": vec,
                    }
                )
            batch_images = []
            batch_metas = []

        frame_idx += 1
        progress.progress(min(100, int((frame_idx / max(1, total_frames)) * 100)))

    # 處理剩餘不足一個 batch 的影格
    if batch_images:
        vectors = embedder.encode(batch_images)
        for i, vec in enumerate(vectors):
            frames_data.append(
                {
                    "frame": batch_metas[i]["frame"],
                    "timestamp": batch_metas[i]["timestamp"],
                    "vector": vec,
                }
            )

    cap.release()
    progress.empty()
    return frames_data


def save_to_database(
    video_path: str,
    frames_data: List[dict],
    output_root: str = "frames",
) -> None:
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    base_name = os.path.basename(video_path)
    safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
    frames_dir = os.path.join(output_root, safe_name)
    os.makedirs(frames_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    rows = []

    for idx, data in enumerate(frames_data):
        filename = f"frame_{idx:06d}.jpg"
        frame_path = os.path.join(frames_dir, filename)
        cv2.imwrite(frame_path, data["frame"])

        source_id = f"{safe_name}_{filename}"
        timestamp = int(data["timestamp"])
        embedding = data["vector"]

        rows.append((source_id, frame_path, video_name, timestamp, embedding))

    if not rows:
        st.warning("沒有要保存的幀")
        return

    # Insert into PostgreSQL
    conn = get_pg_conn()
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {TABLE_NAME}
                (source_id, path, video_name, timestamp, embedding)
            VALUES %s
            ON CONFLICT (source_id) DO UPDATE SET
                path = EXCLUDED.path,
                video_name = EXCLUDED.video_name,
                timestamp = EXCLUDED.timestamp,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.close()


# ==============================================================================
# SEARCH & QUERY PROCESSING
# ==============================================================================
def execute_simple_similarity_search(
    cur, vec: List[float], min_score: float, n_results: int
) -> List[tuple]:
    cur.execute(
        f"""
         SELECT id,
             source_id,
             path,
             video_name,
             timestamp,
             clip_score
        FROM (
            SELECT id,
                   source_id,
                   path,
                   video_name,
                   timestamp,
                 1 - (embedding <=> %s::vector) AS clip_score
            FROM {TABLE_NAME}
        ) t
         WHERE clip_score >= %s
         ORDER BY clip_score DESC
        LIMIT %s
        """,
        (vec, float(min_score), n_results),
    )
    return cur.fetchall()


def execute_time_gap_search(
    cur, vec: List[float], n_results: int, min_score: float, query_time_gap: int
) -> List[tuple]:
    cur.execute(
        f"""
        WITH RECURSIVE ranked AS (
            SELECT id,
                   source_id,
                   path,
                   video_name,
                   timestamp,
                   1 - (embedding <=> %s::vector) AS clip_score
            FROM {TABLE_NAME}
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY clip_score DESC
            LIMIT %s
        ),
        seed AS (
            SELECT DISTINCT ON (video_name)
                   id,
                   source_id,
                   path,
                   video_name,
                   timestamp,
                   clip_score
            FROM ranked
            ORDER BY video_name, timestamp, id
        ),
        rec AS (
            SELECT * FROM seed
            UNION ALL
            SELECT r2.*
            FROM rec r
            JOIN LATERAL (
                SELECT *
                FROM ranked r2
                WHERE r2.video_name = r.video_name
                  AND r2.timestamp >= r.timestamp + %s
                  AND r2.clip_score >= %s
                ORDER BY r2.timestamp, r2.id
                LIMIT 1
            ) r2 ON true
        )
        SELECT id, source_id, path, video_name, timestamp, clip_score
        FROM rec
        ORDER BY clip_score DESC
        LIMIT %s;
        """,
        (
            vec,
            vec,
            float(min_score),
            n_results,
            int(query_time_gap),
            float(min_score),
            n_results,
        ),
    )
    return cur.fetchall()


def expand_query_via_llm(query_text: str) -> List[str]:
    if not st.session_state.get("llm_expansion", True):
        return [query_text]

    import json

    openai = get_openai_client()

    prompt = (
        "Generate concise visual search paraphrases for the query. "
        "Return ONLY a JSON array of unique English phrases. "
        f"Max 3 items. Query: {query_text}"
    )

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.choices[0].message.content.strip()

    if text.startswith("```"):
        text = text.strip("`")

    if text.lower().startswith("json"):
        text = text[4:].strip()

    phrases = json.loads(text)
    return [str(p).strip() for p in phrases[:3]]


def search_database(
    query_text: str,
    wrapper,
    query_time_gap: int = 0,
    min_score: float = 0.0,
    n_results: int = 300,
) -> Tuple[List[dict], dict]:
    t0_total = time.perf_counter()
    t_embed = 0.0
    db_time = 0.0

    if int(query_time_gap) != 0:
        t0 = time.perf_counter()

    terms = expand_query_via_llm(query_text)

    st.write(f"搜尋 '{', '.join(terms)}'")

    if len(terms) == 0:
        stats = {
            "db_query_time_s": db_time,
            "embed_time_s": t_embed,
            "total_time_s": time.perf_counter() - t0_total,
        }
        return [], stats

    t0 = time.perf_counter()
    vecs = wrapper.encode_text(terms)
    vecs = [normalize_vector(list(map(float, v))) for v in vecs]
    t_embed += time.perf_counter() - t0

    records = []
    conn = get_pg_conn()
    t0 = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 256;")
        for vec in vecs:
            if int(query_time_gap) > 0:
                rows = execute_time_gap_search(
                    cur, vec, n_results, min_score, query_time_gap
                )
            else:
                rows = execute_simple_similarity_search(cur, vec, min_score, n_results)

            for row in rows:
                records.append(
                    {
                        "id": row[0],
                        "source_id": row[1],
                        "path": row[2],
                        "video_name": row[3],
                        "timestamp": row[4],
                        "timestamp_hms": sec_to_hms(row[4]),
                        "clip_score": float(row[5]),
                    }
                )

        conn.close()
        db_time = time.perf_counter() - t0

        # Deduplicate and sort by clip_score
        unique_records = {}
        for record in records:
            key = record["path"]
            if record["clip_score"] >= float(min_score):
                if (
                    key not in unique_records
                    or record["clip_score"] > unique_records[key]["clip_score"]
                ):
                    unique_records[key] = record

        final_records = sorted(
            unique_records.values(), key=lambda r: r["clip_score"], reverse=True
        )

    stats = {
        "db_query_time_s": db_time,
        "embed_time_s": t_embed,
        "total_time_s": time.perf_counter() - t0_total,
    }
    return final_records, stats


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def sec_to_hms(sec: int) -> str:
    try:
        sec = int(sec)
    except Exception:
        return "00:00:00"
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==============================================================================
# UI RENDERING FUNCTIONS
# ==============================================================================


def configure_page() -> None:
    st.set_page_config(page_title="影片上傳 AI 分析", layout="wide")


def render_sidebar() -> Optional[List[str]]:
    with st.sidebar:
        st.header("資料庫")
        col_db_1, col_db_2 = st.columns([1, 3], gap="small")
        with col_db_1:
            st.button("清空", key="clear_db", on_click=clear_database)
        with col_db_2:
            st.button("檢視", key="view_db", on_click=show_db_dialog)

        st.divider()

        st.header("上傳影片")

        motion_threshold = st.slider(
            "變動像素比例(小於此值視為靜態)", 0.00, 0.05, 0.03, step=0.01
        )
        st.session_state["motion_threshold"] = motion_threshold

        if "upload_counter" not in st.session_state:
            st.session_state["upload_counter"] = 0

        uploader_key = f"uploaded_files_{st.session_state['upload_counter']}"
        uploaded_files = st.file_uploader(
            "選擇影片檔案",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            key=uploader_key,
        )

        if uploaded_files:
            col1, col2 = st.columns([1, 3], gap="small")
            with col1:
                analyze_clicked = st.button("分析", key="analyze")

            def clear_uploader() -> None:
                st.session_state["upload_counter"] = (
                    st.session_state.get("upload_counter", 0) + 1
                )

            with col2:
                st.button("清除", key="clear", on_click=clear_uploader)

            if analyze_clicked:
                saved_paths = save_uploaded_files_to_tmp(uploaded_files)
                if saved_paths:
                    analyze_videos(saved_paths)
                else:
                    st.warning("未上傳檔案")

    return None


def render_results_grid(records: List[dict], cols_per_row: int = 3) -> None:
    total = len(records)
    for i in range(0, total, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, rec in enumerate(records[i : i + cols_per_row]):
            col = cols[j]
            img_path = rec.get("path")
            if img_path and os.path.exists(img_path):
                col.image(img_path, width=300)

            video_name = rec.get("video_name", "")
            ts = rec.get("timestamp_hms", "")
            clip_score = rec.get("clip_score", 0.0)

            col.markdown(f"**{video_name}**  \n{ts}  \n相似度: {clip_score:.3f}")


def render_options_panel() -> Tuple[int, float]:
    llm_expansion = st.selectbox(
        "LLM 近義詞擴展", ["停用 LLM", "啟用 LLM"], index=1  # 預設為 "啟用 LLM"
    )
    st.session_state["llm_expansion"] = llm_expansion == "啟用 LLM"

    query_time_gap = st.slider("優化搜尋間隔(秒)", 0, 12, 3)
    st.session_state["query_time_gap"] = int(query_time_gap)

    min_score = st.slider("最低相似度", 0.0, 0.1, 0.09, step=0.01)
    st.session_state["min_score"] = float(min_score)

    return int(query_time_gap), float(min_score)


def render_search_pane() -> None:
    query_time_gap, min_score = render_options_panel()

    with st.form("search_form"):
        query_text = st.text_input(
            "輸入搜尋項目", placeholder="物件/場景描述", key="search_query"
        )
        submitted = st.form_submit_button("搜尋")

    if submitted and query_text.strip():
        with st.spinner("搜尋中..."):
            wrapper = load_siglip_embedder()
            records, stats = search_database(
                query_text, wrapper, query_time_gap, min_score
            )

        total = len(records)
        st.write(
            f"找到 {total} 筆 — 總耗時 {stats.get('total_time_s', 0):.2f}s "
            f"(DB: {stats.get('db_query_time_s', 0):.2f}s, Embed: {stats.get('embed_time_s', 0):.2f}s)"
        )

        if records:
            records.sort(key=lambda r: r.get("clip_score", 0), reverse=True)
            render_results_grid(records)
        else:
            st.info("未找到相關結果")


@st.dialog("資料庫內容檢視", width="large")
def show_db_dialog():
    data = view_database_data()

    if data:
        st.write(f"總筆數: {len(data)}")
        st.dataframe(data, use_container_width=True)
    else:
        st.write("無資料")

    if st.button("關閉"):
        st.rerun()


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main() -> None:
    configure_page()
    render_sidebar()
    render_search_pane()


if __name__ == "__main__":
    main()
