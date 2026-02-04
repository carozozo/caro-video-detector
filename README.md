# caro-video-detector

A Streamlit-based video analysis application that uses SigLIP embeddings and PostgreSQL vector search to find semantically similar video clips.

## Features

- **Video Processing**: Extract frames from uploaded videos with motion detection filtering
- **Semantic Search**: Query video content using natural language descriptions
- **Vector Embeddings**: Uses SigLIP2 model for generating image embeddings
- **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
- **LLM Query Expansion**: OpenAI integration to expand queries into multiple search phrases
- **Time-Gap Optimization**: Find clips with temporal constraints for narrative continuity
- **Batch Processing**: Efficient frame embedding with configurable batch sizes

## Requirements

- Python 3.8+
- PostgreSQL with pgvector extension
- FFmpeg/OpenCV for video processing
- Streamlit
- PyTorch
- transformers (for SigLIP model)
- OpenAI API key

## Installation

```bash
pip install streamlit opencv-python pillow psycopg2-binary pgvector
pip install transformers torch openai
```

## Configuration

Set the following secrets in Streamlit config:
- `PG_DSN`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key for query expansion

## Usage

Run the application:
```bash
. start_streamlit.sh
```

### Upload Videos
1. Use the sidebar file uploader to select MP4, AVI, or MOV files
2. Adjust motion detection threshold (default: 0.03)
3. Click "分析" (Analyze) to process videos

### Search
1. Enter search query in natural language
2. Configure options:
   - Enable/disable LLM query expansion
   - Set time gap optimization (seconds)
   - Set minimum similarity threshold
3. Click "搜尋" (Search) to find similar clips

### Database Management
- View database contents via "檢視" (View) button
- Clear all data via "清空" (Clear) button

## Key Components

### Video Processing
- `extract_and_embed_frames()`: Extracts frames and generates embeddings
- `is_static_frame()`: Filters out static frames based on motion threshold
- Motion detection uses optical flow analysis

### Embedding
- `load_siglip_embedder()`: Loads SigLIP2 model with proper normalization
- `embed_frame()`: Converts video frames to embeddings
- L2 normalization for cosine similarity search

### Search
- `search_database()`: Main search function with LLM expansion
- `execute_simple_similarity_search()`: Direct similarity search via HNSW index
- `execute_time_gap_search()`: Recursive search with temporal constraints

### Database
- PostgreSQL table: `_caro_video_clips_2`
- HNSW index for fast vector similarity search
- Supports batch insertions with conflict resolution

## Performance Optimization

- HNSW index with parameters: m=16, ef_construction=64
- Batch embedding processing (default batch_size=8)
- EF search parameter: 256 for balanced accuracy/speed
- Frame sampling via configurable FPS
