# AI Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, 100% offline Document Intelligence Engine and OCR Platform. Built with **FastAPI**, **SQLAlchemy**, **OpenCV**, **PyMuPDF**, **Jinja2**, **Bootstrap 5**, and powered locally by **Qwen2.5-VL-3B-Instruct** Vision-Language Model.

---

## Table of Contents
1. [Key Features](#key-features)
2. [System Requirements for Server Deployment](#system-requirements-for-server-deployment)
3. [Building & Running From Scratch (Developer Guide)](#building--running-from-scratch-developer-guide)
4. [Downloading Model Weights for Offline Deployment](#downloading-model-weights-for-offline-deployment)
5. [Production Server Deployment Guide](#production-server-deployment-guide)
6. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
7. [OCR & Extraction Engine Workflow](#ocr--extraction-engine-workflow)
8. [API Reference & cURL Examples](#api-reference--curl-examples)
9. [Running Tests](#running-tests)

---

## Key Features

- **100% Offline Local Model Execution**: Runs vision-language inference locally with zero external API calls or internet dependency.
- **Multi-Format Processing**: Supports single/batch images (`JPG`, `PNG`, `WEBP`, `TIFF`, `BMP`) and multi-page `PDF` documents.
- **Automated OpenCV Image Preprocessing**:
  - **High-Fidelity 2x Resolution Upscaling**: Automatically upscales narrow/small images (e.g. 367px mobile screenshots) via bicubic interpolation before vision encoding.
  - **Deskew & Auto-Rotation**: Corrects text alignment and orientation angles.
  - **CLAHE Contrast Enhancement**: Enhances contrast on LAB lightness channel while preserving full color.
  - **Bilateral Denoising**: Smooths noise while keeping text edges crisp.
- **Zero-Hallucination Verbatim OCR**: Reads exact pixel text from image lines without inventing external product names or extra rows.
- **Strict Row-Wise Extraction Engine**:
  - Enforces horizontal Y-band scanning from top-to-bottom and left-to-right across columns.
  - Guarantees 1-to-1 row-cell alignment (`len(row_cells) == len(columns)`), populating blank gaps with `""` to prevent cell shifting.
  - Groups multi-line item descriptions in bills/invoices into a single parent row record.
- **15+ Document Category Classification**: Automatically identifies PAN, Aadhaar, Passport, Driving License, Visiting Card, Invoice, Prescription, Form, Spreadsheet, and Unknown categories.
- **Automatic Note & UI Footer Removal**: Strips out conversational preamble, LLM `Note:` lines, and Excel UI metadata (`Sheet1`, `Accessibility`, status bars).
- **Dual-Pane Document UI & Multi-Format Exports**: Interactive document viewer with support for downloading in **TXT**, **Markdown (.md)**, **JSON**, and **CSV** formats.

---

## System Requirements for Server Deployment

### 1. Hardware Requirements

| Component | Minimum Requirements (CPU-Only / Dev) | Recommended Requirements (GPU Production) |
| :--- | :--- | :--- |
| **GPU / Acceleration** | N/A (CPU fallback) | **NVIDIA GPU** (RTX 3090, A10G, T4, L4, V100, or A100) |
| **GPU VRAM** | N/A | **8 GB VRAM** minimum (12 GB+ recommended for bfloat16) |
| **CPU** | 8 vCPUs (Intel Xeon / AMD EPYC) | 4+ vCPUs |
| **RAM (System)** | 16 GB RAM | 16 GB–32 GB RAM |
| **Disk Storage** | 30 GB NVMe SSD | 50 GB NVMe SSD |

### 2. Software & System Dependencies

- **Operating System**: Ubuntu 22.04 LTS / 20.04 LTS (Linux recommended) or Windows Server 2022.
- **Python**: Python `3.10`, `3.11`, or `3.12`.
- **NVIDIA CUDA Toolkit**: CUDA `11.8` or `12.1` with matching `cuDNN` drivers.
- **System Libraries**:
  - Linux: `sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx git curl`
  - Windows: Visual C++ Redistributable 2015-2022.

---

## Building & Running From Scratch (Developer Guide)

Follow these step-by-step instructions to set up the project environment from scratch on a new machine.

### Step 1: Clone or Create Project Repository

```bash
git clone https://github.com/your-org/OCR_Classifier.git
cd OCR_Classifier
```

### Step 2: Create and Activate Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Python Dependencies

```bash
# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (For GPU execution)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required project packages
pip install -r requirements.txt
```

---

## Model Backend Setup

The platform supports both high-performance local **Ollama** serving (recommended for production) and direct **Hugging Face Transformers** offline inference.

### Option A: Local Ollama Serving (Recommended)

1. Install and start [Ollama](https://ollama.com/).
2. Pull the 7B Vision-Language model:
   ```bash
   ollama pull qwen2.5vl:7b
   ```
3. (Optional) For high-concurrency environments on GPUs with 16GB+ VRAM, configure parallel context slots:
   - **Linux / macOS**: `export OLLAMA_NUM_PARALLEL=4`
   - **Windows**: `[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', '4', 'User')`
4. Start Ollama:
   ```bash
   ollama serve
   ```

### Option B: Offline Hugging Face Transformers Fallback

To run without Ollama, pre-download the local weights:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct
```

### Environment Configuration (.env)
Copy `.env.example` to `.env` and adjust your database and model settings:
```bash
cp .env.example .env
```

```ini
APP_NAME="AI Document Intelligence Platform"
APP_ENV="production"
DEBUG=false
HOST="0.0.0.0"
PORT=8080

# Database Connection (MySQL recommended for production, SQLite for local dev)
DATABASE_URL=mysql+pymysql://ocr_user:your_password@127.0.0.1:3306/ocr_platform?charset=utf8mb4

# Model Configurations
MODEL_BACKEND="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="qwen2.5vl:7b"
OLLAMA_TIMEOUT=180
```

---

## Production Server Deployment Guide

### Option 1: Running with Gunicorn & Uvicorn Workers (Linux Server)

For production Linux deployments behind Nginx, run Gunicorn with `uvicorn.workers.UvicornWorker`:

```bash
gunicorn app.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Option 2: Systemd Service Setup (Linux Service)

Create `/etc/systemd/system/ocr-platform.service`:

```ini
[Unit]
Description=AI Document Intelligence Engine Service
After=network.target nvidia-persistenced.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/OCR_Classifier
Environment="PATH=/home/ubuntu/OCR_Classifier/.venv/bin"
ExecStart=/home/ubuntu/OCR_Classifier/.venv/bin/gunicorn app.main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 300
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-platform
sudo systemctl start ocr-platform
sudo systemctl status ocr-platform
```

### Option 3: Nginx Reverse Proxy Configuration

Create `/etc/nginx/sites-available/ocr-platform`:

```nginx
server {
    listen 80;
    server_name ocr.yourdomain.com;
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

Enable site and reload Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ocr-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Project Architecture & Directory Structure

```
d:/OCR_Classifier/
├── app/
│   ├── api/
│   │   ├── routes_ocr.py        # Process, Reprocess, Document Details & Export APIs
│   │   ├── routes_history.py    # Document History & Full-Text Search APIs
│   │   ├── routes_analytics.py  # Analytics Metrics & Department Metrics APIs
│   │   ├── routes_settings.py   # Runtime Platform Settings API
│   │   └── routes_views.py      # Frontend Jinja2 View Controllers
│   ├── core/
│   │   ├── config.py            # Pydantic Settings & Environment Configurations
│   │   ├── database.py          # SQLAlchemy Session & Engine Provider
│   │   ├── exceptions.py        # Global Exception Handlers
│   │   ├── logger.py            # System File & Console Logger
│   │   └── security.py          # Upload File Validation & Magic Bytes Check
│   ├── models/
│   │   └── document.py          # DocumentRecord, DocumentPage, DocumentResult ORM Models
│   ├── prompts/
│   │   └── ocr_prompts.py       # Zero-Hallucination & Row-Wise Vision Prompts
│   ├── services/
│   │   ├── model_service.py     # Qwen2.5-VL Model Singleton & Repetition Penalty Engine
│   │   ├── ocr_pipeline.py      # Master OCR Workflow Orchestrator
│   │   ├── preprocessor.py      # OpenCV Deskew, 2x Resolution Upscaling & CLAHE
│   │   ├── export_service.py    # Exporter (TXT, MD, JSON, CSV)
│   │   ├── pdf_service.py       # PyMuPDF Multi-Page PDF Image Extractor
│   │   └── task_service.py     # Background Async Task Manager
│   ├── utils/
│   │   ├── file_utils.py        # Row-Wise Sanitizer & Table Converters
│   │   ├── structured_schemas.py# Enterprise Envelope Schemas
│   │   └── validator_utils.py   # Field Format Validators (PAN, Aadhaar, Dates)
│   ├── static/                  # CSS, JS, and Asset Files
│   └── templates/               # Jinja2 Frontend HTML Templates
├── uploads/                     # Original Source Document Storage
├── outputs/                     # Preprocessed & Converted Page Images
├── logs/                        # Platform Log Files
├── tests/                       # Automated Pytest Test Suite
├── requirements.txt             # Python Dependencies
├── run.py                       # Platform Server Launcher
└── README.md                    # Platform Documentation
```

---

## OCR & Extraction Engine Workflow

```mermaid
sequenceDiagram
    autonumber
    Client->>FastAPI Route: Upload Document (Image / PDF)
    FastAPI Route->>Preprocessor: Validate & Preprocess Image
    Note over Preprocessor: 1. 2x Resolution Upscale (narrow images)<br/>2. LAB CLAHE Contrast<br/>3. Bilateral Noise Filter
    Preprocessor->>Model Service: Run Pass 1 (Verbatim Row-Wise OCR)
    Model Service->>Pipeline: Extract Verbatim Markdown Table
    Pipeline->>Model Service: Run Pass 2 (Category JSON Extraction)
    Pipeline->>Utils: Sanitize Row Alignment (len(row) == len(columns))
    Utils->>Database: Persist Document Record & Results
    Database->>Client: Return Structured Envelope & Rendered Markdown
```

---

## API Reference & cURL Examples

### 1. Process Document (Synchronous Upload)

```bash
curl -X POST "http://localhost:8080/api/v1/ocr/process" \
  -F "file=@/path/to/invoice.jpg" \
  -F "document_type=AUTO" \
  -F "project_name=MedicalPortal"
```

### 2. Process Document Asynchronously

```bash
curl -X POST "http://localhost:8080/api/v1/ocr/process" \
  -F "file=@/path/to/spreadsheet.png" \
  -F "async_mode=true"
```

### 3. Fetch Document Details

```bash
curl -X GET "http://localhost:8080/api/v1/ocr/document/1"
```

### 4. Download Export Output (CSV, JSON, Markdown, TXT)

```bash
curl -X GET "http://localhost:8080/api/v1/ocr/export/1?format=csv" -o extracted_output.csv
```

---

## Running Tests

Run the complete test suite using Pytest:

```bash
# Run all tests with detailed verbosity
python -m pytest tests/ -v
```

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
