# StructuraCAD: CAD Document Processor

AI-powered web app that ingests CAD PDFs or images, extracts structured fields, and can derive full Bills of Materials (BOM) using Google Gemini. Built with Flask + Tailwind UI, with inline editing, CSV export, and a basic contact form.

## Key Features
- CAD/PDF ingestion: upload PDF/PNG/JPG (<=10MB) with filename sanitization and size limits.
- Adaptive processing: prefers PyMuPDF for PDF-to-image, falls back to pdf2image/poppler, includes image enhancement (resize, sharpen, contrast).
- Gemini-driven extraction: Gemma-3-27B for general fields; Gemini 2.0 Flash for BOM tables, with resilient JSON parsing.
- Structured workspace: analysis page organizes fields into sections (A–D), supports adding/editing/deleting fields and saving back to JSON.
- BOM pipeline: table detection via PyMuPDF, chunked Gemini BOM extraction, CSV streaming/export, raw debug output.
- Document history & viewer: quick links to recent uploads, open original file in new tab, basic confidence heuristic.
- Q&A endpoint: `/chat/<filename>` answers questions using the extracted context (fallback replies if Gemini unavailable).
- Contact form: sends mail via Gmail SMTP (Flask-Mail) when `EMAIL_USER`/`EMAIL_PASS` are set.
- Health check: `/test-api` reports Gemini configuration status.

## Stack
- Backend: Flask, Flask-Mail, pandas, Pillow, PyMuPDF (preferred) / pdf2image, google-generativeai.
- Frontend: Tailwind (CDN) + custom styles (`static/style.css`), templates in Jinja (`templates/`).

## Project Layout
- `app.py` — core Flask app, upload/analysis/BOM/chat/contact endpoints, PDF/Image processing pipeline.
- `templates/index.html` — landing page, upload widget, client-side upload UX.
- `templates/analysis.html` — analysis workspace (sections A–D, BOM table, resizable panels, save/export).
- `templates/contact.html` — contact form page.
- `static/style.css` — custom styling.
- `uploads/` — stored uploads, `<file>.analysis.json`, and optional `<file>_bom.csv`.
- `.env.example` — required environment variables.
- `requirements.txt` — Python dependencies.

## Requirements
- Python 3.10+ recommended.
- Google Gemini API key with access to Gemma-3-27B and Gemini 2.0 Flash.
- Gmail app password for mail sending (if using contact form).
- Optional: Poppler binaries on PATH for pdf2image fallback (PyMuPDF is the preferred path).

## Setup
1. Create environment variables:
   - Copy `.env.example` to `.env` and fill `GEMINI_API_KEY`, `EMAIL_USER`, `EMAIL_PASS`.
   - Optional: `SECRET_KEY` for Flask sessions (defaults to a dev key).
2. Install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
