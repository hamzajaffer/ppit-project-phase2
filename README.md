# Agentic OCR Converter — Phase 2

An autonomous OCR agent that converts images to formatted Word documents using a Perceive → Decide → Act → Learn cycle.

**Course:** Professional Practices in IT (FAST-NUCES, BSAI)
**Project:** Final Project — Phase 2 (NCEAC Compliant)
**Roll Numbers:** i220583, i220554

## Features

- **Agentic pipeline**: image perception, adaptive strategy selection, OCR execution, quality gates, and memory-based learning
- **Five preprocessing strategies** (balanced, aggressive, light, handwritten, high_contrast) chosen automatically
- **Quality gate + auto-retry** with up to N untried strategies if confidence is below threshold
- **Short-term + long-term memory** that learns which strategy works best per image quality profile
- **Human-in-the-loop** override in semi-autonomous mode
- **Full audit trail** and explainability report for every conversion
- **Privacy-first**: all processing in-memory, no server-side storage

## Local Run

```bash
pip install -r requirements.txt
# On Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# On Linux:   sudo apt install tesseract-ocr
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repo** and push this folder:
   ```bash
   cd "i220583_i220554_ppit"
   git init
   git add .
   git commit -m "Phase 2: Agentic OCR Converter"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** https://share.streamlit.io and sign in with GitHub.

3. Click **"New app"** → select your repo → branch `main` → main file `app.py` → **Deploy**.

4. Streamlit Cloud will auto-install:
   - Python deps from `requirements.txt`
   - System deps from `packages.txt` (Tesseract + OpenCV runtime libs)

5. First boot takes ~3–5 minutes. Once it's live, share the public URL.

## Project Structure

```
app.py                     # Streamlit UI + pipeline orchestration
agent_orchestrator.py      # The OCR Agent (Perceive→Decide→Act→Learn)
memory_system.py           # Short-term + long-term memory
safety_logger.py           # Audit trail, explainability, data-handling info
image_preprocessor.py      # Adaptive preprocessing pipelines
ocr_engine.py              # Tesseract integration
formatting_detector.py     # Heading/bullet/alignment detection
docx_generator.py          # .docx output with agent appendix
requirements.txt           # Python dependencies
packages.txt               # System dependencies for Streamlit Cloud
.streamlit/config.toml     # App config (theme, upload limits)
```

## NCEAC Phase 2 Mapping

| Slide # | Topic | Where in code |
|---|---|---|
| 20–25 | Agentic cycle | `agent_orchestrator.py` |
| 26 | Intelligence layer | rule-based + memory in `OCRAgent.decide()` |
| 27 | Memory & context | `memory_system.py` |
| 28 | Autonomy level | `autonomy` selectbox in `app.py` |
| 29 | Human-in-the-loop | `apply_human_override()` |
| 30 | Ethical agent design | `SafetyLogger.get_data_handling_info()` |
| 31 | Risk assessment | quality gate + safety flags |
| 32 | Safety mechanisms | `safety_logger.py` |

## License

Academic project. Tesseract OCR is Apache-2.0. python-docx is MIT.
