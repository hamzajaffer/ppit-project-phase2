"""
Image-to-Word Converter — Phase 2: Agentic System
Autonomous OCR agent with Perceive → Decide → Act → Learn cycle.
"""

import streamlit as st
from PIL import Image
import os, time

from image_preprocessor import preprocess_image, resize_for_ocr
from ocr_engine import extract_text_simple, extract_text_with_data, group_words_into_lines, lines_to_text_blocks
from formatting_detector import detect_formatting
from docx_generator import generate_docx, generate_plain_docx
from agent_orchestrator import OCRAgent, STRATEGIES
from memory_system import AgentMemory
from safety_logger import SafetyLogger

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic OCR Converter", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ─── Session State Init ─────────────────────────────────────────────────────
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = AgentMemory()
if 'conversion_count' not in st.session_state:
    st.session_state.conversion_count = 0

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem;
}
.sub-header { text-align: center; color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem; }
.agent-phase {
    background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
    border: 1px solid #e0e3ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
}
.phase-icon { font-size: 1.5rem; }
.stats-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
    border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stats-number {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stats-label { font-size: 0.8rem; color: #6b7280; font-weight: 500; text-transform: uppercase; }
.decision-box {
    background: #fffbeb; border: 1px solid #fbbf24; border-radius: 10px; padding: 0.8rem; margin: 0.3rem 0;
}
.action-box {
    background: #ecfdf5; border: 1px solid #34d399; border-radius: 10px; padding: 0.8rem; margin: 0.3rem 0;
}
.result-box {
    background: #fafbff; border: 1px solid #e0e3ff; border-radius: 12px;
    padding: 1.5rem; font-family: 'Courier New', monospace; font-size: 0.9rem;
    line-height: 1.6; max-height: 400px; overflow-y: auto;
}
.upload-section {
    background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
    border: 2px dashed #667eea; border-radius: 16px; padding: 2rem; text-align: center;
}
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9ff 0%, #ece6ff 100%); }
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem;
    font-weight: 600; box-shadow: 0 4px 12px rgba(102,126,234,0.3);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.4); }
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    box-shadow: 0 4px 12px rgba(5,150,105,0.3);
}
.privacy-box {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px;
    padding: 1rem; font-size: 0.85rem; color: #166534;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Agent Settings")
    autonomy = st.selectbox("Autonomy Level", ["🔄 Semi-Autonomous", "⚡ Full Autonomous"], index=0)
    confidence_threshold = st.slider("Confidence Threshold", 30, 90, 55, 5, help="Minimum OCR confidence to accept")
    max_retries = st.slider("Max Retries", 1, 5, 3, help="How many strategies to try if quality is low")
    processing_mode = st.selectbox("Output Mode", ["🎨 Formatted (Headings & Bold)", "📄 Plain Text Only"])

    st.markdown("---")
    st.markdown("### 📊 Session Memory")
    summary = st.session_state.agent_memory.get_full_summary()
    sess = summary['session']
    st.metric("Conversions", sess['total_conversions'])
    st.metric("Avg Confidence", f"{sess['avg_confidence']:.0f}%")
    st.metric("Success Rate", f"{sess['success_rate']:.0f}%")
    if sess['total_conversions'] > 0:
        st.caption(f"Strategies used: {', '.join(sess['strategies_used'])}")
        st.caption(f"Human overrides: {sess['human_overrides']}")

    st.markdown("---")
    st.markdown("### 🛡️ Ethics & Privacy")
    with st.expander("Data Handling Policy"):
        logger_tmp = SafetyLogger()
        info = logger_tmp.get_data_handling_info()
        for k, v in info.items():
            st.markdown(f"**{k.replace('_',' ').title()}:** {v}")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>FAST-NUCES • BSAI<br>Phase 2 — Agentic System</div>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🤖 Agentic OCR Converter</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An autonomous agent that perceives, decides, acts, and learns to convert your images</p>', unsafe_allow_html=True)

# Pipeline visualization
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="agent-phase"><span class="phase-icon">👁️</span><br><b>Perceive</b><br><small>Analyze image quality</small></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="agent-phase"><span class="phase-icon">🧠</span><br><b>Decide</b><br><small>Select best strategy</small></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="agent-phase"><span class="phase-icon">⚡</span><br><b>Act</b><br><small>Execute OCR pipeline</small></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="agent-phase"><span class="phase-icon">📝</span><br><b>Learn</b><br><small>Update memory</small></div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Upload ──────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader("Upload your images", type=["jpg","jpeg","png","bmp","tiff"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### 🖼️ Uploaded Images")
    cols = st.columns(min(len(uploaded_files), 3))
    images = []
    for idx, f in enumerate(uploaded_files):
        img = Image.open(f)
        images.append((f.name, img))
        with cols[idx % 3]:
            st.image(img, caption=f.name, use_container_width=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button("🤖 Run Agent Pipeline", use_container_width=True, type="primary")

    if process_btn:
        for file_name, image in images:
            st.markdown(f"### 📄 Processing: `{file_name}`")

            # Create agent for this conversion
            logger = SafetyLogger()
            agent = OCRAgent(
                memory=st.session_state.agent_memory,
                logger=logger,
                confidence_threshold=confidence_threshold,
                max_retries=max_retries,
                autonomy_level='full' if 'Full' in autonomy else 'semi',
            )

            use_formatted = "Formatted" in processing_mode
            progress = st.progress(0)
            status = st.empty()

            try:
                # ── PERCEIVE ──
                status.markdown("👁️ **PERCEIVE:** Analyzing image quality...")
                progress.progress(10)
                analysis = agent.perceive(image)
                time.sleep(0.3)

                # Show perception results
                with st.expander("👁️ Perception — Image Analysis", expanded=True):
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    with pc1:
                        st.markdown(f'<div class="stats-card"><div class="stats-number">{analysis.quality_score:.0f}</div><div class="stats-label">Quality Score</div></div>', unsafe_allow_html=True)
                    with pc2:
                        st.markdown(f'<div class="stats-card"><div class="stats-number">{analysis.blur_score:.0f}</div><div class="stats-label">Sharpness</div></div>', unsafe_allow_html=True)
                    with pc3:
                        st.markdown(f'<div class="stats-card"><div class="stats-number">{analysis.contrast_ratio:.0%}</div><div class="stats-label">Contrast</div></div>', unsafe_allow_html=True)
                    with pc4:
                        profile_emoji = {'clean':'✅','noisy':'🔊','blurry':'🌫️','low_contrast':'🌑','mixed':'🔀'}.get(analysis.quality_profile,'❓')
                        st.markdown(f'<div class="stats-card"><div class="stats-number">{profile_emoji}</div><div class="stats-label">{analysis.quality_profile.title()}</div></div>', unsafe_allow_html=True)

                # ── DECIDE ──
                status.markdown("🧠 **DECIDE:** Selecting optimal strategy...")
                progress.progress(25)
                strategy = agent.decide(analysis)
                time.sleep(0.3)

                with st.expander("🧠 Decision — Strategy Selection"):
                    st.markdown(f'<div class="decision-box">🎯 <b>Selected Strategy:</b> {strategy.name.title()}<br><small>{strategy.description}</small></div>', unsafe_allow_html=True)
                    decisions = logger.get_decision_trail()
                    if decisions:
                        for d in decisions:
                            st.markdown(f"- **{d['message']}**")
                            if d.get('reasoning'):
                                st.caption(f"  ↳ {d['reasoning']}")

                # Semi-auto: show override option
                human_override = False
                if 'Semi' in autonomy:
                    with st.expander("👤 Human-in-the-Loop — Override Strategy"):
                        st.info(f"Agent selected: **{strategy.name}**. You can override below.")
                        override_choice = st.selectbox(
                            "Override strategy:", ["Keep Agent's Choice"] + list(STRATEGIES.keys()),
                            key=f"override_{file_name}"
                        )
                        if override_choice != "Keep Agent's Choice":
                            agent.apply_human_override(override_choice)
                            strategy = agent.current_strategy
                            human_override = True
                            st.success(f"✅ Overridden to: **{strategy.name}**")

                # ── ACT ──
                status.markdown(f"⚡ **ACT:** Running OCR with '{strategy.name}' strategy...")
                progress.progress(50)

                result = agent.run_full_pipeline(image, file_name, use_formatted)
                progress.progress(85)

                with st.expander("⚡ Action — Pipeline Execution"):
                    st.markdown(f'<div class="action-box">✅ Pipeline complete: <b>{result["total_words"]}</b> words, <b>{result["avg_confidence"]:.1f}%</b> confidence</div>', unsafe_allow_html=True)
                    if result.get('agent_metadata', {}).get('retries', 0) > 0:
                        st.warning(f"Agent retried {result['agent_metadata']['retries']} time(s) to improve quality.")

                # ── RESULTS ──
                status.markdown("✅ **Done!** Document generated successfully.")
                progress.progress(100)

                st.markdown("#### 📊 Results")
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.markdown(f'<div class="stats-card"><div class="stats-number">{result["total_words"]}</div><div class="stats-label">Words</div></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f'<div class="stats-card"><div class="stats-number">{result["headings_count"]}</div><div class="stats-label">Headings</div></div>', unsafe_allow_html=True)
                with rc3:
                    st.markdown(f'<div class="stats-card"><div class="stats-number">{result["avg_confidence"]:.0f}%</div><div class="stats-label">Confidence</div></div>', unsafe_allow_html=True)
                with rc4:
                    retries = result.get('agent_metadata', {}).get('retries', 0)
                    st.markdown(f'<div class="stats-card"><div class="stats-number">{retries}</div><div class="stats-label">Retries</div></div>', unsafe_allow_html=True)

                with st.expander("📝 Extracted Text Preview", expanded=True):
                    st.markdown(f'<div class="result-box">{result["text"]}</div>', unsafe_allow_html=True)

                # Generate document
                if use_formatted and result.get('blocks'):
                    docx_buffer = generate_docx(result['blocks'], title=os.path.splitext(file_name)[0], agent_metadata=result.get('agent_metadata'))
                else:
                    docx_buffer = generate_plain_docx(result.get('text', ''), title=os.path.splitext(file_name)[0])

                docx_filename = f"{os.path.splitext(file_name)[0]}_agentic.docx"
                st.download_button(f"📥 Download {docx_filename}", data=docx_buffer, file_name=docx_filename, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

                # ── LEARN ──
                with st.expander("📝 Learning — What the Agent Learned"):
                    st.markdown(f"- Recorded outcome for `{file_name}`")
                    st.markdown(f"- Strategy `{result['strategy_used']}` → {result['avg_confidence']:.1f}% confidence")
                    if human_override:
                        st.markdown("- 👤 Human override was applied")
                    mem_summary = st.session_state.agent_memory.get_full_summary()
                    st.json(mem_summary)

                # ── AUDIT LOG ──
                with st.expander("🛡️ Audit Log — Full Decision Trail"):
                    formatted_log = logger.get_formatted_log()
                    st.markdown(formatted_log)

                    report = logger.get_explainability_report()
                    st.markdown(f"**Total decisions:** {report['total_decisions']} | **Safety flags:** {report['total_safety_flags']} | **Time:** {report['total_time_seconds']}s")

                st.session_state.conversion_count += 1
                st.markdown("---")

            except Exception as e:
                progress.progress(100)
                status.markdown(f"❌ **Error:** {str(e)}")
                st.error(f"Failed to process {file_name}: {str(e)}")
                # Fallback
                st.warning("Attempting fallback plain-text extraction...")
                try:
                    processed = preprocess_image(image)
                    text = extract_text_simple(processed)
                    if text:
                        docx_buffer = generate_plain_docx(text)
                        st.download_button("📥 Download (Fallback)", data=docx_buffer, file_name=f"{os.path.splitext(file_name)[0]}_fallback.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                except Exception as fe:
                    st.error(f"Fallback failed: {str(fe)}")

else:
    # Empty state
    st.markdown("""
    <div class="upload-section">
        <p style="font-size:3rem;margin-bottom:0.5rem;">🤖</p>
        <p style="font-size:1.2rem;font-weight:600;color:#374151;">Drop your images here to activate the agent</p>
        <p style="color:#9ca3af;font-size:0.9rem;">The agent will autonomously analyze, decide, process, and learn</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 How the Agent Works")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown('<div class="stats-card"><p style="font-size:2rem;">👁️</p><p style="font-weight:600;">Perceive</p><p style="color:#6b7280;font-size:0.85rem;">Analyzes blur, noise, contrast, and resolution to understand image quality</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="stats-card"><p style="font-size:2rem;">🧠</p><p style="font-weight:600;">Decide</p><p style="color:#6b7280;font-size:0.85rem;">Selects optimal preprocessing strategy using rules and memory</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="stats-card"><p style="font-size:2rem;">⚡</p><p style="font-weight:600;">Act</p><p style="color:#6b7280;font-size:0.85rem;">Executes OCR pipeline with quality gates and auto-retry</p></div>', unsafe_allow_html=True)
    with f4:
        st.markdown('<div class="stats-card"><p style="font-size:2rem;">📝</p><p style="font-weight:600;">Learn</p><p style="color:#6b7280;font-size:0.85rem;">Stores outcomes in memory to improve future conversions</p></div>', unsafe_allow_html=True)

    # Comparative analysis table
    st.markdown("### 📊 Phase 1 vs Phase 2 Comparison")
    st.markdown("""
| Feature | Phase 1 (Static) | Phase 2 (Agentic) |
|---|---|---|
| **Control** | User-driven | System-driven with human oversight |
| **Intelligence** | Static pipeline | Adaptive, quality-aware |
| **Behavior** | Reactive | Proactive (auto-retry, quality gates) |
| **Memory** | None | Session + persistent memory |
| **Explainability** | None | Full decision audit trail |
| **Ethics** | Not addressed | Privacy controls, data transparency |
    """)

    # Privacy notice
    st.markdown("### 🛡️ Privacy & Ethics")
    st.markdown("""
    <div class="privacy-box">
        🔒 <b>Your data is safe.</b> All images are processed in-memory only — nothing is stored on any server.
        Session data is cleared when you close this tab. No personal data is collected. No cookies or tracking.
        All processing uses open-source Tesseract OCR (offline). You retain full ownership of all uploaded content.
    </div>
    """, unsafe_allow_html=True)
