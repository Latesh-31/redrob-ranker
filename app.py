"""Streamlit web dashboard interface for Redrob Ranker."""

from __future__ import annotations

import io
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import streamlit as st
import pandas as pd

from rank import rank_candidates, load_features_table
from src.utils.constants import DATA_DIR, RETRIEVAL_TOP_K, FINAL_TOP_K
from src.utils.types import RankedCandidate

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text content from an uploaded .docx file."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            doc_xml = z.read("word/document.xml")
            root = ET.fromstring(doc_xml)
            
            paragraphs = []
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            
            return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX file: {e}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from an uploaded .pdf file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        paragraphs = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                paragraphs.append(extracted)
        return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file: {e}")

# Page configuration
st.set_page_config(
    page_title="Redrob Ranker Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Injections
st.markdown("""
<style>
    /* Styling for glassmorphic containers */
    .stApp {
        background-color: #0f1123;
        color: #e2e8f0;
    }
    
    .card {
        background: rgba(30, 34, 64, 0.5);
        border: 1px solid rgba(127, 0, 255, 0.2);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        background: rgba(18, 18, 36, 0.7);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 15px;
    }
    
    .badge {
        font-size: 0.85rem;
        background-color: #7f00ff;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .badge-secondary {
        font-size: 0.85rem;
        background-color: #2d3748;
        color: #cbd5e0;
        padding: 3px 8px;
        border-radius: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .badge-alert {
        font-size: 0.85rem;
        background-color: #e53e3e;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Custom stylized banner
st.markdown("""
<div style="background: linear-gradient(135deg, #1d1233 0%, #0a0b1c 100%); padding: 30px; border-radius: 15px; border: 1px solid rgba(127, 0, 255, 0.35); text-align: center; margin-bottom: 30px;">
    <h1 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-weight: 800; margin: 0; font-size: 2.8rem; letter-spacing: -0.5px;">
        <span style="background: linear-gradient(90deg, #ff007f, #7f00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Redrob</span> Ranker Sandbox
    </h1>
    <p style="color: #a0aec0; font-size: 1.1rem; margin-top: 8px; margin-bottom: 0;">
        Intelligent Candidate Discovery, Multi-Signal Scoring, and Advanced Anti-Cheat Reranking
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.image("https://redrob.io/wp-content/uploads/2023/07/redrob-logo.png", width=150)
st.sidebar.title("Configuration")

use_xgb = st.sidebar.toggle("Use XGBoost Reranker", value=False, help="Enable localized XGBoost ML reranking on candidate features.")
retrieval_k = st.sidebar.slider("Semantic Retrieval Shortlist (K)", 100, 1000, RETRIEVAL_TOP_K, step=50, help="Initial candidate pool retrieved via dense vector similarity.")
final_k = st.sidebar.slider("Final Ranked Pool Output (Top-K)", 10, 200, FINAL_TOP_K, step=10, help="Number of matched candidates to return in output dataset.")

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")

try:
    feat_df = load_features_table()
    total_candidates = len(feat_df)
    st.sidebar.metric("Preprocessed Candidates Pool", f"{total_candidates:,}")
except Exception:
    st.sidebar.error("Warning: Preprocessed features parquet not found. Run preprocessing first.")

# Create tabs
tab_rank, tab_analysis, tab_metrics = st.tabs([
    "🎯 Upload & Rank", 
    "📊 Candidate Analysis", 
    "📈 System Performance"
])

def render_score_progress(label: str, value: float, color: str):
    value_pct = max(0, min(100, int(value * 100)))
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e0; margin-bottom: 3px;">
            <span>{label}</span>
            <span>{value:.2f}</span>
        </div>
        <div style="background-color: #1a202c; border-radius: 4px; height: 8px; overflow: hidden; width: 100%; border: 1px solid rgba(255,255,255,0.05);">
            <div style="background: {color}; width: {value_pct}%; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TAB 1: Upload & Rank candidates
with tab_rank:
    st.subheader("Job Description Submission")
    col_input, col_info = st.columns([2, 1])
    
    with col_input:
        jd_input_method = st.radio("Choose Input Method:", ["Upload Job Description File", "Paste Job Description Text"], horizontal=True)
        
        uploaded_file = None
        jd_text = ""
        
        if jd_input_method == "Upload Job Description File":
            uploaded_file = st.file_uploader("Upload Job Description (.md, .docx, .pdf, .txt)", type=["md", "txt", "docx", "pdf"])
        else:
            jd_text = st.text_area("Paste Job Description Markdown here:", height=250, placeholder="# Senior ML Engineer\n\n## Requirements\n- PyTorch\n- Python...")
            
    with col_info:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); height: 100%;">
            <h4 style="margin-top:0; color:#fff;">Evaluation Pipeline Info</h4>
            <ul style="color:#cbd5e0; font-size:0.9rem; padding-left:20px;">
                <li>Parses JD titles, skills, and qualifications.</li>
                <li>Uses vector cosine comparison for top-K retrieval.</li>
                <li>Flags and filters bot/suspicious profile honeypots.</li>
                <li>Calculates 6-category weighted score fusion.</li>
                <li>Generates fully local explanations per candidate.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    run_pipeline = st.button("🚀 Match and Rank Candidates", type="primary", use_container_width=True)
    
    if run_pipeline:
        jd_path = None
        temp_file = None
        
        if jd_input_method == "Upload Job Description File":
            if uploaded_file is not None:
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                
                if file_name.endswith(".docx"):
                    try:
                        jd_content = extract_text_from_docx(file_bytes)
                    except Exception as e:
                        st.error(f"Failed to read DOCX file: {e}")
                        jd_content = None
                elif file_name.endswith(".pdf"):
                    try:
                        jd_content = extract_text_from_pdf(file_bytes)
                    except Exception as e:
                        st.error(f"Failed to read PDF file: {e}")
                        jd_content = None
                else:
                    jd_content = file_bytes.decode("utf-8", errors="ignore")
                    
                if jd_content:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8') as tmp:
                        tmp.write(jd_content)
                        jd_path = Path(tmp.name)
            else:
                st.warning("Please upload a job description file first.")
        else:
            if jd_text.strip():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8') as tmp:
                    tmp.write(jd_text)
                    jd_path = Path(tmp.name)
            else:
                st.warning("Please enter job description text first.")
                
        if jd_path is not None:
            st.info("🔄 Running ranker pipeline (retrieve $\\rightarrow$ score $\\rightarrow$ rerank $\\rightarrow$ explain)...")
            try:
                ranked_pool = rank_candidates(
                    jd_path=jd_path,
                    retrieval_k=retrieval_k,
                    final_k=final_k,
                    output_path=Path("output/submission.xlsx"),
                    use_xgb_reranker=use_xgb
                )
                
                st.session_state["ranked_results"] = ranked_pool
                st.success(f"🎉 Matching completed! Ranked Top-{len(ranked_pool)} candidates.")
                
            except Exception as e:
                st.error(f"Error during ranking pipeline: {e}")
                st.exception(e)
                
    if "ranked_results" in st.session_state:
        results: list[RankedCandidate] = st.session_state["ranked_results"]
        
        st.markdown("---")
        st.subheader("Match Summary & Leaderboard")
        
        # Stats summary metrics
        total_matched = len(results)
        avg_score = sum(r.score for r in results) / total_matched if total_matched else 0.0
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Candidates Ranked", f"{total_matched}")
        col_m2.metric("Average Match Score", f"{avg_score:.4f}")
        
        with open("output/submission.xlsx", "rb") as file:
            col_m3.download_button(
                label="📥 Download submission.xlsx",
                data=file,
                file_name="submission.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        # Display Leaderboard
        results_data = []
        for r in results:
            results_data.append({
                "Rank": r.rank,
                "Candidate ID": r.candidate_id,
                "Match Score": f"{r.score:.4f}",
                "Reasoning Summary": r.reasoning
            })
            
        leaderboard_df = pd.DataFrame(results_data)
        st.dataframe(leaderboard_df.set_index("Rank"), use_container_width=True)

# TAB 2: Detailed candidate analysis card
with tab_analysis:
    if "ranked_results" not in st.session_state:
        st.info("No candidates ranked yet. Please run the pipeline in 'Upload & Rank' tab.")
    else:
        results: list[RankedCandidate] = st.session_state["ranked_results"]
        
        st.subheader("Deep Candidate Profiling & Score Decomposition")
        
        # Candidate dropdown selection
        cand_dict = {f"Rank {r.rank} - {r.candidate_id}": r for r in results}
        selected_key = st.selectbox("Select Candidate to Inspect:", list(cand_dict.keys()))
        
        if selected_key:
            candidate_item: RankedCandidate = cand_dict[selected_key]
            b = candidate_item.breakdown
            
            # Load full features for display
            features_df = load_features_table()
            raw_cand = features_df[features_df["candidate_id"] == candidate_item.candidate_id].to_dict(orient="records")[0]
            
            # Design profile card grid
            col_card, col_breakdown = st.columns([1, 1])
            
            with col_card:
                st.markdown(f"""
                <div class="card">
                    <h3 style="color:#ffffff; margin-top:0;">Candidate {candidate_item.candidate_id}</h3>
                    <div style="font-size:0.95rem; margin-bottom: 12px; color: #cbd5e0;">
                        <span style="font-weight:bold; color:#a0aec0;">Current Title:</span> {raw_cand.get('current_title', 'N/A')}<br/>
                        <span style="font-weight:bold; color:#a0aec0;">Max Education:</span> Degree level {raw_cand.get('max_education_level', '0')} ({raw_cand.get('education_field', 'N/A')})<br/>
                        <span style="font-weight:bold; color:#a0aec0;">Years Experience:</span> {raw_cand.get('relevant_years', '0')} relevant / {raw_cand.get('total_years', '0')} total years<br/>
                        <span style="font-weight:bold; color:#a0aec0;">Location/Relocation:</span> {raw_cand.get('country', 'N/A')} (Relocate signal: {raw_cand.get('signal_willing_to_relocate', 'N/A')})<br/>
                        <span style="font-weight:bold; color:#a0aec0;">Platform Activity:</span> Inactive days: {raw_cand.get('signal_last_active_date', 'N/A')}
                    </div>
                    <div style="margin-bottom:15px;">
                        <span style="font-weight:bold; color:#a0aec0; display:block; margin-bottom:5px;">Skills Registry:</span>
                """, unsafe_allow_html=True)
                
                # Render skill badges
                skills = raw_cand.get("skills", [])
                if isinstance(skills, str):
                    skills = [s.strip() for s in skills.split(",") if s.strip()]
                for s in skills[:15]:
                    st.markdown(f'<span class="badge-secondary">{s}</span>', unsafe_allow_html=True)
                if len(skills) > 15:
                    st.markdown(f'<span class="badge-secondary">+{len(skills)-15} more</span>', unsafe_allow_html=True)
                    
                st.markdown("""
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Title History
                st.markdown("<div class='card'><h4 style='color:#fff; margin-top:0;'>Title & Career History</h4>", unsafe_allow_html=True)
                titles = raw_cand.get("title_history", [])
                if isinstance(titles, str):
                    titles = [t.strip() for t in titles.split("|") if t.strip()]
                for t in titles:
                    st.write(f"• {t}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_breakdown:
                st.markdown(f"""
                <div class="card">
                    <h3 style="color:#ffffff; margin-top:0; display:flex; justify-content:space-between; align-items:center;">
                        <span>Composite Score</span>
                        <span style="font-size:1.6rem; color:#7f00ff; font-weight:bold;">{b.final_score:.4f}</span>
                    </h3>
                    <p style="color:#cbd5e0; font-size:0.95rem; line-height:1.5; margin-bottom:20px; font-style:italic;">
                        "{candidate_item.reasoning}"
                    </p>
                """, unsafe_allow_html=True)
                
                render_score_progress("Semantic Score (20% wt)", b.semantic, "#4299e1")
                render_score_progress("Skills Overlap (30% wt)", b.skill, "#48bb78")
                render_score_progress("Experience Score (25% wt)", b.experience, "#ecc94b")
                render_score_progress("Title Relevance (15% wt)", b.title, "#ed64a6")
                render_score_progress("Education Fit (5% wt)", b.education, "#9f7aea")
                render_score_progress("Platform Behavior (5% wt)", b.behavior, "#38b2ac")
                render_score_progress("Consistency Multiplier", b.consistency, "#ed8936")
                
                st.markdown("""
                </div>
                """, unsafe_allow_html=True)
                
                # Render matching details & Honeypots details
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#fff; margin-top:0;'>Anti-Cheat Checks & Skill Matches</h4>", unsafe_allow_html=True)
                
                st.write("**Matched Skills:**")
                for s in b.matched_skills:
                    st.markdown(f'<span class="badge">{s}</span>', unsafe_allow_html=True)
                if not b.matched_skills:
                    st.write("None")
                    
                st.write("**Suspicious Honeypot Flags:**")
                for flag in b.honeypot_flags:
                    st.markdown(f'<span class="badge-alert">{flag}</span>', unsafe_allow_html=True)
                if not b.honeypot_flags:
                    st.write("Clean (No warnings triggered)")
                else:
                    st.warning(f"Total Honeypot Penalty Applied: {b.honeypot_penalty:.4f}")
                    
                st.markdown("</div>", unsafe_allow_html=True)

# TAB 3: System Analytics and charts
with tab_metrics:
    if "ranked_results" not in st.session_state:
        st.info("No candidates ranked yet. Please run the pipeline in 'Upload & Rank' tab.")
    else:
        results: list[RankedCandidate] = st.session_state["ranked_results"]
        
        st.subheader("Leaderboard Distribution & Scores Analytics")
        
        # Prepare charts dataframe
        chart_data = pd.DataFrame({
            "Rank": [r.rank for r in results],
            "Score": [r.score for r in results],
            "Semantic Score": [r.breakdown.semantic for r in results],
            "Skill Score": [r.breakdown.skill for r in results],
            "Experience Score": [r.breakdown.experience for r in results],
            "Consistency Multiplier": [r.breakdown.consistency for r in results]
        }).set_index("Rank")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("**Score Decay Distribution**")
            st.line_chart(chart_data[["Score"]], color="#7f00ff")
            st.caption("Visualizes the score slope across the Top-K candidates.")
            
        with col_c2:
            st.write("**Components Breakdown Trends**")
            st.line_chart(chart_data[["Semantic Score", "Skill Score", "Experience Score", "Consistency Multiplier"]])
            st.caption("Comparison showing how separate metric groups slide together across ranks.")
            
        # Distribution stats
        st.markdown("---")
        st.subheader("Statistical Ranges")
        
        df_stats = chart_data.describe().transpose()[["mean", "std", "min", "max"]]
        st.table(df_stats)
