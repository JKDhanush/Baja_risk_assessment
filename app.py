import streamlit as st
import json
import re
import os
import base64

from retrieval.query_faiss import retrieve_context
from llm.groq_client import call_llm
from ml.run_models import run_model
from ml.model_registry import MODEL_REGISTRY
from reports.report_generator import generate_pdf


# ---------------- API KEY CHECK ---------------- #
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()


# ---------------- BACKGROUND HELPER ---------------- #
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------- SAFE JSON PARSER ---------------- #
def safe_json_loads(text):
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        return json.loads(match.group())
    except Exception:
        return None


# ---------------- REMOVE <think> BLOCKS ---------------- #
def remove_think_blocks(text):
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(
    page_title="AI Risk Assessment",
    page_icon="⚠️",
    layout="centered"
)


# ---------------- BACKGROUND IMAGE ---------------- #
bg_path = "assets/bg.jpg"

if os.path.exists(bg_path):
    bg_img = get_base64(bg_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .black-bg {{
            background: rgba(0, 0, 0, 0.65);
            padding: 14px 18px;
            border-radius: 10px;
            color: white;
            margin-bottom: 14px;
        }}

        label {{
            background: rgba(0, 0, 0, 0.65);
            padding: 6px 10px;
            border-radius: 6px;
            color: white !important;
            display: inline-block;
            margin-bottom: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------- TITLE ---------------- #
st.markdown(
    """
    <div class="black-bg">
        <h1>AI-Assisted Risk Assessment & Mitigation Strategy</h1>
        <p>
        Decision-support system for structured business and operational risk assessment.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- USER INPUT ---------------- #
query = st.text_area(
    "Describe the risk scenario:",
    height=150,
    placeholder="Example: Risk of procurement delay due to steel price volatility and supplier dependency."
)


# ---------------- MAIN PIPELINE ---------------- #
if st.button("Generate Risk Assessment Report"):

    if query.strip() == "":
        st.warning("Please enter a risk scenario.")
        st.stop()

    with st.spinner("Generating professional risk assessment report..."):

        # -------- RAG CONTEXT -------- #
        context = retrieve_context(query)

        # -------- LLM: MODEL PLANNER -------- #
        planner_prompt = f"""
Available ML models:
{MODEL_REGISTRY}

Select required models and generate numeric inputs.
Return ONLY valid JSON.

Risk Scenario:
{query}

Format:
{{"models":[{{"name":"","inputs":[]}}]}}
"""

        planner_response = call_llm(planner_prompt)
        plan = safe_json_loads(planner_response)

        # -------- FALLBACK -------- #
        if plan is None or "models" not in plan:
            plan = {
                "models": [
                    {"name": "risk_classifier", "inputs": [0.7, 18, 1, 0.8]},
                    {"name": "delay_predictor", "inputs": [18, 0.6, 5]},
                    {"name": "cost_overrun_predictor", "inputs": [0.7, 1, 0.15]}
                ]
            }

        # -------- RUN ML MODELS -------- #
        ml_results = {}
        for m in plan["models"]:
            ml_results[m["name"]] = run_model(m["name"], m["inputs"])

        # -------- LLM: REPORT WRITER -------- #
        writer_prompt = f"""
Generate a PROFESSIONAL BUSINESS RISK ASSESSMENT REPORT.

Rules:
- No AI reasoning
- No markdown
- No explanations
- Do not mention models or AI
- Do not repeat numeric values in the narrative
- Use formal business language only

Use sections:
1. Executive Summary
2. Risk Scenario Description
3. Key Risks Identified
4. Risk Assessment Summary
5. Mitigation Strategy
6. Implementation Considerations
7. Conclusion

Risk Scenario:
{query}

Quantitative Indicators:
{ml_results}

Context:
{context}
"""

        report_text = remove_think_blocks(call_llm(writer_prompt))

        # -------- PDF GENERATION (IN-MEMORY) -------- #
        pdf_bytes = generate_pdf(report_text, ml_results)

    # -------- DOWNLOAD BUTTON -------- #
    st.download_button(
        label="📄 Download Risk Assessment Report (PDF)",
        data=pdf_bytes,
        file_name="AI_Risk_Assessment_Report.pdf",
        mime="application/pdf"
    )
