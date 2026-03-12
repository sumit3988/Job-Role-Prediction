"""
app.py
======
Streamlit web application for Job Role Prediction.
Users input their educational background and the app predicts
the most likely job role using the trained model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Job Role Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Global font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero h1 { color: #e2e8f0; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero p  { color: #94a3b8; font-size: 1.05rem; margin-top: 0.5rem; }

    /* Cards */
    .card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .card h3 { color: #63b3ed; margin-top: 0; font-size: 1.05rem; letter-spacing: .03em; }

    /* Prediction box */
    .pred-box {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 1px solid #34d399;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(52,211,153,0.2);
    }
    .pred-box h2 { color: #d1fae5; font-size: 1.5rem; margin: 0 0 .5rem; }
    .pred-box p  { color: #6ee7b7; font-size: 2.2rem; font-weight: 700; margin: 0; }

    /* Metric chips */
    .metric-chip {
        display: inline-block;
        background: rgba(99,179,237,0.12);
        border: 1px solid rgba(99,179,237,0.3);
        color: #90cdf4;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    /* Sidebar style */
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load model & encoders
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model          = joblib.load("model.pkl")
    cat_encoders   = joblib.load("artifacts/cat_encoders.pkl")
    target_encoder = joblib.load("artifacts/target_encoder.pkl")
    feature_names  = joblib.load("artifacts/feature_names.pkl")
    return model, cat_encoders, target_encoder, feature_names

model_loaded = os.path.exists("model.pkl") and os.path.exists("artifacts/cat_encoders.pkl")

# ---------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Job Role Predictor</h1>
    <p>Discover the best-fit job role based on your educational background and skills</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Model not found warning
# ---------------------------------------------------------------
if not model_loaded:
    st.error(
        "⚠️ Model artifacts not found. Please run the training pipeline first:\n\n"
        "```\npython generate_dataset.py\npython train_model.py\n```"
    )
    st.stop()

model, cat_encoders, target_encoder, feature_names = load_artifacts()

# ---------------------------------------------------------------
# Sidebar – About
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This application predicts the most suitable **job role** based on your
    educational qualifications, skills, and experience.

    **Supported Job Roles:**
    """)
    roles = list(target_encoder.classes_)
    for r in roles:
        st.markdown(f"• {r}")

    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown(f"• Algorithm: `{type(model).__name__}`")
    st.markdown(f"• Features: `{len(feature_names)}`")

    st.markdown("---")
    if st.checkbox("Show sample input"):
        st.json({
            "Degree": "B.Tech",
            "Major": "Computer Science",
            "Skills": ["Python", "Machine Learning", "SQL"],
            "Certifications": "AWS",
            "GPA": 8.5,
            "Internship Experience": 2,
            "Number of Projects": 5,
        })

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
ALL_SKILLS = [
    "Python", "Java", "SQL", "AWS", "Machine Learning", "JavaScript",
    "React", "Node.js", "Docker", "Kubernetes", "TensorFlow", "Deep Learning",
    "C++", "PHP", "MongoDB", "Azure", "Linux", "Git"
]

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="card"><h3>📋 Educational Background</h3>', unsafe_allow_html=True)

    degree = st.selectbox("Degree", ["B.Tech", "BCA", "MCA", "B.Sc", "M.Tech", "MBA", "B.E"],
                          help="Your highest or current degree")
    major = st.selectbox("Major / Specialization",
                         ["Computer Science", "Information Technology", "Data Science", "Electronics"])
    gpa = st.slider("GPA (out of 10)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🏢 Experience</h3>', unsafe_allow_html=True)
    internships = st.selectbox("Internship Experience (months / count)", [0, 1, 2, 3],
                               format_func=lambda x: f"{x} internship(s)")
    projects    = st.number_input("Number of Projects", min_value=0, max_value=20, value=3, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><h3>💻 Skills & Certifications</h3>', unsafe_allow_html=True)
    selected_skills = st.multiselect(
        "Select your Skills",
        options=ALL_SKILLS,
        default=["Python", "SQL"],
        help="Pick all skills you are proficient in"
    )
    certification = st.selectbox("Certifications",
                                 ["None", "AWS", "Azure", "Google Cloud", "Cisco", "PMP"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Skill coverage indicator
    if selected_skills:
        st.markdown('<div class="card"><h3>🔖 Selected Skills</h3>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="metric-chip">{s}</span>' for s in selected_skills)
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def build_input_vector(degree, major, skills, certification, gpa, internships, projects):
    """Transform user inputs into the feature vector expected by the model."""
    row = {
        "Degree":               degree,
        "Major":                major,
        "Certifications":       certification,
        "GPA":                  gpa,
        "Internship_Experience": internships,
        "Number_of_Projects":   projects,
    }

    # Encode categoricals
    for col in ["Degree", "Major", "Certifications"]:
        le = cat_encoders[col]
        val = row[col] if row[col] in le.classes_ else le.classes_[0]
        row[col] = int(le.transform([val])[0])

    # Multi-hot skill columns
    for skill in ALL_SKILLS:
        col_name = f"skill_{skill.replace(' ', '_').replace('.', '')}"
        row[col_name] = 1 if skill in skills else 0

    # Build DataFrame with correct column order
    input_df = pd.DataFrame([row])[feature_names]
    return input_df


st.markdown("---")
predict_btn = st.button("🔮 Predict My Job Role", use_container_width=True, type="primary")

if predict_btn:
    if not selected_skills:
        st.warning("Please select at least one skill for a better prediction.")
    else:
        input_vec = build_input_vector(
            degree, major, selected_skills, certification,
            gpa, internships, projects
        )

        prediction_encoded = model.predict(input_vec)[0]
        predicted_role     = target_encoder.inverse_transform([prediction_encoded])[0]

        # Probability distribution
        if hasattr(model, "predict_proba"):
            proba    = model.predict_proba(input_vec)[0]
            classes  = target_encoder.inverse_transform(np.arange(len(proba)))
            proba_df = pd.DataFrame({"Job Role": classes, "Confidence": proba * 100}).sort_values(
                "Confidence", ascending=False
            )
        else:
            proba_df = None

        # Display prediction
        st.markdown(f"""
        <div class="pred-box">
            <h2>🎯 Predicted Job Role</h2>
            <p>{predicted_role}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show confidence chart
        if proba_df is not None:
            st.markdown("### 📊 Confidence Distribution")
            fig, ax = plt.subplots(figsize=(9, 4))
            colors = ["#34d399" if r == predicted_role else "#4a5568" for r in proba_df["Job Role"]]
            bars = ax.barh(proba_df["Job Role"], proba_df["Confidence"],
                           color=colors, edgecolor="none", height=0.55)
            for bar, val in zip(bars, proba_df["Confidence"]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=9,
                        color="#e2e8f0")
            ax.set_xlabel("Confidence (%)", color="#94a3b8")
            ax.set_xlim(0, 110)
            ax.tick_params(colors="#94a3b8")
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0f172a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            st.pyplot(fig)
            plt.close()

# ---------------------------------------------------------------
# Visualization gallery in expander
# ---------------------------------------------------------------
st.markdown("---")
with st.expander("📈 View Training Visualizations", expanded=False):
    plot_files = {
        "Job Role Distribution":   "plots/job_role_distribution.png",
        "Feature Importance":      "plots/feature_importance.png",
        "Model Comparison":        "plots/model_comparison.png",
        "Confusion Matrix":        "plots/confusion_matrix.png",
    }
    tab_labels = list(plot_files.keys())
    tabs = st.tabs(tab_labels)
    for tab, (label, path) in zip(tabs, plot_files.items()):
        with tab:
            if os.path.exists(path):
                st.image(path, caption=label, use_column_width=True)
            else:
                st.info(f"Run `python train_model.py` to generate the '{label}' plot.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.85rem;'>"
    "Job Role Predictor · Built with Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)
