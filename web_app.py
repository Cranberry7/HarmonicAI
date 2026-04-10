import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
import sys
import time

st.set_page_config(page_title="HarmonicAI", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #121212 0%, #1e1e2f 100%);
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0f172a;
}
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #ec4899);
    border: none;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.feedback.drift_detector import INTENT_ENCODING
from src.genai.therapy_engine import TherapyEngine, SessionContext
from src.security.security_layer import SecurityLayer, InboundRequest

@st.cache_resource
def load_models():
    tracks_file = ROOT / "data" / "processed" / "tracks_clustered.csv"
    rf_file = ROOT / "models" / "checkpoints" / "random_forest.pkl"
    feat_file = ROOT / "models" / "checkpoints" / "feature_list.pkl"
    safe_file = ROOT / "models" / "checkpoints" / "safety_filter.pkl"
    
    tracks_df = pd.read_csv(tracks_file)
    with open(rf_file, "rb") as f:
        rf_pipeline = pickle.load(f)
    with open(feat_file, "rb") as f:
        feature_list = pickle.load(f)
    with open(safe_file, "rb") as f:
        filter_dict = pickle.load(f)
        
    return tracks_df, rf_pipeline, feature_list, filter_dict

try:
    tracks_df, rf_pipeline, feature_list, filter_dict = load_models()
except Exception as e:
    st.error(f"Error loading models. Please run `setup_pipeline.py` first. Exception: {e}")
    st.stop()

tfidf_pipeline = filter_dict["tfidf_pipeline"]
HARD_BLOCKLIST = filter_dict["hard_blocklist"]
RISK_KEYWORDS = filter_dict["risk_keywords"]
THRESHOLD = filter_dict["threshold"]

def layer1_check(lyric: str) -> tuple[bool, str]:
    lower = lyric.lower()
    for phrase in HARD_BLOCKLIST:
        if phrase in lower:
            return False, phrase
    return True, ""
    
def layer3_check(lyric: str) -> tuple[str, float]:
    lower  = lyric.lower().split()
    scores = {}
    for category, keywords in RISK_KEYWORDS.items():
        hits = sum(1 for w in lower if w in keywords)
        scores[category] = hits / max(len(lower), 1)
    best = max(scores, key=scores.get)
    return best, scores[best]
    
def safety_filter(lyric: str, instrumentalness: float = 0.0, sensitivity_flags: list[str] = []) -> dict:
    if instrumentalness > 0.6: return {"safe": True, "risk_category": "instrumental_skip", "risk_score": 0.0, "block_reason": "", "flagged_terms": []}
    l1_safe, matched = layer1_check(lyric)
    if not l1_safe: return {"safe": False, "risk_category": "self_harm", "risk_score": 1.0, "block_reason": "hard_blocklist", "flagged_terms": [matched]}
    risk_score = float(tfidf_pipeline.predict_proba([lyric])[0, 1])
    if risk_score >= THRESHOLD: return {"safe": False, "risk_category": "self_harm", "risk_score": risk_score, "block_reason": "tfidf_classifier", "flagged_terms": []}
    sem_cat, sem_score = layer3_check(lyric)
    if sem_cat == "grief_trigger" and "grief_sensitive" in sensitivity_flags: return {"safe": False, "risk_category": "grief_trigger", "risk_score": sem_score, "block_reason": "semantic_sensitivity_routing", "flagged_terms": []}
    return {"safe": True, "risk_category": sem_cat if sem_score > 0.1 else "safe", "risk_score": risk_score, "block_reason": "", "flagged_terms": []}

@st.cache_resource
def get_engines():
    therapy = TherapyEngine(backend="mock")
    sec = SecurityLayer(
        rate_limit_requests=30, rate_limit_window_s=60, burst_capacity=5,
        audit_log_path=ROOT / "data" / "security" / "audit.log", pii_salt="harmonicai_demo_salt"
    )
    return therapy, sec

therapy_engine, security = get_engines()

st.title("🎵 HarmonicAI")
st.subheader("Therapeutic Audio Recommendation Engine")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        mood_pre = st.slider("Select your current mood (0-100)", min_value=0.0, max_value=100.0, value=31.0, step=1.0)
    with col2:
        intent = st.selectbox("What is your intent?", ["mood_uplift", "sleep_induction", "anxiety_relief", "grief_processing", "deep_focus"])

if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Analyzing acoustic parameters and validating against safety matrix..."):
        USER_ID = "user_0042"
        SESSION_ID = f"sess_st_{int(time.time())}"
        SENSITIVITY_FLAGS = ["grief_sensitive"]
        
        sec_req = InboundRequest(
            user_id=USER_ID, session_id=SESSION_ID, intent=intent, mood_pre=mood_pre,
            sensitivity_flags=SENSITIVITY_FLAGS, feature_vector={
                "tempo_bpm": 80.0, "energy": 0.3, "valence": 0.4, "acousticness": 0.7, 
                "instrumentalness": 0.5, "speechiness": 0.04, "loudness_db": -18.0,
                "mood_pre": mood_pre, "intent_encoded": float(INTENT_ENCODING[intent])
            }
        )
        gate = security.inspect_request(sec_req)
        if not gate.allowed:
            st.error(f"Request Blocked: {gate.block_reason}")
            st.stop()
            
        INTENT_CLUSTER_AFFINITY = {"anxiety_relief": [0, 3, 5], "sleep_induction": [0, 3, 5], "grief_processing": [3, 5, 0], "mood_uplift": [1, 2, 0], "deep_focus": [3, 4, 0]}
        clusters = INTENT_CLUSTER_AFFINITY.get(intent, list(range(6)))
        cand = tracks_df[tracks_df["cluster_id"].isin(clusters)].copy()
        if len(cand) > 200: cand = cand.sample(200, random_state=42)
        
        safe_tracks = []
        for _, row in cand.iterrows():
            instr = float(row["instrumentalness"])
            result = safety_filter("soft piano drifting" if instr > 0.4 else "gentle streams of comfort", instrumentalness=instr, sensitivity_flags=SENSITIVITY_FLAGS)
            if result["safe"]: safe_tracks.append(row)
        safe_df = pd.DataFrame(safe_tracks)
        
        safe_df["mood_pre"] = mood_pre
        safe_df["intent_encoded"] = float(INTENT_ENCODING[intent])
        X_score = safe_df[feature_list].fillna(0).values
        probs = rf_pipeline.predict_proba(X_score)
        
        classes = list(rf_pipeline.named_steps["rf"].classes_)
        pos_idx = classes.index(1) if 1 in classes else 0
        safe_df["p_therapeutic"] = probs[:, pos_idx]
        top5 = safe_df.nlargest(5, "p_therapeutic").reset_index(drop=True)
        
        ctx = SessionContext(
            user_id=gate.pseudonym, session_id=SESSION_ID, intent=intent, mood_pre=mood_pre,
            top_tracks=top5[["track_id","tempo_bpm","energy","valence","acousticness","instrumentalness","cluster_name"]],
            sensitivity_flags=SENSITIVITY_FLAGS
        )
        script_res = therapy_engine.generate(ctx)
        
        st.success("✅ Analytics complete. Secure recommendations generated.")
        
        st.markdown(f"### 🌿 Personalized Therapy Script\n> *{script_res.script}*")
        
        st.markdown("### 🎧 Top Track Recommendations")
        st.dataframe(top5[["track_id", "p_therapeutic", "tempo_bpm", "energy", "valence", "cluster_name"]], hide_index=True)
