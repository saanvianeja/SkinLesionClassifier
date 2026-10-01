"""Streamlit presentation helpers. Does not affect inference or Grad-CAM."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "DM Sans", sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(59, 130, 246, 0.18), transparent 55%),
    radial-gradient(900px 400px at 100% 0%, rgba(14, 165, 233, 0.10), transparent 50%),
    #0b1220;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none; }

.block-container {
  max-width: 1080px;
  padding-top: 1.1rem;
  padding-bottom: 2.5rem;
}

.hero {
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 42%, #0ea5e9 100%);
  border-radius: 18px;
  padding: 1.55rem 1.7rem 1.35rem;
  color: #fff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
  margin-bottom: 1.15rem;
}
.hero h1 {
  font-family: "DM Sans", sans-serif;
  font-size: 2.05rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 0.35rem 0;
}
.hero p {
  margin: 0;
  opacity: 0.94;
  font-size: 1.02rem;
}
.badge-row { margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.badge {
  display: inline-block;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.22);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 0.22rem 0.55rem;
  border-radius: 999px;
}
.badge.muted { background: rgba(15, 23, 42, 0.22); }

.card {
  background: rgba(18, 26, 43, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 1.15rem 1.2rem 1.05rem;
  box-shadow: 0 10px 28px rgba(2, 6, 23, 0.28);
  margin-bottom: 0.85rem;
}
.card h3 {
  font-family: "DM Sans", sans-serif;
  margin: 0 0 0.55rem 0;
  font-size: 1.12rem;
  color: #f1f5f9;
}
.card p, .card li { color: #cbd5e1; font-size: 0.95rem; line-height: 1.45; }
.howto { display: grid; gap: 0.7rem; }
.howto-item { display: grid; grid-template-columns: 2rem 1fr; gap: 0.65rem; align-items: start; }
.step {
  width: 1.7rem; height: 1.7rem; border-radius: 999px;
  background: #1d4ed8; color: #fff; font-weight: 700; font-size: 0.8rem;
  display: flex; align-items: center; justify-content: center;
}
.howto strong { color: #e2e8f0; }

.section-title {
  font-family: "DM Sans", sans-serif;
  text-align: center;
  font-size: 1.45rem;
  font-weight: 700;
  color: #f8fafc;
  margin: 0.4rem 0 1rem 0;
}

.pred-card {
  text-align: center;
  padding: 1.3rem 1rem 1.15rem;
}
.pred-kicker {
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.72rem;
  color: #93c5fd;
  font-weight: 600;
  margin-bottom: 0.2rem;
}
.pred-value {
  font-family: "DM Sans", sans-serif;
  font-size: 2.15rem;
  font-weight: 700;
  margin: 0.1rem 0 0.55rem 0;
}
.pred-value.benign { color: #67e8f9; }
.pred-value.malignant { color: #fbbf24; }
.pred-sub { color: #94a3b8; font-size: 0.92rem; margin-bottom: 0.85rem; }

.stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; }
@media (max-width: 720px) {
  .stat-row { grid-template-columns: 1fr; }
  .hero h1 { font-size: 1.6rem; }
}
.stat {
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 12px;
  padding: 0.7rem 0.55rem;
}
.stat .label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stat .value { color: #f8fafc; font-size: 1.28rem; font-weight: 700; margin-top: 0.15rem; }

.prob-row {
  display: grid;
  grid-template-columns: 6.2rem 1fr 3.6rem;
  gap: 0.65rem;
  align-items: center;
  margin: 0.45rem 0;
}
.prob-label { color: #cbd5e1; font-size: 0.9rem; }
.prob-track {
  height: 0.72rem;
  background: rgba(51, 65, 85, 0.7);
  border-radius: 999px;
  overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 999px; }
.prob-fill.benign { background: linear-gradient(90deg, #22d3ee, #38bdf8); }
.prob-fill.malignant { background: linear-gradient(90deg, #f59e0b, #f97316); }
.prob-pct { text-align: right; color: #e2e8f0; font-variant-numeric: tabular-nums; font-weight: 600; }

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.7rem;
}
@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
.metric {
  background: rgba(18, 26, 43, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 0.9rem 0.8rem;
  text-align: center;
}
.metric .m-val { font-size: 1.45rem; font-weight: 700; color: #f8fafc; }
.metric .m-lab { color: #94a3b8; font-size: 0.78rem; margin-top: 0.2rem; letter-spacing: 0.04em; text-transform: uppercase; }

.caption-note { color: #94a3b8; font-size: 0.86rem; margin-top: 0.35rem; }
.disclaimer {
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.7);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  color: #cbd5e1;
  font-size: 0.9rem;
  line-height: 1.45;
}

div[data-testid="stFileUploader"] section {
  background: rgba(15, 23, 42, 0.45);
  border: 1px dashed rgba(96, 165, 250, 0.45);
  border-radius: 12px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Skin Lesion Classifier</h1>
          <p>Fine-tuned MobileNetV2 for dermoscopic skin lesion classification</p>
          <div class="badge-row">
            <span class="badge muted">Educational ML Demo</span>
            <span class="badge">PyTorch</span>
            <span class="badge">MobileNetV2</span>
            <span class="badge">HAM10000</span>
            <span class="badge">Grad-CAM</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def how_it_works_card() -> None:
    st.markdown(
        """
        <div class="card">
          <h3>How It Works</h3>
          <div class="howto">
            <div class="howto-item"><div class="step">1</div><div><strong>Upload</strong><br/>Upload a dermoscopic lesion image.</div></div>
            <div class="howto-item"><div class="step">2</div><div><strong>Classify</strong><br/>A fine-tuned MobileNetV2 estimates the probability of the malignant class.</div></div>
            <div class="howto-item"><div class="step">3</div><div><strong>Explain</strong><br/>Grad-CAM highlights image regions that influenced the network.</div></div>
            <div class="howto-item"><div class="step">4</div><div><strong>Review</strong><br/>View the prediction alongside model performance and limitations.</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(p_benign: float, p_malignant: float) -> None:
    st.markdown(
        f"""
        <div class="card">
          <h3>Class probabilities</h3>
          <div class="prob-row">
            <div class="prob-label">Benign</div>
            <div class="prob-track"><div class="prob-fill benign" style="width:{p_benign * 100:.2f}%"></div></div>
            <div class="prob-pct">{p_benign * 100:.1f}%</div>
          </div>
          <div class="prob-row">
            <div class="prob-label">Malignant</div>
            <div class="prob-track"><div class="prob-fill malignant" style="width:{p_malignant * 100:.2f}%"></div></div>
            <div class="prob-pct">{p_malignant * 100:.1f}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_card(label: str, p_benign: float, p_malignant: float, threshold: float) -> None:
    klass = "malignant" if label.lower() == "malignant" else "benign"
    st.markdown(
        f"""
        <div class="card pred-card">
          <div class="pred-kicker">Prediction</div>
          <div class="pred-value {klass}">{label}</div>
          <div class="pred-sub">Displayed class uses P(malignant) ≥ {threshold:.2f}, not argmax at 0.50.</div>
          <div class="stat-row">
            <div class="stat"><div class="label">Malignant probability</div><div class="value">{p_malignant * 100:.1f}%</div></div>
            <div class="stat"><div class="label">Benign probability</div><div class="value">{p_benign * 100:.1f}%</div></div>
            <div class="stat"><div class="label">Decision threshold</div><div class="value">29%</div></div>
          </div>
          <p class="caption-note">The classification threshold (0.29) was selected using the validation set before final test evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def performance_cards() -> None:
    st.markdown(
        """
        <div class="metric-grid">
          <div class="metric"><div class="m-val">0.896</div><div class="m-lab">ROC-AUC</div></div>
          <div class="metric"><div class="m-val">82.6%</div><div class="m-lab">Sensitivity</div></div>
          <div class="metric"><div class="m-val">76.0%</div><div class="m-lab">Specificity</div></div>
          <div class="metric"><div class="m-val">79.3%</div><div class="m-lab">Balanced Acc.</div></div>
        </div>
        <p class="caption-note" style="margin:0.75rem 0 0.4rem;">
          Held-out test set (not clinical performance). Malignant F1 = 0.582.
          Final evaluation: 1,505 held-out test images (288 malignant, 1,217 benign).
        </p>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
          Educational use only. This application is a machine-learning portfolio project and is not a medical device.
          Its predictions should not be used for diagnosis, treatment, or clinical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )
