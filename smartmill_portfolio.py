"""
SmartMill Uganda — Interactive Portfolio App
=============================================
Final Year Project | Computer Engineering & Informatics
Busitema University, Uganda

Run:  streamlit run smartmill_portfolio.py
Deploy free: https://streamlit.io/cloud  (connect GitHub repo, one click)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as spkurt, skew
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartMill Uganda | Portfolio",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* dark engineering terminal palette */
  html, body, [class*="css"] { font-family: 'JetBrains Mono', 'Courier New', monospace; }
  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
  h1 { color: #f59e0b; letter-spacing: .06em; }
  h2 { color: #60a5fa; }
  h3 { color: #4ade80; }
  .stMetric label { font-size: 0.72rem !important; color: #6b7280 !important; text-transform: uppercase; letter-spacing: .1em; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700; }
  div[data-testid="stSidebarNav"] { background: #0f1221; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS (match Phase 1 report exactly) ───────────────────────────────────
SR   = 1000        # Hz  sampling rate
WIN  = 512         # samples per window
SHAFT_HZ = 25.0   # Hz  ≈ 1450 RPM / 60
DEFECT_HZ = 110.0 # Hz  BPFI for 6205 bearing at 1450 RPM
ISO_A, ISO_B, ISO_C = 4.0, 7.0, 11.0   # mm/s zone thresholds (ISO 10816)

FAULT_LABELS = {
    "normal":    "Normal Operation",
    "imbalance": "Rotor Imbalance",
    "misalign":  "Shaft Misalignment",
    "bearing":   "Bearing Degradation",
    "overload":  "Motor Overload",
}
FAULT_COLORS = {
    "normal": "#4ade80", "imbalance": "#60a5fa",
    "misalign": "#a78bfa", "bearing": "#f97316", "overload": "#ef4444",
}

# ── SIGNAL GENERATION (identical physics to simulation.py) ────────────────────
def make_signal(state: str, severity: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.arange(WIN) / SR
    def n(amp): return (rng.random(WIN) - 0.5) * 2 * amp
    sig = np.zeros(WIN)
    if state == "normal":
        sig = np.sin(2*np.pi*SHAFT_HZ*t)*0.35 + np.sin(2*np.pi*50*t)*0.08 + n(0.08)
    elif state == "imbalance":
        sig = np.sin(2*np.pi*SHAFT_HZ*t)*(0.35+0.95*severity) + np.sin(2*np.pi*50*t)*0.1 + n(0.1)
    elif state == "misalign":
        sig = np.sin(2*np.pi*SHAFT_HZ*t)*0.4 + np.sin(2*np.pi*50*t)*(0.55*severity) + np.sin(2*np.pi*75*t)*(0.22*severity) + n(0.1)
    elif state == "bearing":
        base = np.sin(2*np.pi*SHAFT_HZ*t)*0.3 + n(0.09)
        impulse_times = rng.random(int(6*severity)) * (WIN/SR)
        for it in impulse_times:
            idx = int(it * SR)
            if idx < WIN - 8:
                decay = np.exp(-np.arange(8)*2.5)
                base[idx:idx+8] += (0.8 + rng.random()*0.4) * severity * decay
        sig = base
    elif state == "overload":
        sig = sum(np.sin(2*np.pi*SHAFT_HZ*k*t)*(0.55*severity/k) for k in range(1,5)) + n(0.18)
    elif state == "belt":
        belt_freq = 6.0
        sig = np.sin(2*np.pi*SHAFT_HZ*t)*0.35 + np.sin(2*np.pi*belt_freq*t)*(0.45*severity) + n(0.12)
    return sig * 10.0   # convert to mm/s scale

# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────────
def extract_features(sig: np.ndarray) -> np.ndarray:
    rms  = np.sqrt(np.mean(sig**2))
    peak = np.max(np.abs(sig))
    spec = np.abs(rfft(sig)) * 2 / WIN
    freq = rfftfreq(WIN, 1/SR)
    # top 3 peaks
    idx = np.argsort(spec[1:])[::-1][:3] + 1
    f1, a1 = freq[idx[0]], spec[idx[0]]
    f2, a2 = freq[idx[1]], spec[idx[1]]
    return np.array([
        rms, peak, np.std(sig),
        float(spkurt(sig, fisher=False)),
        peak / rms if rms > 0 else 0,
        float(skew(sig)),
        f1, a1, f2, a2
    ])

FEAT_NAMES = ["RMS","Peak","Std","Kurtosis","CrestFactor","Skewness",
              "FFT_f1","FFT_a1","FFT_f2","FFT_a2"]

# ── BUILD DATASET (300 normal + 200 per fault, mixed severities) ───────────────
@st.cache_data(show_spinner="Building ML dataset …")
def build_dataset():
    X, y = [], []
    np.random.seed(42)
    # Normal training pool (300 windows)
    X_normal = [extract_features(make_signal("normal", seed=i)) for i in range(300)]
    for i, feat in enumerate(X_normal):
        X.append(feat); y.append(0)
    fault_keys = list(FAULT_LABELS.keys())
    for cls_idx, fname in enumerate(fault_keys):
        if fname == "normal": continue
        for j, sev in enumerate([0.6, 1.0, 1.5]):
            for k in range(66):
                seed = 1000 + cls_idx*1000 + j*100 + k
                X.append(extract_features(make_signal(fname, sev, seed=seed))); y.append(cls_idx)
    return np.array(X_normal), np.array(X), np.array(y), fault_keys

# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ SmartMill Uganda")
st.sidebar.markdown("**Final Year Project Portfolio**")
st.sidebar.markdown("*Computer Engineering & Informatics*")
st.sidebar.markdown("*Busitema University, Uganda*")
st.sidebar.divider()

pages = {
    "🏠  Overview":             "overview",
    "📍  Field Study":          "field",
    "🔧  System Architecture":  "arch",
    "📡  Live Signal Demo":     "signal",
    "🤖  ML Pipeline":          "ml",
    "🛠️  Skills & Stack":       "skills",
}
selection = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
page = pages[selection]

st.sidebar.divider()
st.sidebar.caption("SmartMill Uganda — Phase 1 Complete")
st.sidebar.caption("Phase 2: Prototype assembly in progress")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.title("SmartMill Uganda")
    st.markdown("#### IoT Predictive Maintenance for Grain Milling Equipment")
    st.markdown("""
    > *Reducing unplanned downtime at rural grain mills in Uganda through edge-computed 
    vibration analytics, machine learning anomaly detection, and solar-powered IoT hardware.*
    """)
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mills Surveyed", "4", "Busia & Tororo districts")
    col2.metric("Fault Types Modelled", "5", "FMEA-prioritised")
    col3.metric("IF Detection Rate", "≥ 97%", "severity × 1.2")
    col4.metric("RF Accuracy", "≥ 95%", "5-class, 10-feature")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### The Problem")
        st.markdown("""
        Rural grain mills across Eastern Uganda operate **without any condition monitoring**.
        Bearing failures, belt wear, and shaft misalignment go undetected until catastrophic
        breakdown — costing operators **UGX 800,000–2,500,000** per incident in lost income and 
        emergency repairs, often during peak post-harvest season.

        No commercial PdM system is designed for:
        - **Solar-only power** with LiFePO₄ backup
        - **2G/GSM-only connectivity**
        - **Operator literacy constraints** (SMS alerts, not dashboards)
        - **< USD 80 BOM cost**
        """)
    with c2:
        st.markdown("### The Solution")
        st.markdown("""
        **SmartMill** is a three-layer IoT system:

        | Layer | Technology |
        |-------|-----------|
        | **Sensor** | ESP32 + MPU6050 (vibration), ACS712 (current), NTC (temp) |
        | **Edge** | MicroPython firmware, 10-feature extraction, Isolation Forest |
        | **Cloud** | AWS IoT Core → Lambda → DynamoDB → SNS (SMS alert) |

        A **three-stage ML pipeline** (ISO rule-check → Isolation Forest → Random Forest)
        delivers named fault identification with no false-positive overload —
        designed specifically for operator trust in a low-connectivity rural context.
        """)

    st.info("👉 Use the sidebar to explore field data, live signal demos, and the ML pipeline interactively.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FIELD STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "field":
    st.title("📍 Field Study — Phase 1")
    st.markdown("Four sites visited in Busia and Tororo districts. Measurements taken with calibrated accelerometer at 1 kHz, 512-sample windows.")
    st.divider()

    field_data = pd.DataFrame({
        "Mill ID":       ["M01 (Busia Market)", "M02 (Tororo Town)", "M03 (Malaba Border)", "M04 (Busia Industrial)"],
        "Motor (kW)":    [7.5, 5.5, 11.0, 7.5],
        "Motor Age (yr)":[6, 4, 9, 3],
        "Vib RMS (mm/s)": [5.8, 4.4, 7.1, 3.2],
        "ISO Zone":      ["B", "B", "C ⚠️", "A ✓"],
        "Primary Fault": ["Rotor imbalance", "Shaft misalignment", "Bearing + misalign", "No fault detected"],
        "Dominant Freq (Hz)": [24.8, 49.6, 24.5, 25.1],
        "2× Harmonic Elevated": ["No", "Yes ⚠️", "Yes ⚠️", "No"],
        "Current Anomaly": ["Spikes detected", "Normal", "Normal", "Normal"],
    })
    st.dataframe(field_data, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vibration Severity by Site")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0f1221")
        ax.set_facecolor("#0f1221")
        mills = ["M01\nBusia Mkt", "M02\nTororo", "M03\nMalaba", "M04\nBusia Ind"]
        rms   = [5.8, 4.4, 7.1, 3.2]
        colors = ["#f59e0b", "#f59e0b", "#ef4444", "#4ade80"]
        bars = ax.bar(mills, rms, color=colors, edgecolor="#1c2040", linewidth=1.2, zorder=3)
        ax.axhline(ISO_A, color="#4ade80", linestyle="--", linewidth=1, label="Zone A/B (4 mm/s)")
        ax.axhline(ISO_B, color="#f59e0b", linestyle="--", linewidth=1, label="Zone B/C (7 mm/s)")
        ax.axhline(ISO_C, color="#ef4444", linestyle="--", linewidth=1, label="Zone C/D (11 mm/s)")
        ax.set_ylabel("RMS Velocity (mm/s)", color="#dde3f8", fontsize=9)
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.spines[:].set_color("#1c2040")
        ax.yaxis.label.set_color("#dde3f8")
        ax.legend(fontsize=7, facecolor="#0f1221", labelcolor="#dde3f8", edgecolor="#1c2040")
        ax.grid(axis="y", color="#1c2040", linewidth=0.6, zorder=0)
        for bar, val in zip(bars, rms):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val}", ha="center", va="bottom",
                    color="#dde3f8", fontsize=9, fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Field Findings")
        st.markdown("""
        **M03 (Malaba)** is the most critical site: the only mill in ISO Zone C, 
        with both a strong 2× misalignment harmonic and elevated broadband noise 
        consistent with spalled bearing inner race. Operator reported "unusual noise 
        for 3 weeks" — classic symptom of advancing BPFI fault.

        **M01 (Busia)** shows unbalanced rotor: dominant 1× harmonic elevation, 
        current spikes on startup consistent with mechanical eccentricity. Hammer 
        wear confirmed visually. 

        **M02 (Tororo)** shows clean misalignment signature: 2× harmonic exceeds 
        1× in amplitude, no broadband floor elevation. Recently aligned motor had 
        probably shifted on its mounting feet.

        **M04** was the only healthy mill — recently serviced, new bearings, 
        confirmed alignment. Its vibration profile was used as the Isolation Forest 
        training baseline.

        > All findings consistent with synthetic fault models in the simulation.
        """)
        st.markdown("**FMEA Top-5 (by RPN)**")
        fmea = pd.DataFrame({
            "Failure Mode": ["Bearing seizure", "Shaft misalignment", "Rotor imbalance", "Belt failure", "Motor overload"],
            "RPN": [360, 288, 210, 180, 189],
            "Field Observed": ["M02, M03", "M02, M03", "M01, M03", "All sites", "M01"],
        })
        st.dataframe(fmea, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "arch":
    st.title("🔧 System Architecture")
    st.markdown("A four-layer hardware + cloud architecture with solar power subsystem spanning all layers.")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Hardware Stack")
        hw = pd.DataFrame({
            "Component":  ["ESP32 WROOM-32", "MPU6050 IMU", "ACS712-5A", "ZMPT101B", "NTC Thermistor", "HX711 + Load Cell", "18V / 20Wp PV Panel", "LS1024B MPPT", "3.7V 3000mAh LiPo", "MT3608 Boost", "SSD1306 OLED"],
            "Function":   ["Main MCU + WiFi/BT", "3-axis vibration (±16g)", "Current sensing (0–5A)", "Voltage sensing (AC)", "Motor temp monitoring", "Grain throughput (kg)", "Solar input", "Solar charge controller", "Energy storage", "3.7V → 5V for ESP32", "Local operator display"],
            "Interface":  ["—", "I²C 0x68", "ADC GPIO34", "ADC GPIO35", "ADC GPIO32", "SPI", "—", "—", "—", "—", "I²C 0x3C"],
            "BOM Cost ($)":["3.50", "0.80", "1.20", "1.10", "0.40", "4.50", "18.00", "9.00", "8.00", "0.60", "3.50"],
        })
        st.dataframe(hw, use_container_width=True, hide_index=True)
        st.caption(f"**Total BOM estimate:** USD ~${sum([3.5,0.8,1.2,1.1,0.4,4.5,18,9,8,0.6,3.5]):.2f}  |  Target: < USD 80 ✓")

    with col2:
        st.markdown("### Cloud Pipeline")
        st.markdown("""
        ```
        ESP32 (Edge)
         │  MicroPython firmware
         │  512-sample window (0.512s)
         │  10-feature extraction
         │  Stage 0 ISO rule-check
         │  Stage 1 Isolation Forest
         │  MQTT publish (JSON)
         │
        AWS IoT Core
         │  Thing registry / X.509 cert
         │  Topic: smartmill/{site}/telemetry
         │
        AWS Lambda
         │  Stage 2 Random Forest (Python)
         │  Fault classification
         │  Alert trigger logic
         │
        AWS DynamoDB
         │  Telemetry archive (time-series)
         │  Alert history
         │
        AWS SNS
         │  SMS via Twilio (2G compatible)
         └→ Operator receives:
            "M03 BEARING FAULT VIB:0.91g
             Schedule maintenance."
        ```
        """)

    st.divider()
    st.markdown("### Three-Stage Detection Pipeline")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **Stage 0 — ISO Rule Check**
        - Runs on ESP32, no cloud needed
        - RMS velocity vs ISO 10816 zones
        - Zone C → orange alert
        - Zone D → red + immediate SMS
        - Zero training data required
        - ✅ Active from day 1
        """)
    with col_b:
        st.markdown("""
        **Stage 1 — Isolation Forest**
        - Trains on 300 normal windows
        - Learns what *this mill* looks like healthy
        - Flags anything statistically anomalous
        - No fault labels needed
        - 5th percentile threshold (~10% FPR)
        - ✅ Active after ~4hr calibration
        """)
    with col_c:
        st.markdown("""
        **Stage 2 — Random Forest**
        - 10-feature, 5-class classifier
        - Named fault identification
        - "BEARING FAULT" not just "anomaly"
        - Requires labeled training data
        - Runs in AWS Lambda (not ESP32)
        - ✅ Active once field data accumulates
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE SIGNAL DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "signal":
    st.title("📡 Live Signal Demo")
    st.markdown("Select a fault type and severity — the engine generates a synthetic vibration signal using the exact same physics model as the simulation, then extracts the 10-element feature vector.")
    st.divider()

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        fault_choice = st.selectbox(
            "Fault Type",
            options=list(FAULT_LABELS.keys()),
            format_func=lambda k: FAULT_LABELS[k],
        )
        severity = st.slider("Severity", 0.3, 2.0, 1.0, 0.05,
                             help="Scale factor applied to fault amplitude. 1.0 = field-measured baseline.")
        seed_val = st.number_input("Random seed", 0, 9999, 42,
                                   help="Change to get a different realisation of the same fault.")
        st.divider()
        st.markdown("**Extracted Features**")

    sig = make_signal(fault_choice, severity, seed=int(seed_val))
    feat = extract_features(sig)

    with col_ctrl:
        feat_df = pd.DataFrame({"Feature": FEAT_NAMES, "Value": [f"{v:.4f}" for v in feat]})
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=340)
        # ISO zone
        rms_mms = feat[0]
        zone = "A ✓" if rms_mms < ISO_A else "B" if rms_mms < ISO_B else "C ⚠️" if rms_mms < ISO_C else "D 🚨"
        st.metric("RMS (mm/s)", f"{rms_mms:.2f}", f"ISO Zone {zone}")
        st.metric("Kurtosis", f"{feat[3]:.2f}", "Normal baseline = 3.0")

    with col_plot:
        t = np.arange(WIN) / SR * 1000  # ms
        freq = rfftfreq(WIN, 1/SR)
        spec = np.abs(rfft(sig)) * 2 / WIN

        fig = plt.figure(figsize=(9, 6), facecolor="#0f1221")
        gs  = gridspec.GridSpec(2, 1, hspace=0.42)
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
        color = FAULT_COLORS[fault_choice]

        # Time domain
        ax1.plot(t, sig, color=color, linewidth=0.9, alpha=0.9)
        ax1.axhline(0, color="#1c2040", linewidth=0.6)
        ax1.set_xlabel("Time (ms)", color="#6b7280", fontsize=8)
        ax1.set_ylabel("Velocity (mm/s)", color="#dde3f8", fontsize=8)
        ax1.set_title(f"Time-Domain Signal — {FAULT_LABELS[fault_choice]}", color="#dde3f8", fontsize=10, fontweight="bold")
        ax1.set_facecolor("#080b14"); ax1.tick_params(colors="#6b7280", labelsize=7)
        ax1.spines[:].set_color("#1c2040")

        # Frequency domain
        ax2.fill_between(freq[:WIN//2], spec[:WIN//2], color=color, alpha=0.35)
        ax2.plot(freq[:WIN//2], spec[:WIN//2], color=color, linewidth=1.2)
        ax2.axvline(SHAFT_HZ,   color="#60a5fa", linewidth=1, linestyle="--", label=f"1× {SHAFT_HZ}Hz")
        ax2.axvline(2*SHAFT_HZ, color="#a78bfa", linewidth=1, linestyle="--", label=f"2× {2*SHAFT_HZ}Hz")
        ax2.axvline(3*SHAFT_HZ, color="#6366f1", linewidth=0.8, linestyle=":", label=f"3× {3*SHAFT_HZ}Hz")
        ax2.axvline(DEFECT_HZ,  color="#f97316", linewidth=1, linestyle="--", label=f"BPFI {DEFECT_HZ}Hz")
        ax2.set_xlabel("Frequency (Hz)", color="#6b7280", fontsize=8)
        ax2.set_ylabel("|Amplitude| (mm/s)", color="#dde3f8", fontsize=8)
        ax2.set_title("FFT Frequency Spectrum", color="#dde3f8", fontsize=10, fontweight="bold")
        ax2.set_xlim(0, 300); ax2.set_facecolor("#080b14")
        ax2.tick_params(colors="#6b7280", labelsize=7)
        ax2.spines[:].set_color("#1c2040")
        ax2.legend(fontsize=7, facecolor="#0f1221", labelcolor="#dde3f8",
                   edgecolor="#1c2040", ncol=4, loc="upper right")

        st.pyplot(fig, use_container_width=True)

    st.info("💡 **Notice:** Bearing degradation → kurtosis jumps above 3 as impulses appear. "
            "Misalignment → the 2× harmonic (50 Hz) dominates the FFT. "
            "These are the exact signatures measured at M02 and M03 in the field.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ml":
    st.title("🤖 ML Pipeline — Live Demo")
    st.divider()

    X_normal, X_all, y_all, fault_keys = build_dataset()

    tab1, tab2 = st.tabs(["Stage 1 — Isolation Forest", "Stage 2 — Random Forest"])

    # ── TAB 1: Isolation Forest ────────────────────────────────────────────────
    with tab1:
        st.markdown("### Unsupervised Anomaly Detection")
        st.markdown("""
        Trained **only on 300 normal windows** — no fault labels needed.
        The model learns the normal vibration profile, then flags anything statistically
        different as anomalous. This is the algorithm deployed on day 1 of installation.
        """)

        col_p, col_r = st.columns([1, 1])
        with col_p:
            contamination = st.slider("Contamination (expected anomaly fraction)", 0.01, 0.15, 0.05, 0.01)
            test_severity = st.slider("Test fault severity", 0.5, 2.0, 1.2, 0.1)

        with st.spinner("Training Isolation Forest …"):
            clf_if = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            clf_if.fit(X_normal)
            raw_scores = clf_if.score_samples(X_normal)
            threshold  = np.percentile(raw_scores, contamination * 100)

            def anomaly_score_01(scores):
                lo, hi = raw_scores.min(), raw_scores.max()
                return np.clip((scores - lo) / (hi - lo + 1e-9), 0, 1)

        # Generate test data for each fault class
        results = {}
        for fname in fault_keys:
            test_sigs   = [make_signal(fname, test_severity, seed=500+i) for i in range(60)]
            test_feats  = np.array([extract_features(s) for s in test_sigs])
            raw = clf_if.score_samples(test_feats)
            scores_01   = anomaly_score_01(raw)
            detected    = np.sum(raw < threshold)
            results[fname] = {"scores": scores_01, "detected": detected, "total": 60}

        thresh_01 = anomaly_score_01(np.array([threshold]))[0]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1221")
        ax.set_facecolor("#080b14")
        positions = []
        labels    = []
        all_scores = []
        for i, (fname, res) in enumerate(results.items()):
            positions.append(i)
            labels.append(FAULT_LABELS[fname])
            all_scores.append(res["scores"])

        bp = ax.boxplot(all_scores, positions=positions, patch_artist=True,
                        widths=0.55, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for patch, fname in zip(bp["boxes"], fault_keys):
            patch.set_facecolor(FAULT_COLORS[fname] + "44")
            patch.set_edgecolor(FAULT_COLORS[fname])
        for median in bp["medians"]:
            median.set_color("#ffffff"); median.set_linewidth(2)
        for whisker in bp["whiskers"]: whisker.set_color("#4a5278")
        for cap in bp["caps"]: cap.set_color("#4a5278")

        ax.axhline(thresh_01, color="#ef4444", linewidth=2, linestyle="--",
                   label=f"Alert threshold ({contamination*100:.0f}th pct)")
        ax.fill_between([-0.5, len(fault_keys)-0.5], [thresh_01, thresh_01], [1, 1],
                        color="#ef444415", zorder=0)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=12, color="#dde3f8", fontsize=8)
        ax.set_ylabel("Anomaly Score (0 = normal, 1 = max anomalous)", color="#dde3f8", fontsize=9)
        ax.set_title(f"Isolation Forest — Anomaly Score Distribution  |  Severity × {test_severity}",
                     color="#dde3f8", fontsize=10)
        ax.tick_params(colors="#6b7280", labelsize=8); ax.spines[:].set_color("#1c2040")
        ax.legend(fontsize=8, facecolor="#0f1221", labelcolor="#dde3f8", edgecolor="#1c2040")
        st.pyplot(fig, use_container_width=True)

        # Detection rate table
        det_df = pd.DataFrame([{
            "Fault Class":    FAULT_LABELS[fname],
            "Detected":       f'{r["detected"]}/{r["total"]}',
            "Detection Rate": f'{r["detected"]/r["total"]*100:.1f}%',
            "Status":         "✅ Reliable" if r["detected"]/r["total"] > 0.85 else "⚠️ Marginal",
        } for fname, r in results.items()])
        st.dataframe(det_df, use_container_width=True, hide_index=True)

    # ── TAB 2: Random Forest ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### Supervised Fault Classification")
        st.markdown("""
        Once labeled field data is available, the Random Forest provides **named fault identification** —
        not just "anomaly" but "BEARING FAULT" or "SHAFT MISALIGNMENT".
        Runs in AWS Lambda; the ESP32 only runs Stage 1.
        """)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            n_trees = st.slider("Number of trees", 50, 300, 150, 25)
        with col_c2:
            test_frac = st.slider("Test split", 0.2, 0.4, 0.3, 0.05)

        with st.spinner("Training Random Forest …"):
            X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all,
                                                        test_size=test_frac, stratify=y_all, random_state=42)
            clf_rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
            clf_rf.fit(X_tr, y_tr)
            y_pred = clf_rf.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)

        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.metric("Overall Accuracy", f"{acc*100:.1f}%", f"n_test = {len(y_te)}")
            class_names = [FAULT_LABELS[k] for k in fault_keys]
            cm = confusion_matrix(y_te, y_pred)
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5), facecolor="#0f1221")
            ax2.set_facecolor("#080b14")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax2, colorbar=False, cmap="Blues")
            ax2.set_title("Confusion Matrix", color="#dde3f8", fontsize=10)
            ax2.tick_params(colors="#dde3f8", labelsize=7)
            ax2.xaxis.label.set_color("#dde3f8"); ax2.yaxis.label.set_color("#dde3f8")
            for text in ax2.texts: text.set_color("#dde3f8")
            plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")
            st.pyplot(fig2, use_container_width=True)

        with col_fi:
            st.markdown("**Feature Importance**")
            fi = clf_rf.feature_importances_
            fi_df = pd.DataFrame({"Feature": FEAT_NAMES, "Importance": fi}) \
                      .sort_values("Importance", ascending=True)
            fig3, ax3 = plt.subplots(figsize=(5.5, 4.5), facecolor="#0f1221")
            ax3.set_facecolor("#080b14")
            colors_fi = ["#f59e0b" if f in ["Kurtosis","FFT_a1","FFT_a2","RMS"] else "#4a5278"
                         for f in fi_df["Feature"]]
            ax3.barh(fi_df["Feature"], fi_df["Importance"], color=colors_fi,
                     edgecolor="#1c2040", linewidth=0.6)
            ax3.set_xlabel("Mean Decrease in Impurity", color="#dde3f8", fontsize=8)
            ax3.set_title("Feature Importance", color="#dde3f8", fontsize=10)
            ax3.tick_params(colors="#6b7280", labelsize=8); ax3.spines[:].set_color("#1c2040")
            st.pyplot(fig3, use_container_width=True)
            st.caption("🟡 Highlighted = top discriminating features: Kurtosis (bearing), FFT_a1 (imbalance), FFT_a2 (misalignment)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SKILLS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "skills":
    st.title("🛠️ Skills & Technical Stack")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔩 Hardware & Embedded")
        skills_hw = {
            "ESP32 / MicroPython": 85,
            "Circuit design (sensor integration)": 75,
            "I²C / SPI protocols": 80,
            "Solar / MPPT power systems": 70,
            "Oscilloscope / signal debugging": 70,
        }
        for skill, level in skills_hw.items():
            st.markdown(f"**{skill}**")
            st.progress(level)

    with col2:
        st.markdown("### 🧠 Machine Learning & Data")
        skills_ml = {
            "Python (NumPy / SciPy / Pandas)": 90,
            "scikit-learn (IF, RF, SVM)": 85,
            "Signal processing (FFT, features)": 80,
            "Matplotlib / Seaborn / Plotly": 80,
            "Streamlit (ML dashboards)": 75,
        }
        for skill, level in skills_ml.items():
            st.markdown(f"**{skill}**")
            st.progress(level)

    with col3:
        st.markdown("### ☁️ Cloud & Frontend")
        skills_cloud = {
            "AWS IoT Core / Lambda / DynamoDB": 75,
            "AWS SNS (SMS alerting)": 70,
            "MQTT protocol": 75,
            "React (dashboards / digital twins)": 80,
            "Technical documentation & report writing": 85,
        }
        for skill, level in skills_cloud.items():
            st.markdown(f"**{skill}**")
            st.progress(level)

    st.divider()
    st.markdown("### 📁 Project Artefacts Produced — SmartMill Uganda")
    artefacts = pd.DataFrame({
        "Artefact": [
            "Phase 1 Field Study Report",
            "Python Signal Simulation (matplotlib)",
            "React Digital Twin Dashboard",
            "This Streamlit Portfolio App",
            "System Architecture SVG",
            "Hardware Block Diagram",
            "Physical Prototype (in progress)",
        ],
        "Format":      ["DOCX/PDF", "Python (.py)", "React (.jsx)", "Python (.py)", "SVG", "SVG", "Physical + Photo"],
        "Status":      ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "🔄 Phase 2"],
        "Demonstrates": [
            "Research methodology, FMEA, ISO 10816",
            "Signal physics, FFT, ML pipeline (sklearn)",
            "Real-time physics simulation, React, Recharts",
            "Python depth, ML intuition, communication",
            "Systems architecture communication",
            "Hardware design literacy",
            "Hands-on prototype build",
        ],
    })
    st.dataframe(artefacts, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🎓 About This Project")
    st.markdown("""
    SmartMill Uganda is a final-year capstone project at **Busitema University, Department of Computer Engineering and Informatics**.
    It is being developed as a real deployable system — not a purely academic exercise.
    Phase 1 (field study + simulation + architecture) is complete. Phase 2 (prototype assembly) is underway.

    The project targets rural grain mill operators in Eastern Uganda and is designed to be
    deployable at a BOM cost of under USD 80, powered entirely by solar, and operable over 2G.
    """)
