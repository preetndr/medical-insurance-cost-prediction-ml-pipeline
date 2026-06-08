import streamlit as st
import pandas as pd
import pickle
import numpy as np
import streamlit.components.v1 as components

# =============================================================================
# Streamlit Application Configuration
# =============================================================================
st.set_page_config(
    page_title="Health Insurance Estimator",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Custom UI Styling (Brown & Pink Theme)
# -----------------------------------------------------------------------------
# Contains premium UI styling, animations, typography, inputs, button
# interactions, and responsive visual enhancements.
# =============================================================================
st.markdown(
    """
    <style>
    
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700,800&f[]=satoshi@400,500,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, span, div, li {
        font-family: 'Satoshi', sans-serif !important;
    }
    
    h1, h2, h3, .gradient-text, .section-header {
        font-family: 'Clash Display', sans-serif !important;
    }
    
    button, .stButton > button,
    label, .stLabel,
    input, select, textarea,
    div[data-baseweb="input"] *,
    div[data-baseweb="select"] * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.03em !important; 
    }

    /* Base Theme */
    .stApp {
        background-color: #1a1615; /* Deep rich brown/black background */
    }

    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.03;
        z-index: 0;
        background-image:
            radial-gradient(rgba(244, 114, 182, 0.4) 1px, transparent 1px); /* Pink dot matrix */
        background-size: 6px 6px;
    }

    /* Labels */
    .stNumberInput label, .stSelectbox label {
        color: #A09692 !important; /* Soft brown/grey */
        font-size: 0.85rem !important;
        font-weight: 600 !important; 
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
        margin-bottom: 12px !important;
        transition: color 0.4s ease !important;
    }
    .stNumberInput:hover label, .stSelectbox:hover label {
        color: #FFFFFF !important; 
    }

    /* Input & Selectbox Styling (Glassmorphism) */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        min-height: 44px !important; /* Uniform height to prevent clipping */
        padding: 0px 12px !important; /* Removed vertical padding that squishes text */
        box-shadow: none !important; 
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div[data-baseweb="input"] > div:hover, div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(244, 114, 182, 0.35) !important; /* Pink border */
        box-shadow: 0 0 15px rgba(244, 114, 182, 0.05) !important;
    }
    
    /* Input Text Coloring */
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div > div:first-child,
    div[data-baseweb="select"] > div > div:first-child * {
        color: #F472B6 !important; /* Pink text */
        -webkit-text-fill-color: #F472B6 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        transition: color 0.3s ease !important;
    }

    div[data-baseweb="input"]:hover input,
    div[data-baseweb="select"]:hover > div > div:first-child {
        text-shadow: 0 0 10px rgba(244, 114, 182, 0.3);
    }
    
    /* Perfectly center the selectbox text */
    div[data-baseweb="select"] > div > div:first-child {
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
    }

    /* Section Headers */
    .section-header {
        color: #8D6E63; /* Warm Brown */
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(141, 110, 99, 0.2); 
        background: none;
        box-shadow: none;
        display: block;
    }

    /* Premium Button */
    .element-container:has(.stButton) {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        margin-top: 18px !important;
    }

    div.stButton {
        width: auto !important;
        display: flex !important;
        justify-content: center !important;
    }

    div.stButton > button {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 13px 30px !important;
        border-radius: 8px !important;
        background: rgba(255,255,255,0.045) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #D8D8D8 !important;
        backdrop-filter: blur(10px) !important;
        overflow: hidden !important;

        transition:
            transform 0.45s cubic-bezier(0.16, 1, 0.3, 1),
            background 0.45s ease,
            border-color 0.45s ease,
            box-shadow 0.45s ease !important;

        box-shadow:
            0 0 0 rgba(244, 114, 182, 0),
            0 8px 30px rgba(0,0,0,0.18);
    }

    div.stButton > button p,
    div.stButton > button span {
        margin: 0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        transition: color 0.35s ease !important;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        inset: -40%;
        background:
            radial-gradient(
                circle at center,
                rgba(244, 114, 182, 0.20) 0%, /* Pink glow */
                rgba(141, 110, 99, 0.15) 30%, /* Brown edge */
                transparent 70%
            );

        opacity: 0;
        transform: translateX(-30%) translateY(10%) scale(0.8);
        transition:
            opacity 0.6s ease,
            transform 0.8s cubic-bezier(0.16,1,0.3,1);
        pointer-events: none;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.015);
        background: rgba(255,255,255,0.065) !important;
        border-color: rgba(244, 114, 182, 0.3) !important;
        box-shadow:
            0 0 40px rgba(244, 114, 182, 0.12),
            0 12px 40px rgba(0,0,0,0.3);
    }

    div.stButton > button:hover::before {
        opacity: 1;
        transform: translateX(15%) translateY(-10%) scale(1.15);
    }

    div.stButton > button:hover p,
    div.stButton > button:hover span {
        color: white !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) scale(0.985);
    }

    /* Input Animations */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stNumberInput, .stSelectbox {
        animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        margin-bottom: 24px !important;
    }

    /* Gradient Text (Brown & Pink) */
    @keyframes textShine {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    .gradient-text {
        background: linear-gradient(120deg, #8D6E63 30%, #F472B6 50%, #8D6E63 70%); /* Brown & Pink */
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }
    
    .block-container {
        margin-top: -64px !important; 
    }

    .section-header {
        margin-top: 62px;
    }

    /* Page load animation */
    .main .block-container {
        animation: pageReveal 900ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes pageReveal {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Result Reveal Animation */
    .result-reveal {
        animation: resultReveal 650ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes resultReveal {
        from { opacity: 0; transform: translateY(14px); filter: blur(6px); }
        to { opacity: 1; transform: translateY(0); filter: blur(0px); }
    }

    </style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Load Models (Cached for speed)
# =============================================================================
@st.cache_resource
def load_models():
    """Loads the trained pipeline and the target transformer"""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("target_transformer.pkl", "rb") as f:
        target_transformer = pickle.load(f)
    return model, target_transformer


try:
    pipeline, pt = load_models()
except FileNotFoundError:
    st.error(
        "Error: Missing 'model.pkl' or 'target_transformer.pkl' in the current directory."
    )
    st.stop()


st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# Hero Section
# =============================================================================
st.markdown(
    """
    <h1 style='font-size: 60px; margin-bottom: 0; line-height: 1.1; text-align: center;'>
        Health Insurance<br>
        <span class='gradient-text' style='font-size: 68px; font-weight: 800; display: block;'>Estimator</span>
    </h1>
    <p style='color: #A09692; font-size: 16px; margin-top: 20px; text-align: center; font-weight: 500;'>
        Predicting patient annual charges using demographic analysis.
    </p>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <hr style="
        width:76%;
        margin:-34px auto 26px auto;
        border:none;
        height:0.5px;
        background-color:rgba(255,255,255,0.06);
    ">
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# =============================================================================
# User Input Section
# =============================================================================

st.markdown(
    "<div class='section-header'>👤 Patient Details</div>", unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])


st.markdown(
    "<div class='section-header'>⚖️ Body Metrics & Family</div>", unsafe_allow_html=True
)
col3, col4 = st.columns(2)
with col3:
    bmi = st.number_input(
        "BMI", min_value=15.00, max_value=55.00, value=25.00, step=0.10
    )
with col4:
    children = st.number_input(
        "Dependents (Children)", min_value=0, max_value=5, value=0, step=1
    )


st.markdown(
    "<div class='section-header'>🏥 Lifestyle & Location</div>", unsafe_allow_html=True
)
col5, col6 = st.columns(2)
with col5:
    smoker = st.selectbox("Smoker Status", options=["No", "Yes"])
with col6:
    region = st.selectbox(
        "Region", options=["Southwest", "Southeast", "Northwest", "Northeast"]
    )


# =============================================================================
# Prediction Execution
# =============================================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Calculate Estimate"):
    # Construct DataFrame to match the exact format of the training data
    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex.lower()],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker.lower()],
            "region": [region.lower()],
        }
    )

    with st.spinner("Processing demographics..."):
        try:
            # 1. Model predicts the transformed target
            transformed_prediction = pipeline.predict(input_data)

            # 2. Inverse transform back to actual dollars
            actual_charge = pt.inverse_transform(
                transformed_prediction.reshape(-1, 1)
            ).ravel()[0]

            # 3. Premium Result Reveal
            st.markdown(
                """
                <div class="result-reveal" style="text-align:center; margin-top:38px; padding:42px 20px 12px;">
                    <p style="color:#8D6E63; font-size:11px; font-weight:700; letter-spacing:0.22em; text-transform:uppercase; margin:0 0 18px;">
                        Estimated Annual Cost
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h1 style='text-align:center; font-size:68px; margin:0; font-family:Clash Display, sans-serif; line-height:1;' class='gradient-text'><span style='font-size:32px; vertical-align:super;'>$</span>{actual_charge:,.2f}</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='width:46px; height:1px; background:rgba(244, 114, 182, 0.22); margin:28px auto 0;'></div>",
                unsafe_allow_html=True,
            )

            # 4. Auto-Scroll to Result Card Using JavaScript
            components.html(
                """
                <script>
                    setTimeout(function() {
                        const result = window.parent.document.querySelector('.result-reveal');
                        if (result) {
                            result.scrollIntoView({
                                behavior: 'smooth', block: 'center'
                            });
                        }
                    }, 100);
                </script>
                """,
                height=0,
            )

        except Exception as e:
            st.error(f"Computation Error: {e}")

st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

# =============================================================================
# Frontend Interaction Enhancements
# -----------------------------------------------------------------------------
# JavaScript used for hover motion effects and smooth UI interactions.
# =============================================================================
components.html(
    """
    <script>
    const doc = window.parent.document;

    function centerButton() {
        doc.querySelectorAll('.stButton').forEach(el => {
            el.style.setProperty('display', 'flex', 'important');
            el.style.setProperty('justify-content', 'center', 'important');
            let parent = el.parentElement;
            while (parent) {
                parent.style.setProperty('display', 'flex', 'important');
                parent.style.setProperty('justify-content', 'center', 'important');
                if (parent.classList.contains('block-container')) break;
                parent = parent.parentElement;
            }
        });
    }

    centerButton();
    new MutationObserver(centerButton).observe(doc.body, { childList: true, subtree: true });

    const buttons = doc.querySelectorAll('.stButton > button');

    buttons.forEach(btn => {
        btn.addEventListener('mousemove', e => {
            const rect = btn.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const moveX = (x - rect.width / 2) * 0.04;
            const moveY = (y - rect.height / 2) * 0.10;

            btn.style.transform = `
                translate(${moveX}px, ${moveY - 2}px)
                scale(1.015)
            `;
        });

        btn.addEventListener('mouseleave', () => {
            btn.style.transform = '';
        });
    });
    </script>
    """,
    height=0,
    width=0,
)
