import streamlit as st
import pandas as pd
import pickle
import numpy as np
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. Page Configuration & Advanced Theme CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Insurance Estimator",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Advanced CSS: Radial background, Glassmorphism, and Animated UI elements
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force a dark radial background with grid */
    .stApp {
        background-color: #0F172A;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'><circle cx='20' cy='20' r='1' fill='rgba(255,255,255,0.05)'/></svg>");
        font-family: 'Inter', sans-serif;
    }
    
    /* App Logo/Icon */
    .app-logo {
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: -15px; 
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    
    /* Elegant typography */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #F8FAFC; 
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #94A3B8; 
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 3rem;
    }
    
    /* Input Field Styling */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"] > div:hover, div[data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2) !important;
    }
    .stNumberInput label, .stSelectbox label {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.5), 0 8px 10px -6px rgba(2, 132, 199, 0.1) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    /* Glassmorphism Result Card */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);
        text-align: center;
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0284C7, #38BDF8);
    }
    .result-label {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .result-value {
        color: #F8FAFC;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        margin: 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Load Models (Cached for speed)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Application Header
# -----------------------------------------------------------------------------
st.markdown("<div class='app-logo'>⚕️</div>", unsafe_allow_html=True)
st.markdown(
    "<h1 class='main-title'>Health Insurance Cost Estimator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Enter patient demographics to predict annual charges</p>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Clean User Input Layout
# -----------------------------------------------------------------------------
# Using columns to create a balanced, symmetrical form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    bmi = st.number_input(
        "BMI", min_value=15.00, max_value=55.00, value=25.00, step=0.10
    )
    children = st.number_input(
        "Dependents (Children)", min_value=0, max_value=5, value=0, step=1
    )

with col2:
    smoker = st.selectbox("Smoker Status", options=["No", "Yes"])
    sex = st.selectbox("Sex", options=["Male", "Female"])
    region = st.selectbox(
        "Region", options=["Southwest", "Southeast", "Northwest", "Northeast"]
    )

# -----------------------------------------------------------------------------
# 5. Prediction Logic & Auto-Scroll
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:
    predict_button = st.button("Calculate Estimate", use_container_width=True)

if predict_button:
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

    with st.spinner("Processing..."):
        # 1. Model predicts the transformed target
        transformed_prediction = pipeline.predict(input_data)

        # 2. Inverse transform back to actual dollars
        actual_charge = pt.inverse_transform(
            transformed_prediction.reshape(-1, 1)
        ).ravel()[0]

        # 3. Display Result Card
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Annual Cost</div>
                <div class="result-value">${actual_charge:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Auto-Scroll to Result Card Using JavaScript
        components.html(
            """
            <script>
            // We use a slight delay to ensure Streamlit has finished rendering the HTML
            setTimeout(function() {
                const doc = window.parent.document;
                const resultCard = doc.querySelector('.result-card');
                if (resultCard) {
                    // Scrolls the screen so the result card is in the center
                    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);
            </script>
            """,
            height=0,
            width=0,
        )

# -----------------------------------------------------------------------------
# 6. JavaScript Injector (Button Animations)
# -----------------------------------------------------------------------------
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
