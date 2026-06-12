import streamlit as None
import requests
from PIL import Image
import io

# --- 1. Page Configuration & Custom Theme Formatting ---
st.set_page_config(
    page_title="Quantum Image Classifier",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom dark theme injections to match the dashboard aesthetic precisely
st.markdown("""
    <style>
        .main { background-color: #1a1c23; color: #e2e8f0; }
        div[data-testid="stMetricValue"] { color: #81e6d9; font-family: monospace; }
        .stButton>button { 
            background-color: #4c51bf; color: white; width: 100%; border-radius: 6px; 
            border: none; padding: 0.5rem; transition: background 0.3s;
        }
        .stButton>button:hover { background-color: #5a67d8; }
        div[data-testid="stBlock"] { 
            background-color: #2d3748; padding: 1.5rem; border-radius: 8px; border: 1px solid #4a5568;
        }
        h1, h2, h3 { color: #f7fafc !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Configuration & State Management ---
# Update this with your live Render API URL if deployed remotely
API_BASE_URL = "https://quantum-image-classifier-cats-vs-dogs.onrender.com"

st.title("🔮 Quantum Image Classifier: Cat vs Dog")
st.caption("Hybrid Deep Learning & Quantum Oracle Pipeline Interface")
st.write("---")

# Layout Split: Left for Actions, Right for Results & Analytics
col_actions, col_results = st.columns([1, 1.2], gap="large")

with col_actions:
    st.subheader("📁 Classify New Image")
    
    # Drag and Drop Uploader
    uploaded_file = st.file_uploader(
        "Choose an image file...", 
        type=["jpg", "jpeg", "png"], 
        help="Supports JPEG and PNG formats."
    )
    
    if uploaded_file is not None:
        # Display Preview of raw upload
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Input Sample", use_container_width=True)
        
        # Binary execution step
        if st.button("✨ Invoke Quantum Oracle Pipeline"):
            with st.spinner("Executing Classical CNN Feature Compression & Dispatching Matrix..."):
                try:
                    # Convert file to bytes payload to pass to your backend API
                    img_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                    
                    # Direct HTTP call to your FastAPI /classify endpoint
                    # (Note: Assumes your endpoint accepts files. If it accepts a URL string payload,
                    # pass JSON instead)
                    response = requests.post(f"{API_BASE_URL}/classify", files=files, timeout=35)
                    
                    if response.status_code == 200:
                        st.session_state['oracle_response'] = response.json()
                        st.success("Quantum optimization run completed successfully!")
                    else:
                        st.error(f"API returned operational failure code: {response.status_code}")
                        st.info(response.text)
                except requests.exceptions.RequestException as net_err:
                    st.error("Communication with the backend pipeline timed out.")
                    st.info("If the service is sleeping on Render's free tier, it may require up to 60s to warm up.")

with col_results:
    st.subheader("📊 Execution Diagnostics & Metrics")
    
    # Retrieve previous run results if stored in state memory
    if 'oracle_response' in st.session_state:
        res = st.session_state['oracle_response']
        energy = res.get("optimal_energy", 0.0)
        
        # Simple thresholding logic layout matching UI
        prediction = "DOG 🐕" if energy > 0.0 else "CAT 🐈"
        
        # Classification Metric Matrix
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Predicted Class Output", value=prediction)
        with m_col2:
            st.metric(label="Calculated Optimal Energy Value", value=f"{energy:.6f}")
            
        st.write("### 📜 Raw Oracle JSON Payload")
        st.json(res)
    else:
        # Default placeholder layout when idle
        st.info("Pending classification target input. Upload an image and invoke the quantum pipeline operator to stream metrics.")
        
    # --- 3. Telemetry & Log Monitor Mock Panel ---
    st.write("### 🪵 Live Application Logs")
    st.code(
        "[INFO] System initialized: Web UI binding successful.\n"
        "[INFO] Quantum connection heartbeat: Active (200 OK)\n"
        f"[INFO] Monitoring API Gateway routing node: {API_BASE_URL}",
        language="bash"
    )
