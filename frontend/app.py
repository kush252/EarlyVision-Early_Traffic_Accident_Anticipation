import streamlit as st
import streamlit.components.v1 as components
import requests
import os
import tempfile
import cv2
import time
import numpy as np
import plotly.graph_objects as go

# Backend API URL
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Accident Risk Prediction", layout="wide")

st.title("Accident Risk Prediction Dashboard")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        /* Global Font and Color */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }
        
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #ffffff;
        }

        /* Black Theme with Subtle White Blur */
        .stApp {
            background-color: #000000;
            background-image: 
                radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* White Text for Content */
        .stMarkdown, .stText, p, label, div {
            color: #ffffff !important;
        }
        
        /* Only Title (h1) Red */
        h1 {
            color: #ff0000 !important;
        }
        
        /* Other headers White */
        h2, h3 {
            color: #ffffff !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 50, 50, 0.2);
        }
        
        /* Primary Button (Red) */
        div.stButton > button:first-child {
            background-color: #e60000;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #ff3333;
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.6);
            transform: translateY(-2px);
        }

        /* Titles/Headers Gradient */
        h1 {
            background: linear-gradient(90deg, #ffffff, #ff4444);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
### Instructions:
1. **Select Input Source**: Choose between uploading your own video or using a pre-loaded sample from the sidebar.
2. **Upload/Select Video**: 
    - If uploading, ensure the file is in `.mp4`, `.avi`, or `.mov` format.
    - **Note**: Video should be at least **5 seconds** long and ideally around **10 FPS** for best results.
    - If using a sample, pick one from the dropdown list.
3. **Analyze**: Click the **"Analyze Risk"** button to start the AI inference.
4. **View Results**: Watch the processed video with real-time risk probability graph.
""")

# Sidebar for Input Selection
st.sidebar.header("Input Selection")
input_option = st.sidebar.radio("Choose Input Source", ("Upload Video", "Use Sample Video"))

video_file_path = None

if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video (mp4, avi)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_file_path = tfile.name

elif input_option == "Use Sample Video":
    # List samples from the project directory
    # Assuming the app is run from project root, or we can find absolute paths
    # Hardcoding paths based on project structure known
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample data")
    
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith('.mp4')]
        selected_sample = st.selectbox("Select a sample video", files)
        if selected_sample:
            video_file_path = os.path.join(SAMPLE_DIR, selected_sample)
    else:
        st.error(f"Sample directory not found: {SAMPLE_DIR}")

# Main Content
if video_file_path:
    # Analysis Button Section
    st.subheader("Analysis")
    if st.button("Analyze Risk", type="primary", use_container_width=True):
        with st.spinner("Processing video (this may take a moment)..."):
            try:
                # Send to backend
                # Re-open file for sending
                files = {'file': open(video_file_path, 'rb')}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if "predictions" in data:
                        predictions = data["predictions"]
                        st.success("Analysis Complete!")
                        
                        # Auto-scroll to results
                        # We place the anchor slightly lower and use JS to wait a bit for rendering
                        st.markdown('<div id="results_section"></div>', unsafe_allow_html=True)
                        components.html(
                            """
                            <script>
                                setTimeout(function() {
                                    window.parent.document.getElementById("results_section").scrollIntoView({behavior: "smooth", block: "start"});
                                }, 500);
                            </script>
                            """,
                            height=0
                        )
                        
                        st.divider()
                        st.subheader("Real-time Risk Visualization")
                        
                        # 3. Side-by-Side Visualization (Video Left, Graph Right)
                        viz_col1, viz_col2 = st.columns([1, 1])
                        
                        with viz_col1:
                            st.markdown("### Safe/Risky Video Feed")
                            video_placeholder = st.empty()
                            
                        with viz_col2:
                            st.markdown("### Risk Probability Graph")
                            risk_text_placeholder = st.empty()
                            chart_placeholder = st.empty()
                        
                        # Process video for synchronized playback
                        cap = cv2.VideoCapture(video_file_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        if total_frames > 0 and len(predictions) > 0:
                            # Interpolate predictions to match total frames
                            risk_array = np.interp(
                                np.linspace(0, len(predictions)-1, total_frames),
                                np.arange(len(predictions)),
                                predictions
                            )
                            
                            # Setup Plotly Chart
                            fig = go.Figure()
                            # Start with empty data
                            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Risk', line=dict(color='red', width=2)))
                            fig.update_layout(
                                xaxis=dict(range=[0, total_frames], title="Frame", gridcolor='#333333'),
                                yaxis=dict(range=[0, 1], title="Risk Probability", gridcolor='#333333'),
                                height=400,
                                margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            
                            x_data = []
                            y_data = []
                            
                            frame_idx = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Convert BGR to RGB for Streamlit
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Update Video (Left Column)
                                video_placeholder.image(frame, channels="RGB", use_container_width=True)
                                
                                # Update Data
                                current_risk = risk_array[frame_idx]
                                x_data.append(frame_idx)
                                y_data.append(current_risk)
                                
                                # Update Chart (Right Column) - Update every N frames
                                if frame_idx % 2 == 0:
                                    fig.data[0].x = x_data
                                    fig.data[0].y = y_data
                                    
                                    # Highlight risk text
                                    color = "green"
                                    if current_risk > 0.8: color = "red"
                                    elif current_risk > 0.5: color = "orange"
                                    
                                    risk_text_placeholder.markdown(f"**Current Risk:** <span style='color:{color}; font-size: 24px'>{current_risk:.2f}</span>", unsafe_allow_html=True)
                                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_{frame_idx}")
                                
                                frame_idx += 1
                                
                                # Control Framerate
                                # Ensure at least 5 seconds duration or ~10 FPS (0.1s delay)
                                # If video is short, it will play slower to meet 5s requirement.
                                # If video is long, it will play at ~10 FPS.
                                delay = max(0.1, 5.0 / total_frames) if total_frames > 0 else 0.1
                                time.sleep(delay) 
                                
                            cap.release()
                        else:
                            st.warning("Not enough data to generated visualization.")
                            
                    elif "error" in data:
                        st.error(f"Error from backend: {data['error']}")
                else:
                    st.error(f"Failed to get response: {response.status_code}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
