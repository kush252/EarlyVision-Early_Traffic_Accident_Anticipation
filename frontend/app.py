import streamlit as st
import json
import streamlit.components.v1 as components
import requests
import os
import tempfile
import cv2
import time
import numpy as np
import plotly.graph_objects as go

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Accident Risk Prediction", layout="wide")

st.title("Accident Risk Prediction Dashboard")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        /* --- Animations --- */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes pulseRed {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }

        /* --- Global Font and Color --- */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }
        
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #ffffff;
            animation: fadeIn 1.2s ease-out;
        }

        /* --- Black Theme with Subtle White Blur --- */
        .stApp {
            background-color: #000000;
            background-image: 
                radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.08) 0%, transparent 70%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* White Text for Content */
        .stMarkdown, .stText, p, label, div {
            color: #ffffff !important;
        }
        
        /* Title (h1) Styling */
        h1 {
            background: linear-gradient(90deg, #ffffff, #ff4444);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            color: #ff0000 !important; /* Fallback */
            text-shadow: 0px 0px 15px rgba(255, 0, 0, 0.2);
            margin-bottom: 1rem;
        }
        
        /* Other headers White */
        h2, h3 {
            color: #ffffff !important;
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: rgba(10, 10, 10, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 50, 50, 0.15);
            transition: transform 0.3s ease;
        }
        
        /* --- Smooth Content Entry --- */
        div.block-container {
            animation: slideUp 0.8s ease-out;
        }

        /* --- Primary Button (Red) Interactions --- */
        div.stButton > button:first-child {
            background-color: #e60000;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        div.stButton > button:first-child:hover {
            background-color: #ff3333;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
            transform: translateY(-2px) scale(1.01);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        div.stButton > button:first-child:active {
            transform: translateY(0);
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
        }

        /* --- File Uploader Styling --- */
        [data-testid="stFileUploader"] {
            border: 1px dashed rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #ff4444;
            background-color: rgba(255, 0, 0, 0.05);
        }

        /* --- Progress Bar Glow --- */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, #e60000, #ff4444);
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.6);
            transition: width 0.5s ease-in-out;
        }
        
        /* --- Card/Image Hover Effects --- */
        [data-testid="stImage"] img {
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        [data-testid="stImage"] img:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 40px rgba(0,0,0,0.8);
        }
        
        /* --- Warnings/Success Messages --- */
        .stAlert {
            border-radius: 10px;
            backdrop-filter: blur(10px);
            animation: fadeIn 0.5s ease;
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

st.sidebar.header("Input Selection")
input_option = st.sidebar.radio("Choose Input Source", ("Upload Video", "Use Sample Video"))

video_file_path = None

if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video (mp4, avi)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_file_path = tfile.name

elif input_option == "Use Sample Video":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample data")
    
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith('.mp4')]
        selected_sample = st.selectbox("Select a sample video", files)
        if selected_sample:
            video_file_path = os.path.join(SAMPLE_DIR, selected_sample)
    else:
        st.error(f"Sample directory not found: {SAMPLE_DIR}")

if video_file_path:
    st.subheader("Analysis")
    if st.button("Analyze Risk", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            files = {'file': open(video_file_path, 'rb')}
            
            with requests.post(API_URL, files=files, stream=True) as response:
                
                if response.status_code == 200:
                    predictions = []
                    error_msg = None
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                update = json.loads(line)
                                if "progress" in update:
                                    progress_bar.progress(min(update["progress"], 100))
                                if "status" in update:
                                    status_text.markdown(f"**{update['status']}**")
                                if "predictions" in update:
                                    predictions = update["predictions"]
                                if "error" in update:
                                    error_msg = update["error"]
                            except json.JSONDecodeError:
                                continue
                    
                    if error_msg:
                        progress_bar.empty()
                        status_text.empty()
                        if "Inappropriate content" in error_msg or "Validation Failed" in error_msg:
                            st.warning(f"⚠️ **Video Content Check Failed**\n\n{error_msg}\n\nPlease verify you used a dashcam video as per instructions.")
                        else:
                            st.error(f"Error from backend: {error_msg}")
                    elif predictions:
                        progress_bar.progress(100)
                        status_text.success("Analysis Complete!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
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
                        
                        viz_col1, viz_col2 = st.columns([1, 1])
                        
                        with viz_col1:
                            st.markdown("### Safe/Risky Video Feed")
                            video_placeholder = st.empty()
                            
                        with viz_col2:
                            st.markdown("### Risk Probability Graph")
                            risk_text_placeholder = st.empty()
                            chart_placeholder = st.empty()
                        
                        cap = cv2.VideoCapture(video_file_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        if total_frames > 0 and len(predictions) > 0:
                            risk_array = np.interp(
                                np.linspace(0, len(predictions)-1, total_frames),
                                np.arange(len(predictions)),
                                predictions
                            )
                            
                            fig = go.Figure()
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
                                
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                video_placeholder.image(frame, channels="RGB", use_container_width=True)
                                
                                current_risk = risk_array[frame_idx]
                                x_data.append(frame_idx)
                                y_data.append(current_risk)
                                
                                if frame_idx % 2 == 0:
                                    fig.data[0].x = x_data
                                    fig.data[0].y = y_data
                                    
                                    color = "green"
                                    if current_risk > 0.8: color = "red"
                                    elif current_risk > 0.5: color = "orange"
                                    
                                    risk_text_placeholder.markdown(f"**Current Risk:** <span style='color:{color}; font-size: 24px'>{current_risk:.2f}</span>", unsafe_allow_html=True)
                                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_{frame_idx}")
                                
                                frame_idx += 1
                                
                                delay = max(0.1, 5.0 / total_frames) if total_frames > 0 else 0.1
                                time.sleep(delay) 
                                
                            cap.release()
                        else:
                            st.warning("Not enough data to generated visualization.")
                            
                    elif "error" in data:
                        error_msg = data['error']
                        if "Inappropriate content" in error_msg or "Validation Failed" in error_msg:
                            st.warning(f"⚠️ **Video Content Check Failed**\n\n{error_msg}\n\nPlease verify you used a dashcam video as per instructions.")
                        else:
                            st.error(f"Error from backend: {error_msg}")
                elif response.status_code == 422:
                     try:
                         data = response.json()
                         err_detail = data.get('detail', "Invalid Video Content")
                     except:
                         err_detail = "Invalid Video Content"
                         
                     progress_bar.empty()
                     status_text.empty()
                     st.warning(f"⚠️ **Inappropriate Content Detected**\n\n{err_detail}")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    try:
                        err_detail = response.json().get('detail', str(response.status_code))
                    except:
                        err_detail = str(response.status_code)
                    
                    if "Inappropriate content" in str(err_detail) or "422" in str(err_detail):
                         st.warning("⚠️ **Upload Dashcam Video**\n\nNo traffic content found in your video (e.g., no cars, roads, or signs found).")
                    else:
                         st.error(f"Failed to get response: {err_detail}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
