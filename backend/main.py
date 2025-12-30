from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import joblib
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import os
import threading
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse  # <--- Add this import

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Allow React to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
camera = cv2.VideoCapture(0)
lock = threading.Lock()
current_frame = None

# --- LOAD MODELS (Same as before) ---
model = joblib.load('lgbm_hazard_model.pkl')
with open('model_artifacts.json', 'r') as f:
    artifacts = json.load(f)

MODEL_FEATURES = artifacts['features']
CAT_MAPPINGS = artifacts['cat_mappings']
NUM_MEDIANS = artifacts['num_medians']

yolo = YOLO('yolov8n.pt') 
gemini_flash = genai.GenerativeModel('gemini-1.5-flash')

# --- HELPER FUNCTIONS ---
def prepare_features(gemini_json):
    """(Copy the prepare_features_for_model function from previous step here)"""
    # For brevity, I'm omitting the full logic, but PASTE IT HERE from previous response
    # ... logic to convert JSON to DataFrame ...
    input_data = {feat: NUM_MEDIANS.get(feat, 0) for feat in MODEL_FEATURES}
    # ... mapping logic ...
    # (Use the code from the previous response for this function)
    df = pd.DataFrame([input_data])
    for col, cats in CAT_MAPPINGS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=cats)
    return df

# --- VIDEO STREAM GENERATOR ---
def generate_frames():
    global current_frame
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Run YOLO for visual feedback
        results = yolo(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        with lock:
            current_frame = frame.copy() # Save raw frame for AI analysis
            
        # Encode for browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ENDPOINTS ---

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/analyze")
def analyze_scene():
    global current_frame
    with lock:
        if current_frame is None: return {"error": "No frame"}
        frame_to_analyze = current_frame.copy()

    # 1. Gemini Vision
    img_rgb = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    prompt = f"""
    Analyze this road scene. Output JSON ONLY with these keys matching options:
    - "Weather": {str(CAT_MAPPINGS.get('Weather', []))[:100]}...
    - "RoadType": {str(CAT_MAPPINGS.get('RoadType', []))[:100]}...
    - "Lighting": {str(CAT_MAPPINGS.get('Light', []))[:100]}...
    """
    
    try:
        response = gemini_flash.generate_content([prompt, pil_img])
        text = response.text.replace("```json", "").replace("```", "")
        scene_data = json.loads(text)
        
        # 2. LightGBM
        input_df = prepare_features(scene_data)
        risk_prob = float(model.predict_proba(input_df)[0][1]) # Convert to python float
        
        return {
            "risk_score": risk_prob,
            "scene_details": scene_data
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)