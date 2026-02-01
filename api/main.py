import logging
import base64
import tempfile
import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

import torch
import librosa
import numpy as np
import yaml

from rawnet.model import RawNet
from fastapi.security import APIKeyHeader

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

API_KEY = "sk_test_123456789"

SUPPORTED_LANGUAGES = {"tamil", "english", "hindi", "malayalam", "telugu"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "rawnet", "pre_trained_DF_RawNet2.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "rawnet", "model_config_RawNet.yaml")

DEVICE = "cpu"

SPOOF_HIGH_THRESHOLD = 0.85
SPOOF_MEDIUM_THRESHOLD = 0.65

# ------------------------------------------------------------------
# AUTH
# ------------------------------------------------------------------

AUTH_KEY_NAME = "x-api-key"
auth_key_header = APIKeyHeader(name=AUTH_KEY_NAME, auto_error=False)

async def verify_auth_key(auth_key: str = Depends(auth_key_header)):
    if auth_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return True

# ------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

model = RawNet(cfg["model"], DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()

print("RawNet2 Model Loaded")

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------

app = FastAPI(title="AI Voice Detection API (RawNet2 - ASVspoof)")

# ------------------------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------------------------

class VoiceDetectRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ------------------------------------------------------------------
# AUDIO + PREDICTION
# ------------------------------------------------------------------

def load_audio(path):
    wav, sr = librosa.load(path, sr=16000)
    if len(wav) < 64000:
        wav = np.pad(wav, (0, 64000 - len(wav)))
    return torch.tensor(wav).float().unsqueeze(0)

def predict(path):
    audio = load_audio(path)

    with torch.no_grad():
        out = model(audio)
        prob_spoof = torch.softmax(out, dim=1)[0][0].item()  # class 0 = spoof

    if prob_spoof >= SPOOF_HIGH_THRESHOLD:
        label = "AI_GENERATED"
        risk = "high"
    elif prob_spoof >= SPOOF_MEDIUM_THRESHOLD:
        label = "SUSPECTED_AI"
        risk = "medium"
    else:
        label = "HUMAN"
        risk = "low"

    return label, round(prob_spoof, 4), risk

# ------------------------------------------------------------------
# API ENDPOINT
# ------------------------------------------------------------------

@app.post("/api/voice-detection")
async def detect_voice(
    payload: VoiceDetectRequest,
    _: bool = Depends(verify_auth_key)
):
    language = payload.language.lower()
    audio_format = payload.audioFormat.lower()
    audio_b64 = payload.audioBase64

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if audio_format != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 audio format supported")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        classification, confidence, risk = predict(audio_path)
    finally:
        os.remove(audio_path)

    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": confidence,
        
    }
