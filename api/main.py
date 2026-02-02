import logging
import base64
import tempfile
import os
import signal

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel

import torch
import librosa
import numpy as np
import yaml
from dotenv import load_dotenv
from rawnet.model import RawNet
from fastapi.security import APIKeyHeader

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("VOICE_API_KEY")
logger.info("API key loaded")

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
        logger.warning("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(
    title="AI Voice Detection API (RawNet2 - ASVspoof)",
    docs_url=None,        # prevent Swagger timeout on Render
    redoc_url=None,
    openapi_url="/openapi.json"
)

# ------------------------------------------------------------------
# MODEL LOAD (ONCE)
# ------------------------------------------------------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    logger.info("Loading RawNet2 model...")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    m = RawNet(cfg["model"], DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt)
    m.eval()
    model = m
    logger.info("✅ RawNet2 Model Loaded")

@app.get("/")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------------------------
class VoiceDetectRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ------------------------------------------------------------------
# TIMEOUT HANDLER
# ------------------------------------------------------------------
class InferenceTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise InferenceTimeout()

signal.signal(signal.SIGALRM, _timeout_handler)

# ------------------------------------------------------------------
# AUDIO + PREDICTION
# ------------------------------------------------------------------
def load_audio(path: str):
    # librosa can hang → keep it minimal
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if len(wav) < 64000:
        wav = np.pad(wav, (0, 64000 - len(wav)))
    return torch.tensor(wav).float().unsqueeze(0)

def predict(path: str):
    audio = load_audio(path)
    with torch.no_grad():
        out = model(audio)
        prob_spoof = torch.softmax(out, dim=1)[0][0].item()

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
    request: Request,
    _: bool = Depends(verify_auth_key)
):
    logger.info("Received voice detection request")

    # ---- HARD SIZE LIMIT (Render-safe)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1 * 1024 * 1024:
        raise HTTPException(413, "Audio file too large (max 1MB)")

    # ---- VALIDATION
    language = payload.language.lower()
    audio_format = payload.audioFormat.lower()
    audio_b64 = payload.audioBase64

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, "Unsupported language")

    if audio_format != "mp3":
        raise HTTPException(400, "Only mp3 audio format supported")

    # ---- EARLY BASE64 GUARD
    if len(audio_b64) < 5000:
        raise HTTPException(400, "Invalid or empty audio payload")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        # ---- HARD INFERENCE TIMEOUT (15s)
        signal.alarm(15)
        classification, confidence, risk = predict(audio_path)
    except InferenceTimeout:
        logger.error("Inference timeout")
        raise HTTPException(504, "Inference timeout")
    finally:
        signal.alarm(0)
        os.remove(audio_path)

    logger.info("Inference complete")

    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": confidence
    }
