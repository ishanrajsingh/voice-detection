import torch
import librosa
import numpy as np
from model import RawNet
import yaml

DEVICE = "cpu"

with open("model_config_RawNet.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = RawNet(cfg["model"], DEVICE)
ckpt = torch.load("pre_trained_DF_RawNet2.pth", map_location=DEVICE)

model.load_state_dict(ckpt)
model.eval()

print("✅ Model Loaded")

def load_audio(path):
    wav, sr = librosa.load(path, sr=16000)
    if len(wav) < 64000:
        wav = np.pad(wav, (0, 64000-len(wav)))
    return torch.tensor(wav).float().unsqueeze(0)

def predict(path):
    audio = load_audio(path)
    with torch.no_grad():
        out = model(audio)
        prob = torch.softmax(out, dim=1)[0][0].item()  # CLASS 0 = SPOOF

    label = "AI_GENERATED" if prob > 0.5 else "HUMAN"
    return label, round(prob, 4)

if __name__ == "__main__":
    file = "audios/audio_4.mp3"
    label, score = predict(file)

    print("Prediction:", label)
    print("Confidence:", score)
