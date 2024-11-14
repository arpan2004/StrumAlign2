from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io.wavfile import read as read_wav
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
import tempfile
import os

app = Flask(__name__)
CORS(app)

class CNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):    
        out = self.conv(x)
        out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, dims, num_classes = 2):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(1, len(dims)):
            self.layers.append(CNNBlock(input_dim = dims[i - 1], output_dim = dims[i]))
        
        self.gap = nn.AvgPool2d(kernel_size = (25, 18))
        self.fc = nn.Linear(dims[-1], num_classes)
        
        
    def forward(self, x):     
        for layer in self.layers:
            x = layer(x)
        out = self.gap(x)
        shape = out.shape
        out = out.reshape(shape[0], shape[1])
        out = self.fc(out)
        return out    

model = CNN([1, 32, 64, 128])
model.load_state_dict(torch.load('guitar_chord_model.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def record_audio(duration=2, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    return audio.flatten()

def audio_to_spectrogram(audio, sample_rate=16000):
    # Convert audio to tensor for torchaudio processing
    waveform = torch.tensor(audio)
    # Convert to spectrogram
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    # Add channel dimension to match model input
    return specgram.unsqueeze(0)

def save_audio_to_wav(audio, sample_rate, filename="recorded_audio.wav"):
    # Convert audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        sf.write(temp_wav.name, audio, sample_rate)
        return temp_wav.name

#@app.route('/predict')
def predict_live_chord():
    # Step 1: Record Audio
    audio = record_audio(duration=2, sample_rate=16000)
    
    # Step 2: Convert to Spectrogram
    specgram = audio_to_spectrogram(audio, sample_rate=16000)
    specgram = specgram.unsqueeze(0)
    specgram = specgram.to(device).float()  # Send to device, e.g., CPU or GPU

    # Step 3: Model Prediction
    with torch.no_grad():
        output = model(specgram)
        probabilities = F.softmax(output, dim=1)
        
        # Get the predicted class (index with highest probability)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get the probability of the predicted class
        predicted_prob = probabilities[0][predicted_class].item()

    # Step 4: Interpret and Print Result
    chord_type = "Major" if predicted_class == 0 else "Minor"
    audio_file_path = save_audio_to_wav(audio, 16000)
    print(f"Predicted Chord: {chord_type} (Confidence: {predicted_prob:.2f})")
    return jsonify({
        'chord': chord_type,
        'audio_url': audio_file_path
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)