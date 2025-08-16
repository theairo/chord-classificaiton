from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import uuid, os, subprocess, librosa
import tempfile
import random
import numpy as np


loaded_model = load_model("pitch_detection_cnn.keras")



temp_dir = "C:\\Temp"
os.makedirs(temp_dir, exist_ok=True)

app = FastAPI()

fs = 44100  # sampling rate

notes_window = 0.3

max_frames = 40
chunk_size = int(notes_window * fs)
threshold = 0.2 # 0.02

buffer = np.zeros(int(0.3 * fs))
hit_number = 0  # global counter

ignore_samples = int(notes_window * fs)  # ignore next 0.5 sec after a hit
samples_since_last_hit = ignore_samples  # initialize above threshold

score = 0


# Allow CORS for all origins (for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



# Serve static files under /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def get_page():
    # redirect to static page
    with open("static/page.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

import numpy as np
pre_trigger_sec = 0.03  # 50 ms

# Check current state

was_above_threshold = False

post_trigger_sec = 0.1  # seconds after hit
post_trigger_samples = int(post_trigger_sec * fs)
collecting_post = False
post_audio = []
pre_snippet = None
post_needed = 0
prediction = ""

def callback(audio_chunk):
    global buffer, hit_number, samples_since_last_hit, was_above_threshold, collecting_post, post_audio, pre_snippet, post_needed, prediction
    samples_since_last_hit += len(audio_chunk)
    current_above_threshold = np.max(np.abs(audio_chunk)) > threshold

    predicted = False

    if collecting_post:
        post_audio.append(audio_chunk)
        total_post = np.concatenate(post_audio)
        if len(total_post) >= post_needed:
            post = total_post[:post_needed]
            # Combine pre and post
            if pre_snippet is not None and post is not None and pre_snippet.ndim == 1 and post.ndim == 1:
                snippet = np.concatenate([pre_snippet, post])
                snippet = snippet / (np.max(np.abs(snippet)) + 1e-6)
                prediction = predict(snippet)
                predicted = True
                print(len(snippet))
            else:
                print("Warning: pre_snippet or post is invalid!", type(pre_snippet), type(post))
            hit_number += 1
            samples_since_last_hit = 0
            collecting_post = False
            post_audio = []
            pre_snippet = None
        # Always update buffer
        buffer = np.roll(buffer, -len(audio_chunk))
        buffer[-len(audio_chunk):] = audio_chunk
        was_above_threshold = current_above_threshold
        return predicted
    
    # if np.max(np.abs(audio_chunk)) > threshold and samples_since_last_hit >= ignore_samples and current_above_threshold and not was_above_threshold:
        

    if np.max(np.abs(audio_chunk)) > threshold and samples_since_last_hit >= ignore_samples and current_above_threshold:
        
        print(f"Hit detected! Total hits: {hit_number}")

        above_thresh = np.where(np.abs(audio_chunk) > threshold)[0]
        if len(above_thresh) == 0:
            return
        start_idx = above_thresh[0]

        # Optional: keep a short pre-trigger window (e.g., 50 ms)
        pre_trigger_samples = int(pre_trigger_sec * fs)
        start_idx = max(0, start_idx - pre_trigger_samples)

        # Trim from here onward
        pre_snippet = audio_chunk[start_idx:]

        # Cap length to 0.25 s
        snippet_length = int(0.25 * fs)
        if len(pre_snippet) > snippet_length:
            pre_snippet = pre_snippet[:snippet_length]

        # Start collecting post-trigger audio
        collecting_post = True
        post_audio = []

        # Save how many more samples we need (for post-trigger)
        post_needed = max(0, snippet_length - len(pre_snippet))
        print(post_needed, len(pre_snippet))

    # Update circular buffer
    buffer = np.roll(buffer, -len(audio_chunk))
    buffer[-len(audio_chunk):] = audio_chunk
    was_above_threshold = current_above_threshold

    return predicted


@app.post("/chunk")
async def receive_chunk(chunk: UploadFile = File(...)):
    import numpy as np
    import librosa
    import tempfile
    import os

    predicted = False

    # Save uploaded chunk to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await chunk.read())
        temp_path = tmp.name

    try:
        # Load audio as mono
        y, sr = librosa.load(temp_path, sr=44100, mono=True)



        # Call your callback on the chunk
        predicted = callback(y)
        print(predicted)

        # Compute some stats to return
        amplitude = float(np.mean(np.abs(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    finally:
        os.remove(temp_path)

    return {
        "amplitude": amplitude,
        "rms": rms,
        "zcr": zcr,
        "sr": sr,
        "duration": round(len(y) / sr, 3),
        "prediction": prediction if predicted else "No prediction"
    }

pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
               'F#', 'G', 'G#', 'A', 'A#', 'B']
current_note_index = 0
notes_list = random.choices(pitch_names, k=2000)

@app.get("/notes")
async def get_notes():
    notes_list = random.choices(["C", "D", "E", "F", "G", "A", "B"], k=2000)
    return JSONResponse(content={"notes": notes_list})
    clients.remove(websocket)
    if not clients:  # No more clients, stop recording
        stop_recording()


def predict(snippet):
    global current_note_index
    global buffer  # only if you really need to use or modify it
    global current_target, score

    # Chroma CQT feature
    chroma = librosa.feature.chroma_cqt(y=snippet, sr=fs, bins_per_octave=36)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-6)

    # Pad or truncate chroma to fixed length
    if chroma.shape[1] < max_frames:
        pad_width = max_frames - chroma.shape[1]
        chroma = np.pad(chroma, ((0, 0), (0, pad_width)), mode='constant')
    else:
        chroma = chroma[:, :max_frames]

    chroma_input = np.expand_dims(chroma, axis=0)  # shape: (1, 12, max_frames)

    prediction = loaded_model.predict(chroma_input)
    predicted_class = np.argmax(prediction)
    predicted_pitch = pitch_names[(predicted_class + 9) % 12]

    print("Predicted pitch:", predicted_pitch)

    current_note_index += 1
    if current_note_index < len(notes_list):
        current_target = notes_list[current_note_index]
        print(f"Next note: {current_target}")
    else:
        print("End of notes list reached.")

    return predicted_pitch

    # Play snippet for verification
    #sd.play(snippet, fs)



