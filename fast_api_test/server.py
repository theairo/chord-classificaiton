from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import uuid, os, subprocess, librosa
import tempfile
import random
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pitch_detection_cnn.keras")
loaded_model = load_model(MODEL_PATH)



temp_dir = "C:\\Temp"
os.makedirs(temp_dir, exist_ok=True)

app = FastAPI()

fs = 44100  # sampling rate

notes_window = 0.3

max_frames = 40
chunk_size = int(notes_window * fs)
threshold = 0.05 # 0.02

buffer = np.zeros(int(0.3 * fs))
hit_number = 0  # global counter

ignore_samples = int(notes_window * fs)  # ignore next 0.5 sec after a hit
samples_since_last_hit = ignore_samples  # initialize above threshold

score = 0


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your ngrok URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(__file__)  # folder containing server.py
STATIC_DIR = os.path.join(BASE_DIR, "static")
PAGE_PATH = os.path.join(BASE_DIR, "static", "page.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/")
async def get_page():
    # redirect to static page
    with open(PAGE_PATH, encoding="utf-8") as f:
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
    # 1. We only do this process if "was_above_threshold" is True 
    # 2. We need to finish the previous note, so we get the needed post_snippet amount from the chunk and predict the note as usual. 
    # 3. Then we cut this from the chunk and focus on what's left. 
    # 4. We split the the rest of the chunk into 0.03s frames and find mean amplitude for each. We iterate from left to right finding the first moment where the monotonity of amplitude changes (that is mean_i < mean_(i+1). 
    # 5. We cut the previous frames up to this moment and feed the rest of the chunk to the process() function.
    global buffer, hit_number, samples_since_last_hit, was_above_threshold
    global collecting_post, post_audio, pre_snippet, post_needed, prediction

    samples_since_last_hit += len(audio_chunk)
    current_above_threshold = np.max(np.abs(audio_chunk)) > threshold

    predicted = False
    remaining_chunk = audio_chunk.copy()

    # Step 1: Finish previous note if collecting_post
    if collecting_post:
        post_audio.append(audio_chunk)
        total_post = np.concatenate(post_audio)
        if len(total_post) >= post_needed:
            post = total_post[:post_needed]
            if pre_snippet is not None and post.ndim == 1:
                snippet = np.concatenate([pre_snippet, post])
                snippet = snippet / (np.max(np.abs(snippet)) + 1e-6)
                prediction = predict(snippet)
                predicted = True
                print(f"Predicted snippet length: {len(snippet)}")
            else:
                print("Warning: pre_snippet or post is invalid!", type(pre_snippet), type(post))

            hit_number += 1
            samples_since_last_hit = 0
            collecting_post = False
            post_audio = []
            pre_snippet = None

            # Remove used samples from remaining chunk
            remaining_chunk = total_post[post_needed:]

    # Step 2: Split remaining chunk into frames for monotonicity check
    frame_len = int(0.03 * fs)  # 30 ms frames
    num_frames = len(remaining_chunk) // frame_len
    if num_frames > 1:
        frame_means = [np.mean(np.abs(remaining_chunk[i*frame_len:(i+1)*frame_len])) 
                       for i in range(num_frames)]

        # Find first moment where mean amplitude stops decreasing
        cut_frame = 0
        for i in range(1, len(frame_means)):
            if frame_means[i] > frame_means[i-1]:
                cut_frame = i
                break

        # Remaining chunk after this point
        remaining_chunk = remaining_chunk[cut_frame*frame_len:]

    # Step 3: Detect new hit in the remaining chunk
    if (np.max(np.abs(remaining_chunk)) > threshold and 
        samples_since_last_hit >= ignore_samples and 
        current_above_threshold):

        print(f"Hit detected! Total hits: {hit_number}")

        above_thresh = np.where(np.abs(remaining_chunk) > threshold)[0]
        if len(above_thresh) > 0:
            start_idx = max(0, above_thresh[0] - int(pre_trigger_sec * fs))
            pre_snippet = remaining_chunk[start_idx:]
            snippet_length = int(0.25 * fs)
            if len(pre_snippet) > snippet_length:
                pre_snippet = pre_snippet[:snippet_length]

            collecting_post = True
            post_audio = []
            post_needed = max(0, snippet_length - len(pre_snippet))
            print(f"Post needed: {post_needed}, pre_snippet length: {len(pre_snippet)}")

    # Step 4: Update buffer
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



