import sounddevice as sd
import numpy as np
import librosa
import random
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from tensorflow.keras.models import load_model
from typing import List
import threading
from fastapi.responses import JSONResponse



loaded_model = load_model("pitch_detection_cnn.keras")



app = FastAPI()
main_loop = asyncio.get_event_loop()

pitch_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
current_note_index = 0
notes_list = random.choices(pitch_names, k=2000)

@app.get("/notes")
async def get_notes():
    return JSONResponse(content={"notes": notes_list})

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
clients: List[WebSocket] = []
recording_thread = None
recording_active = threading.Event()

def start_recording():
    recording_active.set()
    with sd.InputStream(channels=1, callback=callback, samplerate=fs, blocksize=chunk_size):
        print("Recording started.")
        while recording_active.is_set():
            sd.sleep(100)

def stop_recording():
    recording_active.clear()
    print("Recording stopped.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global recording_thread
    await websocket.accept()
    clients.append(websocket)
    if recording_thread is None or not recording_thread.is_alive():
        recording_thread = threading.Thread(target=start_recording, daemon=True)
        recording_thread.start()
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        clients.remove(websocket)
        if not clients:  # No more clients, stop recording
            stop_recording()

def send_prediction(prediction: str):
    print(f"Sending to clients: {prediction}")
    for ws in clients:
        asyncio.run_coroutine_threadsafe(ws.send_text(str(prediction)), main_loop)

notes_window = 0.3

fs = 44100
max_frames = 40
chunk_size = int(notes_window * fs)
threshold = 0.002 # 0.02

buffer = np.zeros(int(0.3 * fs))
hit_number = 0  # global counter




# Define pitch classes (C, C#, D, ..., B)
pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
               'F#', 'G', 'G#', 'A', 'A#', 'B']

ignore_samples = int(notes_window * fs)  # ignore next 0.5 sec after a hit
samples_since_last_hit = ignore_samples  # initialize above threshold



# Game state
current_target = random.choice(pitch_names)
score = 0

print(f"Play this note: {current_target}")

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

    send_prediction(predicted_pitch)

    # Play snippet for verification
    #sd.play(snippet, fs)


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

def callback(indata, frames, time, status):
    global buffer, hit_number, samples_since_last_hit, was_above_threshold, collecting_post, post_audio, pre_snippet, post_needed
    audio_chunk = indata[:, 0]  # mono
    samples_since_last_hit += len(audio_chunk)
    current_above_threshold = np.max(np.abs(audio_chunk)) > threshold

    

    if collecting_post:
        post_audio.append(audio_chunk)
        total_post = np.concatenate(post_audio)
        if len(total_post) >= post_needed:
            post = total_post[:post_needed]
            # Combine pre and post
            if pre_snippet is not None and post is not None and pre_snippet.ndim == 1 and post.ndim == 1:
                snippet = np.concatenate([pre_snippet, post])
                predict(snippet)
                snippet = snippet / (np.max(np.abs(snippet)) + 1e-6)
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
        return
    
    if np.max(np.abs(audio_chunk)) > threshold and samples_since_last_hit >= ignore_samples and current_above_threshold and not was_above_threshold:
        
        print(f"Hit detected! Total hits: {hit_number}")

        # Find first sample above threshold
        above_thresh = np.where(np.abs(audio_chunk) > threshold)[0]
        if len(above_thresh) == 0:
            return
        start_idx = above_thresh[0]

        print(len(above_thresh))

        # Include pre-trigger
        pre_trigger_samples = int(pre_trigger_sec * fs)
        start_idx = max(0, start_idx - pre_trigger_samples)
        trimmed = audio_chunk[start_idx:]

        # Take exactly 0.25 s for pre-trigger
        snippet_length = int(0.25 * fs)
        if len(trimmed) >= snippet_length:
            pre_snippet = trimmed[:snippet_length]
        else:
            pad_width = snippet_length - len(trimmed)
            pre_snippet = np.pad(trimmed, (0, pad_width), mode='constant')

        # Start collecting post-trigger audio
        collecting_post = True
        post_audio = []
        # Save how many more samples we need
        post_needed = snippet_length - len(above_thresh)

    # Update circular buffer
    buffer = np.roll(buffer, -len(audio_chunk))
    buffer[-len(audio_chunk):] = audio_chunk
    was_above_threshold = current_above_threshold
    
def start_recording():
    with sd.InputStream(channels=1, callback=callback, samplerate=fs, blocksize=chunk_size):
        print("Recording... Press Ctrl+C to stop")
        while True:
            sd.sleep(1000)
