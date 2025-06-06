import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import pyaudio
import whisper
import numpy as np
import torch
import webrtcvad
import collections
import time
import queue
import webbrowser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ───────────────────────────────────────────────────────────────────────────────
# PyInstaller로 묶인 exe이거나 .py 상태를 구분해서 base_dir 설정
if getattr(sys, "frozen", False):
    # PyInstaller로 묶인 exe가 실행 중인 경우
    base_dir = sys._MEIPASS
else:
    # .py 파일을 직접 실행할 때
    base_dir = os.path.dirname(os.path.abspath(__file__))

# 모델 폴더(whisper, nllb-200-distilled-600M)는 base_dir 아래에 있다고 가정
WHISPER_ROOT = os.path.join(base_dir, "whisper")
NLLB_PATH    = os.path.join(base_dir, "nllb-200-distilled-600M")
# ───────────────────────────────────────────────────────────────────────────────

# Whisper 모델 로드
model = whisper.load_model("medium", download_root=WHISPER_ROOT)

# HF Transformers NLLB 모델 로드
tokenizer  = AutoTokenizer.from_pretrained(NLLB_PATH)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH)
translator = pipeline(
    "translation",
    model=nllb_model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="kor_Hang",
    device=0 if torch.cuda.is_available() else -1
)

FORMAT               = pyaudio.paInt16
CHANNELS             = 2
TARGET_CHANNEL       = 0
RATE                 = 16000
FRAME_DURATION       = 30
FRAME_SIZE           = int(RATE * FRAME_DURATION / 1000)
NUM_PADDING_FRAMES   = 10
VAD_AGGRESSIVENESS   = 2
MAX_SPEECH_DURATION  = 12
vad                  = webrtcvad.Vad(VAD_AGGRESSIVENESS)

translate_queue = queue.Queue()

def translate_en_to_ko(text):
    try:
        result = translator(text, max_length=512)
        return result[0]['translation_text']
    except Exception as e:
        return f"[번역 실패: {e}]"

def clean_output(text):
    sentences = text.strip().split(".")
    seen      = set()
    cleaned   = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return ". ".join(cleaned).strip() + "."

def extract_channel(frame, channel, total_channels):
    audio = np.frombuffer(frame, dtype=np.int16)
    mono  = audio[channel::total_channels].astype(np.float32) / 32768.0
    return mono

def open_main_gui(device_index):
    root = tk.Tk()
    root.title("WEGO")

    chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', height=25)
    chat_window.pack(padx=10, pady=(10, 5), fill=tk.BOTH, expand=True)
    chat_window.tag_configure("left", justify="left")
    chat_window.tag_configure("right", justify="right")

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    status_label      = tk.Label(bottom_frame, text="🟡 대기 중", fg="orange", font=("Arial", 12, "bold"))
    status_label.pack(side=tk.LEFT)

    translating_label = tk.Label(bottom_frame, text="", fg="blue", font=("Arial", 12, "bold"))
    translating_label.pack(side=tk.RIGHT)

    volume_canvas = tk.Canvas(root, height=20, bg="black")
    volume_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))

    def open_blog(event=None):
        webbrowser.open("https://happyweed.tistory.com")

    link_frame = tk.Frame(root)
    link_frame.pack(fill=tk.X, padx=10, pady=(0, 5), anchor="e")
    link_label = tk.Label(
        link_frame,
        text="@made by Weed",
        fg="blue",
        cursor="hand2",
        font=("Arial", 10, "underline")
    )
    link_label.pack(side=tk.RIGHT)
    link_label.bind("<Button-1>", open_blog)

    def set_status(text, color):
        status_label.config(text=text, fg=color)
        root.update_idletasks()

    def set_translating(text):
        translating_label.config(text=text)
        root.update_idletasks()

    def add_message(text, side="left"):
        chat_window.configure(state='normal')
        tag = "right" if side == "right" else "left"
        chat_window.insert(tk.END, text + "\n", tag)
        chat_window.configure(state='disabled')
        chat_window.see(tk.END)

    def get_volume_level(samples: np.ndarray):
        rms = np.sqrt(np.mean(samples ** 2))
        return min(rms, 1.0)

    def draw_volume_bar(volume):
        volume_canvas.delete("all")
        width = int(volume_canvas.winfo_width() * volume)
        volume_canvas.create_rectangle(0, 0, width, 20, fill="lime")
        return volume

    def translation_worker():
        while True:
            raw_audio = translate_queue.get()
            set_translating("🟢 번역 중")
            audio_np    = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)
            result       = model.transcribe(audio_tensor, language="en")
            eng          = result["text"]
            kor          = clean_output(translate_en_to_ko(eng))
            add_message(f"📣 {eng}\n-> {kor}", side="left")
            set_translating("")
            translate_queue.task_done()

    def voice_loop():
        p      = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAME_SIZE
        )
        ring_buffer  = collections.deque(maxlen=NUM_PADDING_FRAMES)
        triggered    = False
        audio_frames = []
        start_time   = None
        blink_state  = [False]

        def blink():
            while True:
                if blink_state[0]:
                    set_status("🔴 듣는 중", "red")
                    time.sleep(0.5)
                    set_status("", "red")
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)

        threading.Thread(target=blink, daemon=True).start()

        try:
            while True:
                frame        = stream.read(FRAME_SIZE, exception_on_overflow=False)
                mono         = extract_channel(frame, TARGET_CHANNEL, CHANNELS)
                int16_frame  = (mono * 32768.0).astype(np.int16).tobytes()
                volume_level = get_volume_level(mono)
                active       = draw_volume_bar(volume_level)

                if active < 0.01:
                    set_status("🟡 대기 중", "orange")

                is_speech = vad.is_speech(int16_frame, RATE)

                if not triggered:
                    if is_speech:
                        blink_state[0] = True
                    ring_buffer.append((int16_frame, is_speech))
                    if sum(s for _, s in ring_buffer) > 0.9 * NUM_PADDING_FRAMES:
                        triggered    = True
                        start_time   = time.time()
                        audio_frames.extend(f for f, _ in ring_buffer)
                        ring_buffer.clear()
                else:
                    audio_frames.append(int16_frame)
                    ring_buffer.append((int16_frame, is_speech))
                    elapsed = time.time() - start_time
                    if (
                        elapsed > MAX_SPEECH_DURATION
                        or sum(not s for _, s in ring_buffer) > 0.9 * NUM_PADDING_FRAMES
                    ):
                        triggered    = False
                        blink_state[0] = False
                        set_status("🟡 대기 중", "orange")
                        raw_audio    = b"".join(audio_frames)
                        audio_frames = []
                        ring_buffer.clear()
                        translate_queue.put(raw_audio)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    threading.Thread(target=voice_loop, daemon=True).start()
    threading.Thread(target=translation_worker, daemon=True).start()
    root.mainloop()

def device_selector():
    selector = tk.Tk()
    selector.title("🎚 오디오 입력 장치 선택")

    label = tk.Label(selector, text="입력 장치를 선택해주세요:", font=("Arial", 12))
    label.pack(pady=10)

    device_listbox = tk.Listbox(selector, width=60, height=10)
    device_listbox.pack(padx=10, pady=10)

    volume_canvas = tk.Canvas(selector, height=20, bg="black")
    volume_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))

    def draw_volume_bar(vol):
        volume_canvas.delete("all")
        width = int(volume_canvas.winfo_width() * vol)
        volume_canvas.create_rectangle(0, 0, width, 20, fill="lime")

    p       = pyaudio.PyAudio()
    devices = [
        (i, p.get_device_info_by_index(i)['name'])
        for i in range(p.get_device_count())
        if p.get_device_info_by_index(i).get('maxInputChannels', 0) > 0
    ]

    for idx, name in devices:
        device_listbox.insert(tk.END, f"{idx}: {name}")

    monitoring = True

    def monitor_loop():
        stream = None
        while monitoring:
            try:
                sel = device_listbox.curselection()
                if sel:
                    device_index = int(device_listbox.get(sel[0]).split(":")[0])
                    if stream:
                        stream.stop_stream()
                        stream.close()
                    stream = p.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=FRAME_SIZE
                    )
                    frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
                    mono  = extract_channel(frame, TARGET_CHANNEL, CHANNELS)
                    vol   = np.sqrt(np.mean(mono ** 2))
                    draw_volume_bar(min(vol, 1.0))
            except:
                pass
            time.sleep(0.1)

    threading.Thread(target=monitor_loop, daemon=True).start()

    def on_select():
        nonlocal monitoring
        sel = device_listbox.curselection()
        if not sel:
            messagebox.showwarning("선택 없음", "입력 장치를 선택해주세요.")
            return
        device_index = int(device_listbox.get(sel[0]).split(":")[0])
        monitoring = False
        selector.destroy()
        open_main_gui(device_index)

    select_btn = tk.Button(selector, text="🎤 시작", font=("Arial", 11), command=on_select)
    select_btn.pack(pady=(0, 10))

    selector.mainloop()

if __name__ == "__main__":
    device_selector()
