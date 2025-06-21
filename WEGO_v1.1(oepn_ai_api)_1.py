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
import gc
from openai import OpenAI  # 최신 방식

model = None
selected_model_name = "medium"
openai_api_key = ""
translate_queue = queue.Queue()
client = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
whisper_root = os.path.join(BASE_DIR, "whisper")

FORMAT = pyaudio.paInt16
CHANNELS = 2
TARGET_CHANNEL = 0
RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
NUM_PADDING_FRAMES = 10
VAD_AGGRESSIVENESS = 2
MAX_SPEECH_DURATION = 12
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def translate_en_to_ko(text):
    global openai_api_key, client
    try:
        if client is None:
            client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate English into Korean."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI 번역 실패: {e}]"

def clean_output(text):
    sentences = text.strip().split(".")
    seen = set()
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return ". ".join(cleaned).strip() + "."

def extract_channel(frame, channel, total_channels):
    audio = np.frombuffer(frame, dtype=np.int16)
    mono = audio[channel::total_channels].astype(np.float32) / 32768.0
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

    status_label = tk.Label(bottom_frame, text="🟡 대기 중", fg="orange", font=("Arial", 12, "bold"))
    status_label.pack(side=tk.LEFT)

    translating_label = tk.Label(bottom_frame, text="", fg="blue", font=("Arial", 12, "bold"))
    translating_label.pack(side=tk.RIGHT)

    volume_canvas = tk.Canvas(root, height=20, bg="black")
    volume_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))

    def open_blog(event=None):
        webbrowser.open("https://happyweed.tistory.com")

    link_frame = tk.Frame(root)
    link_frame.pack(fill=tk.X, padx=10, pady=(0, 5), anchor="e")
    link_label = tk.Label(link_frame, text="@made by Weed", fg="blue", cursor="hand2", font=("Arial", 10, "underline"))
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
            try:
                audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_np)
                result = model.transcribe(audio_tensor, language="en")

                eng = result["text"]
                add_message(f"📣 {eng}", side="left")

                kor = clean_output(translate_en_to_ko(eng))
                add_message(f"🈶 {kor}", side="left")

            except Exception as e:
                add_message(f"[번역 중 오류: {e}]", side="left")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                set_translating("")
                translate_queue.task_done()

    def voice_loop():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAME_SIZE
        )
        ring_buffer = collections.deque(maxlen=NUM_PADDING_FRAMES)
        triggered = False
        audio_frames = []
        start_time = None
        blink_state = [False]

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
                frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
                mono = extract_channel(frame, TARGET_CHANNEL, CHANNELS)
                int16_frame = (mono * 32768.0).astype(np.int16).tobytes()
                volume_level = get_volume_level(mono)
                active = draw_volume_bar(volume_level)

                if active < 0.01:
                    set_status("🟡 대기 중", "orange")

                is_speech = vad.is_speech(int16_frame, RATE)

                if not triggered:
                    if is_speech:
                        blink_state[0] = True
                    ring_buffer.append((int16_frame, is_speech))
                    if sum(s for _, s in ring_buffer) > 0.9 * NUM_PADDING_FRAMES:
                        triggered = True
                        start_time = time.time()
                        audio_frames.extend(f for f, _ in ring_buffer)
                        ring_buffer.clear()
                else:
                    audio_frames.append(int16_frame)
                    ring_buffer.append((int16_frame, is_speech))
                    elapsed = time.time() - start_time
                    if elapsed > MAX_SPEECH_DURATION or sum(not s for _, s in ring_buffer) > 0.9 * NUM_PADDING_FRAMES:
                        triggered = False
                        blink_state[0] = False
                        set_status("🟡 대기 중", "orange")
                        raw_audio = b''.join(audio_frames)
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
    global openai_api_key, model, selected_model_name

    selector = tk.Tk()
    selector.title("🎚 오디오 장치 & Whisper 모델 선택")

    tk.Label(selector, text="입력 장치를 선택해주세요:", font=("Arial", 12)).pack(pady=10)

    device_listbox = tk.Listbox(selector, width=60, height=10)
    device_listbox.pack(padx=10, pady=10)

    volume_canvas = tk.Canvas(selector, height=20, bg="black")
    volume_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))

    tk.Label(selector, text="🔑 OpenAI API 키 입력:", font=("Arial", 10)).pack()
    api_entry = tk.Entry(selector, width=50)
    api_entry.pack(pady=(0, 10))

    tk.Label(selector, text="🧠 Whisper 모델 선택:", font=("Arial", 10)).pack()
    model_var = tk.StringVar(selector)
    model_var.set("medium")
    model_menu = tk.OptionMenu(selector, model_var, "tiny", "base", "small", "medium", "large")
    model_menu.config(width=20)
    model_menu.pack(pady=(0, 10))

    def draw_volume_bar(vol):
        volume_canvas.delete("all")
        width = int(volume_canvas.winfo_width() * vol)
        volume_canvas.create_rectangle(0, 0, width, 20, fill="lime")

    p = pyaudio.PyAudio()
    devices = [(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())
               if p.get_device_info_by_index(i).get('maxInputChannels', 0) > 0]
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
                    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                    input_device_index=device_index, frames_per_buffer=FRAME_SIZE)
                    frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
                    mono = extract_channel(frame, TARGET_CHANNEL, CHANNELS)
                    vol = np.sqrt(np.mean(mono ** 2))
                    draw_volume_bar(min(vol, 1.0))
            except:
                pass
            time.sleep(0.1)

    threading.Thread(target=monitor_loop, daemon=True).start()

    def on_select():
        nonlocal monitoring
        global model, openai_api_key
        sel = device_listbox.curselection()
        if not sel:
            messagebox.showwarning("선택 없음", "입력 장치를 선택해주세요.")
            return
        key = api_entry.get().strip()
        if not key:
            messagebox.showwarning("API 키 누락", "OpenAI API 키를 입력해주세요.")
            return

        selected_model = model_var.get()

        try:
            selector.title(f"⏳ 모델 로딩 중: {selected_model}")
            if model is not None:
                del model
                model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            model = whisper.load_model(selected_model, download_root=whisper_root)
        except Exception as e:
            messagebox.showerror("모델 로딩 실패", f"Whisper 모델 로딩 실패:\n{e}")
            return

        openai_api_key = key
        device_index = int(device_listbox.get(sel[0]).split(":")[0])
        monitoring = False
        selector.destroy()
        open_main_gui(device_index)

    tk.Button(selector, text="🎤 시작", font=("Arial", 11), command=on_select).pack(pady=(0, 10))
    selector.mainloop()

if __name__ == "__main__":
    device_selector()
