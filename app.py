from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import io
import os
import uuid

import ollama
import pyttsx3
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

# ----------------------------
# CONFIG
# ----------------------------
VISION_MODEL = "llava"
WHISPER_MODEL_SIZE = "small"  # tiny / base / small / medium
AUDIO_OUTPUT_PATH = "static/output_audio.wav"

# ----------------------------
# Lazy-load Whisper (IMPORTANT)
# ----------------------------
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print(f"[INFO] Loading Whisper model: {WHISPER_MODEL_SIZE} ...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
        print("[INFO] Whisper model loaded successfully ✅")
    return whisper_model


@app.get("/")
def home():
    return render_template("index.html")


# ----------------------------
# 1) IMAGE -> DESCRIPTION (Ollama LLaVA)
# ----------------------------
@app.post("/api/image-caption")
def image_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    mode = request.form.get("mode", "detailed")
    image_file = request.files["image"]

    try:
        img = Image.open(image_file).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    if mode == "alt":
        prompt = "Generate a single-line alt-text for this image for a visually impaired user."
    elif mode == "safety":
        prompt = "Describe the image focusing on safety risks and obstacles for a visually impaired user."
    else:
        prompt = "Describe the image clearly for a visually impaired user. Mention objects, actions, background, and any visible text."

    try:
        res = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes]
                }
            ]
        )

        return jsonify({
            "mode": mode,
            "description": res["message"]["content"]
        })

    except Exception as e:
        return jsonify({"error": f"Ollama failed: {str(e)}"}), 500


# ----------------------------
# 2) SPEECH -> TEXT (+ Translation using Deep Translator)
# ----------------------------
@app.post("/api/speech-to-text")
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]:
        ext = ".wav"

    temp_name = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join("static", temp_name)

    try:
        # Save uploaded audio
        audio_file.save(temp_path)

        # Transcribe using Faster-Whisper
        model = get_whisper_model()
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            language = None
        )


        transcript = " ".join([seg.text.strip() for seg in segments]).strip()
        transcript = transcript.replace("  ", " ").strip()

        # Translation option
        translate_to = request.form.get("translate_to", "").strip()
        translated_text = None

        if translate_to:
            lang_map = {
                "English": "en",
                "Tamil": "ta",
                "Hindi": "hi",
                "Korean": "ko"
            }

            target_lang = lang_map.get(translate_to, "en")

            translated_text = GoogleTranslator(
                source="auto",
                target=target_lang
            ).translate(transcript)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "language": info.language,
            "transcript": transcript,
            "translated": translated_text
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# ----------------------------
# 3) TEXT -> SPEECH (pyttsx3)
# ----------------------------
@app.post("/api/text-to-speech")
def text_to_speech():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty"}), 400

    try:
        # Ensure static folder exists
        os.makedirs("static", exist_ok=True)

        engine = pyttsx3.init()
        engine.save_to_file(text, AUDIO_OUTPUT_PATH)
        engine.runAndWait()

        return send_file(AUDIO_OUTPUT_PATH, mimetype="audio/wav", as_attachment=False)

    except Exception as e:
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Ensure static folder exists
    os.makedirs("static", exist_ok=True)

    app.run(debug=True, port=5000)
