from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import io
import os
import uuid

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS

app = Flask(__name__)
CORS(app)

WHISPER_MODEL_SIZE = "small"
AUDIO_OUTPUT_PATH = "static/output_audio.mp3"

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


# -----------------------------------
# 1) IMAGE -> DESCRIPTION (Demo Mode)
# -----------------------------------
@app.post("/api/image-caption")
def image_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(request.files["image"]).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Demo description (cloud-safe)
    description = "This is a demo image caption generated for accessibility purposes."

    return jsonify({
        "description": description
    })


# -----------------------------------
# 2) SPEECH -> TEXT
# -----------------------------------
@app.post("/api/speech-to-text")
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]:
        ext = ".wav"

    os.makedirs("static", exist_ok=True)
    temp_path = os.path.join("static", f"temp_{uuid.uuid4().hex}{ext}")

    try:
        audio_file.save(temp_path)

        model = get_whisper_model()
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True
        )

        transcript = " ".join([seg.text.strip() for seg in segments]).strip()

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

        os.remove(temp_path)

        return jsonify({
            "language": info.language,
            "transcript": transcript,
            "translated": translated_text
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# 3) TEXT -> SPEECH (Cloud Safe)
# -----------------------------------
@app.post("/api/text-to-speech")
def text_to_speech():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty"}), 400

    try:
        os.makedirs("static", exist_ok=True)

        tts = gTTS(text)
        tts.save(AUDIO_OUTPUT_PATH)

        return send_file(AUDIO_OUTPUT_PATH, mimetype="audio/mpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=False)
