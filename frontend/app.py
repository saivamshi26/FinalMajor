"""
Minimal front end for SimpleHTR.
Upload image -> press Run -> output text appears on the page.

Run:
  cd "C:\\Major Pro\\Mukesh_new\\SimpleHTR_modified\\SimpleHTR-main"
  .\\venv\\Scripts\\python.exe frontend\\app.py
  Then open: http://127.0.0.1:5000
"""

import os
import sys
import uuid

from flask import Flask, jsonify, render_template, request


FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))  # .../SimpleHTR-main/frontend
ROOT = os.path.dirname(FRONTEND_DIR)  # .../SimpleHTR-main

# Make src importable (so `from model import ...` works)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import Model, DecoderType  # noqa: E402
from sentence_infer import full_image_predict  # noqa: E402


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        char_path = os.path.join(ROOT, "model", "charList.txt")
        with open(char_path, "r", encoding="utf-8", errors="replace") as f:
            char_list = f.read()

        # BeamSearch usually gives better CTC decoding than greedy best-path.
        _MODEL = Model(char_list, DecoderType.BeamSearch, must_restore=True, dump=False)
    return _MODEL


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_ocr():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400

    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    # Keep it simple: allow common image types
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return jsonify({"ok": False, "error": "Unsupported file type"}), 400

    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    try:
        model = get_model()
        text = full_image_predict(model, path, verbose=False, spell_check=True)
        return jsonify({"ok": True, "text": text if text else "(empty)"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

