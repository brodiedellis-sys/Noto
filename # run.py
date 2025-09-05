# run.py
from app import app, _ping_lm

if __name__ == "__main__":
    print("[Noto] Starting server on http://127.0.0.1:5000")
    _ping_lm()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
