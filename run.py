# run.py (top of file)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, _ping_lm

if __name__ == "__main__":
    print("[Noto] Starting server on http://127.0.0.1:5000")
    _ping_lm()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
