# run.py
from app import app, _ping_lm

if __name__ == "__main__":
    print("[Noto] Using templates at:", app.template_folder, "root:", app.root_path)
    _ping_lm()
    app.run(debug=True)
