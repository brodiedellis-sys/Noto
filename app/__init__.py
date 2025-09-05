# NOTO ACTUAL LAUNCH (package version)
from flask import Flask, request, jsonify, render_template, abort
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timedelta
import requests, re, os, random, json, math, ipaddress
from typing import List, Dict, Any, Optional
from functools import wraps

# --- Short-greeting tone gate ---
_SHORT_GREETING = re.compile(
    r"^\s*(hey|hi|hello|yo|sup|what(?:'|’)s up|whats up)\s*[\.\?!]*\s*$",
    re.IGNORECASE
)
_AFFECT_WORDS = {"amazing", "great", "promoted", "sad", "stressed", "overwhelmed", "excited", "pumped", "tired", "empty"}

def is_short_neutral_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if _SHORT_GREETING.match(t):
        return True
    words = re.findall(r"[a-zA-Z']+", t.lower())
    return (len(words) <= 4) and (not any(w in _AFFECT_WORDS for w in words))

def short_greeting_reply() -> str:
    return random.choice([
        "Hey! How’s your day going?",
        "Hi :) what’s up today?",
        "Hey—what’s happening?",
        "Yo! Anything fun or just checking in?"
    ])

# ------------------------------------------------------------------------------
# APP + DB INIT (uses Flask instance/ dir for DB)
# ------------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)
import os

@app.route("/_debug")
def _debug():
    return {
        "file": __file__,
        "cwd": os.getcwd(),
        "template_folder": app.template_folder,
        "routes": [str(r) for r in app.url_map.iter_rules()],
    }

os.makedirs(app.instance_path, exist_ok=True)  # make sure /instance exists

DB_PATH = os.path.join(app.instance_path, "noto.db")
print(f"[Noto] Using database at: {DB_PATH}")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DB_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ------------------------------------------------------------------------------
# LOCKED-DOWN DEBUG ACCESS
# ------------------------------------------------------------------------------
def _is_private_ip(ip: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private or ip_obj.is_loopback
    except Exception:
        return False

def require_debug_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        client_ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or ""
        is_local = _is_private_ip(client_ip)
        token_required = os.environ.get("NOTO_DEBUG_TOKEN", "").strip()
        provided = (request.headers.get("X-Debug-Token") or "").strip()
        if is_local:
            return fn(*args, **kwargs)
        if token_required and provided == token_required:
            return fn(*args, **kwargs)
        return jsonify({"error": "forbidden"}), 403
    return wrapper

# ------------------------------------------------------------------------------
# TUNABLES
# ------------------------------------------------------------------------------
PREF_GAIN = 2.0
NOVELTY_WINDOW = 4
NOVELTY_DROP = 0.6
TEMP = 0.6
EMA_ALPHA = 0.3
MAX_MEMORY_LEN = 64
MAX_MEMORIES_FROM_HISTORY = 5
CRISIS_COOLDOWN_MINUTES = 20

STYLE_SINGLE   = "single_action"
STYLE_MULTI    = "multiple_options"
STYLE_REFLECT  = "reflection"

# ------------------------------------------------------------------------------
# CONFIGURATION (env overrides supported)
# ------------------------------------------------------------------------------
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")
ALLOWED_USERS = {u.lower() for u in ["brodie", "tanner", "aiden", "riley", "dad", "anon"]}
LMSTUDIO_TIMEOUT = 60 if os.name == 'nt' else 30

class FallbackRotator:
    def __init__(self):
        self.last_used = None
        self.fallbacks = [
            "Hey—tell me what’s up and we’ll go from there.",
            "Got you. What’s on your mind right now?",
            "I’m here. Want to share a bit about your day?",
            "All good. We can just chat—what’s happening?"
        ]
    def get(self):
        available = [f for f in self.fallbacks if f != self.last_used]
        choice = random.choice(available or self.fallbacks)
        self.last_used = choice
        return choice

fallback_rotator = FallbackRotator()

# ------------------------------------------------------------------------------
# DATABASE MODELS
# ------------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    handle = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    country = db.Column(db.String(32), nullable=True)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    user_message = db.Column(db.Text)
    ai_response = db.Column(db.Text)
    vitality = db.Column(db.Float)
    __table_args__ = (db.Index('ix_user_timestamp', 'user_id', 'timestamp'),)

class InteractionMeta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=False)
    style_goal = db.Column(db.String(32), nullable=False)
    effectiveness = db.Column(db.String(16))  # 'positive' | 'negative' | None

class UserMemory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)
    data = db.Column(db.Text, nullable=False)  # JSON per user
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------------------------------------------------------
# SCHEMA BOOTSTRAP
# ------------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    try:
        rows = db.session.execute(db.text("PRAGMA table_info(user)")).fetchall()
        cols = [r[1] for r in rows] if rows else []
        if rows and "country" not in cols:
            db.session.execute(db.text("ALTER TABLE user ADD COLUMN country VARCHAR(32)"))
            db.session.commit()
            print("[Noto] Added user.country column")
    except Exception as e:
        print("[WARN] schema bootstrap:", e)

# ------------------------------------------------------------------------------
# MEMORY LAYER
# ------------------------------------------------------------------------------
DEFAULT_MEMORY: Dict[str, Any] = {
    "goals": [],
    "wins": [],
    "moods": [],
    "preferences": {"depth": "adaptive", "tone": "direct", "style": "natural"},
    "facts": [],
    "rituals": {"night_skipped_streak": 0, "last_marked_date": None},
    "style_effectiveness": {
        STYLE_SINGLE:  {"ema": 0.50, "positive": 0, "negative": 0},
        STYLE_MULTI:   {"ema": 0.50, "positive": 0, "negative": 0},
        STYLE_REFLECT: {"ema": 0.50, "positive": 0, "negative": 0}
    },
    "user_triggers": {"positive": [], "negative": []},
    "recent_styles": [],
    "last_style_pending": None,
    "cooldowns": {"crisis_until": None},
    "last_updated": None
}

MOOD_WORDS = {"meh", "hopeful", "motivated", "stressed", "tired", "anxious", "calm", "grateful", "angry", "sad", "okay", "good", "foggy", "low", "pumped", "empty"}
GOAL_HINTS = {"goal", "focus", "this week", "plan to", "i will", "my aim", "target"}
WIN_HINTS  = {"finished", "did", "made", "completed", "sent", "called", "shipped", "pushed", "wrote"}
NEGATIVE_WORDS = {"nah", "no", "don’t want", "dont want", "won’t", "wont", "not really", "idk", "i don't feel like"}
POSITIVE_WORDS = {"ok", "okay", "done", "let’s", "lets", "sure", "i can", "i will", "yeah", "yup"}

# --- Greeting heuristics ---
GREETING_WORDS = {"hi","hey","hello","yo","sup","hiya","heyy","heyya","hallo"}
def is_casual_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if len(t) <= 12 and sum(1 for w in re.split(r"\s+", t) if w) <= 3:
        tokens = re.split(r"\W+", t)
        return any(w in GREETING_WORDS for w in tokens if w)
    return False

def greeting_reply() -> str:
    hr = datetime.utcnow().hour
    if 5 <= hr < 12:
        pool = ["morning! what’s up?", "morning — want a tiny plan for today or just chat?"]
    elif 12 <= hr < 17:
        pool = ["hey! what’s on your mind this afternoon?", "yo — quick check-in or just saying hi?"]
    elif 17 <= hr < 22:
        pool = ["hey! how’s your evening going?", "hi — want a 2-min wrap-up or just talk?"]
    else:
        pool = ["hey! what’s up?", "yo 🙂 what’s on your mind?", "hi! want me to suggest a tiny nudge or just hang?"]
    return random.choice(pool)

NEUTRAL_POSITIVE_HINTS = [
    "good","pretty good","great","okay","ok","fine","decent","alright",
    "productive","chill","relaxed","fun","happy","solid","not bad"
]
NEGATIVE_HINTS = [
    "bad","tired","exhausted","rough","drained","sad","low",
    "awful","overwhelmed","frustrated","angry","lonely","empty","no hope"
]

TONE_GATE_SKIP_PHRASES = {
    "just answer directly", "answer directly", "be direct",
    "no small talk", "no pep talk", "skip the tone",
    "no questions", "short answer", "plain answer",
    "skip tone", "dont analyze", "don't analyze", "no motivation speech"
}

POSITIVE_MARKERS = {
    "great", "awesome", "amazing", "good", "nice", "fun",
    "excited", "pumped", "stoked", "happy", "win", "promotion", "promoted", "🎉", "!"
}
NEGATIVE_MARKERS = {
    "sucks", "awful", "terrible", "overwhelmed", "tired", "anxious", "sad",
    "empty", "no hope", "hopeless", "stressed", "down", "burned out", "burnt out"
}
ASKS_FOR_IDEAS_MARKERS = {
    "any ideas", "suggestions", "what should i do", "what do i do",
    "how should i", "help me decide", "give me options", "nudge"
}
STUCKNESS_MARKERS = {
    "stuck", "don’t know where to start", "dont know where to start",
    "idk what to do", "not sure what to do", "blocked"
}
MOMENTUM_MARKERS = {
    "i could keep going", "i can keep going", "i'm on a roll", "im on a roll",
    "made progress", " making progress "
}

def detect_energy(text: str) -> str:
    t = (text or "").lower()
    if re.fullmatch(r"\s*(hi|hey|hello|yo|sup|what'?s up)\s*[.!?]?\s*", t):
        return "neutral"
    if any(tok in t for tok in POSITIVE_MARKERS):
        return "positive"
    if any(tok in t for tok in NEGATIVE_MARKERS):
        return "negative"
    return "neutral"

def detect_intents(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {
        "tone_gate_skip": any(p in t for p in TONE_GATE_SKIP_PHRASES),
        "asks_for_ideas": any(p in t for p in ASKS_FOR_IDEAS_MARKERS),
        "shows_stuckness": any(p in t for p in STUCKNESS_MARKERS),
        "shows_momentum": any(p in t for p in MOMENTUM_MARKERS),
    }

def detect_tone(user_text: str) -> str:
    t = (user_text or "").lower()
    if any(w in t for w in NEGATIVE_HINTS):
        return "negative"
    if any(w in t for w in NEUTRAL_POSITIVE_HINTS):
        return "positive"
    return "neutral"

QUESTION_STARTS = (
    "what","how","why","when","where","who","which","whats","how's","why's","where's","who's","which's"
)
QUESTION_PHRASES = (
    "can you","could you","do you","did you","are you","is it","should i","would you",
    "what's","what is","how do","how to","why is","why do","where do","where can"
)

def is_direct_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if "?" in t:
        return True
    words = re.split(r"\s+", t)
    if words and words[0] in QUESTION_STARTS:
        return True
    if any(phrase in t for phrase in QUESTION_PHRASES):
        return True
    return False

def _ensure_mem_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    mem = {**DEFAULT_MEMORY, **(data or {})}
    for k in ("preferences", "rituals", "style_effectiveness", "user_triggers", "cooldowns"):
        mem[k] = {**DEFAULT_MEMORY[k], **(mem.get(k) or {})}
    mem.setdefault("recent_styles", [])
    mem.setdefault("last_style_pending", None)
    return mem

def load_user_memory(user_id: int) -> Dict[str, Any]:
    rec = UserMemory.query.filter_by(user_id=user_id).first()
    if not rec:
        mem = json.loads(json.dumps(DEFAULT_MEMORY))
        rec = UserMemory(user_id=user_id, data=json.dumps(mem))
        db.session.add(rec)
        db.session.commit()
        return mem
    try:
        return _ensure_mem_shape(json.loads(rec.data))
    except Exception:
        return json.loads(json.dumps(DEFAULT_MEMORY))

def save_user_memory(user_id: int, mem: Dict[str, Any]) -> None:
    mem["last_updated"] = datetime.utcnow().isoformat()
    for key in ("goals", "wins", "moods", "facts", "recent_styles"):
        if isinstance(mem.get(key), list) and len(mem[key]) > MAX_MEMORY_LEN:
            mem[key] = mem[key][-MAX_MEMORY_LEN:]

    payload = json.dumps(mem, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 10000:
        print(f"[WARN] Memory overflow for user {user_id} — resetting to default")
        mem = json.loads(json.dumps(DEFAULT_MEMORY))
        payload = json.dumps(mem, ensure_ascii=False)

    rec = UserMemory.query.filter_by(user_id=user_id).first()
    if rec:
        rec.data = payload
        rec.updated_at = datetime.utcnow()
    else:
        rec = UserMemory(user_id=user_id, data=payload)
        db.session.add(rec)
    db.session.commit()

def infer_memory_updates(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    t_low = t.lower()
    updates: Dict[str, Any] = {"goals": [], "wins": [], "moods": [], "facts": [], "rituals": {}}
    for w in MOOD_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t_low):
            updates["moods"].append(w)
    if any(h in t_low for h in WIN_HINTS):
        updates["wins"].append(t[:140].replace("\n", " ").strip())
    if any(h in t_low for h in GOAL_HINTS):
        m = re.search(r"([^.!?]{0,200}(goal|focus|i will|plan to|this week)[^.!?]{0,200})", t_low)
        updates["goals"].append(((m.group(0) if m else t)[:180]))
    if re.search(r"\b(i\s+choose|i'm\s+stopping|i\s+stop|from now on|i commit)\b", t_low):
        updates["facts"].append(t[:200])
    if ("night" in t_low or "nightly" in t_low) and "ritual" in t_low:
        if any(w in t_low for w in ("skip", "skipped", "didn't", "didnt")):
            updates["rituals"]["night_skipped_streak"] = "+1"
        elif any(w in t_low for w in ("did", "completed", "finished")):
            updates["rituals"]["night_skipped_streak"] = "reset"
        updates["rituals"]["last_marked_date"] = date.today().isoformat()
    return updates

def apply_memory_updates(mem: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    if not updates:
        return mem
    for k in ("goals", "wins", "moods", "facts"):
        for item in updates.get(k, []) or []:
            if item and item not in mem[k]:
                mem[k].append(item)
    rit_u = updates.get("rituals") or {}
    if rit_u:
        if rit_u.get("night_skipped_streak") == "+1":
            mem["rituals"]["night_skipped_streak"] = int(mem["rituals"].get("night_skipped_streak", 0)) + 1
        elif rit_u.get("night_skipped_streak") == "reset":
            mem["rituals"]["night_skipped_streak"] = 0
        if rit_u.get("last_marked_date"):
            mem["rituals"]["last_marked_date"] = rit_u["last_marked_date"]
    return mem

def summarize_memory_for_prompt(mem: Dict[str, Any]) -> str:
    lines = []
    if mem.get("moods"): lines.append(f"Mood trend: {', '.join(mem['moods'][-5:])}")
    if mem.get("goals"):
        for g in mem["goals"][-3:]: lines.append(f"Goal: {g}")
    if mem.get("wins"): lines.append(f"Recent win: {mem['wins'][-1]}")
    if mem.get("rituals", {}).get("night_skipped_streak"):
        lines.append(f"Nightly ritual skipped streak: {mem['rituals']['night_skipped_streak']}")
    eff = mem.get("style_effectiveness", {})
    if eff: lines.append("Style EMA: " + ", ".join(f"{k}:{v.get('ema',0.5):.2f}" for k,v in eff.items()))
    tr = mem.get("user_triggers", {})
    if tr.get("negative"): lines.append("Avoid phrases: " + ", ".join(tr["negative"][-3:]))
    if tr.get("positive"): lines.append("Use phrases: " + ", ".join(tr["positive"][-3:]))
    if not lines: lines.append("(no notable memory yet; start light)")
    return "\n".join(lines)

# ------------------------------------------------------------------------------
# PROMPT CORE
# ------------------------------------------------------------------------------
def load_base_prompt() -> str:
    return (
        "You are Noto — a catalyst, mirror, and honest friend. Your purpose is to meet the user where they are: "
        "amplify joy in positive moments, validate neutral states without pushing, and support only when struggle appears. "
        "Speak naturally, like a close friend who’s direct and kind.\n\n"
        "TONE & STYLE:\n"
        "- Plain, human, brief. Max 3 sentences unless depth is clearly needed.\n"
        "- Match the user’s energy (positive / neutral / struggling). No therapy clichés, hype, or scripted lines.\n"
        "- Be a curious friend first, coach second. Don’t project sensations the user didn’t state.\n\n"
        "ENERGY / TONE GATES:\n"
        "A) Very short greeting (≤ 4 words) with no affective words (e.g., 'hey', 'what's up') → "
        "reply casually + one light question. Do NOT suggest actions.\n"
        "B) Positive → celebrate + 1 specific curiosity about the thing. "
        "Do NOT suggest tasks unless they ask for ideas.\n"
        "C) Neutral → light validation + optional engagement ('want to unpack or just vibe?'). "
        "Avoid auto-optimization.\n"
        "D) Struggling/negative → brief validation first; then either one tiny optional step or a clear, relevant question.\n\n"
        "DECISION CUES:\n"
        "- If the user asked a direct question: answer it clearly first.\n"
        "- If vague: ask one short clarifying question before any advice.\n"
        "- If they show openness: offer 1 tiny step or a simple A/B choice.\n"
        "- If they resist or say 'don’t tell me to X': respect that and offer a different path or just listen.\n\n"
        "MEMORY CALLBACKS:\n"
        "- Mention past details only if relevant now. Use at most one callback and paraphrase it.\n\n"
        "BOUNDARIES:\n"
        "- No medical/diagnostic, legal, or unsafe advice.\n\n"
        "OUTPUT CONSTRAINTS:\n"
        "- Respond in plain conversational text only.\n"
    )

# ------------------------------------------------------------------------------
# CRISIS DETECTION
# ------------------------------------------------------------------------------
CRISIS_PATTERNS_ACTIVE = [
    r"\bi\s*(will|am going to)\b.*\b(kill myself|end it|suicide|end my life|harm myself)\b",
    r"\b(plan|planning)\b.*\b(kill myself|end it|end my life|suicide)\b",
    r"\b(i might|i’m going to|im going to)\b.*\b(hurt myself|harm myself)\b",
]
CRISIS_PATTERNS_PASSIVE = [
    r"\b(wish i (weren't|wasn't) here|don’t want to be alive|dont want to be alive)\b",
    r"\b(no reason to live|life (is|feels) pointless|nothing matters)\b",
]
BOUNDARY_PATTERN = r"don'?t tell me to ([^\.!?\n]+)"

HELPLINES = {
    "US": "Call or text **988** for 24/7 support. If you’re in immediate danger, call **911**.",
    "United States": "Call or text **988** for 24/7 support. If you’re in immediate danger, call **911**.",
    "Canada": "Call or text **988** for 24/7 support. If you’re in immediate danger, call your local emergency number.",
    "UK": "Call **Samaritans 116 123** (free, 24/7). If you’re in immediate danger, call **999**.",
    "United Kingdom": "Call **Samaritans 116 123** (free, 24/7). If you’re in immediate danger, call **999**.",
    "Ireland": "Call **Samaritans 116 123** (free, 24/7). If you’re in immediate danger, call **112/999**.",
    "Australia": "Call **Lifeline 13 11 14** (24/7). If you’re in immediate danger, call **000**.",
    "New Zealand": "Call or text **1737** (24/7). If you’re in immediate danger, call **111**.",
    "India": "Call **Kiran 1800-599-0019** (24/7). If you’re in immediate danger, call your local emergency number."
}

def helpline_line(country_hint: Optional[str]) -> str:
    if not country_hint:
        return "If you’re in the U.S. or Canada, you can call or text **988** for 24/7 support. Otherwise, please call your **local emergency number**."
    for key, msg in HELPLINES.items():
        if country_hint.strip().lower() == key.lower():
            return msg
    return "If you’re in the U.S. or Canada, you can call or text **988** for 24/7 support. Otherwise, please call your **local emergency number**."

def detect_crisis_level(text: str) -> Optional[str]:
    t = (text or "").lower()
    for pat in CRISIS_PATTERNS_ACTIVE:
        if re.search(pat, t):
            return "active"
    for pat in CRISIS_PATTERNS_PASSIVE:
        if re.search(pat, t):
            return "passive"
    if re.search(BOUNDARY_PATTERN, t):
        return "boundary"
    return None

def build_crisis_response(user_text: str, level: str, country_hint: Optional[str]) -> str:
    t = (user_text or "").lower()
    m = re.search(BOUNDARY_PATTERN, t)
    avoid = m.group(1).strip() if m else None
    line = helpline_line(country_hint)

    if level == "active":
        msg = (
            "I’m really glad you told me. **Are you safe right now?**\n\n"
            f"If you feel at immediate risk, please call your **local emergency number** now. {line}\n\n"
            "We can stay here quietly if you’d like—no pressure to do anything."
        )
        if avoid:
            msg = msg.replace("no pressure to do anything", f"no pressure—and I won’t suggest {avoid}.")
        return msg

    if level == "passive":
        msg = (
            "That sounds incredibly heavy. Thank you for telling me. **Are you safe right now?**\n\n"
            f"{line}\n\n"
            "If talking helps, I’m here to listen—no fixes. If you want, we can name one tiny anchor that helps you get through the next few minutes."
        )
        if avoid:
            msg = msg.replace("—no fixes.", f"—no fixes, and I won’t suggest {avoid}.")
        return msg

    if avoid:
        return (
            f"Heard—no **{avoid}**. We can skip that. "
            "Do you want quiet company for a minute, or a simple check-in question?"
        )

    return (
        "I’m here with you. If you’re at risk, please call your **local emergency number**. "
        "In the U.S./Canada you can also call or text **988** for 24/7 support."
    )

def start_crisis_cooldown(mem: Dict[str, Any], minutes: int = CRISIS_COOLDOWN_MINUTES):
    mem.setdefault("cooldowns", {})
    mem["cooldowns"]["crisis_until"] = (datetime.utcnow() + timedelta(minutes=minutes)).isoformat()

def crisis_cooldown_active(mem: Dict[str, Any]) -> bool:
    ts = (mem.get("cooldowns") or {}).get("crisis_until")
    if not ts: return False
    try:
        return datetime.utcnow() < datetime.fromisoformat(ts)
    except Exception:
        return False

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------
def calculate_vitality(text: str) -> float:
    text = text.lower()
    boosters = {"did", "finished", "went", "made", "called", "completed"}
    if any(verb in text for verb in boosters):
        return min(1.0, 0.5 + 0.15 * sum(1 for verb in boosters if verb in text))
    return 0.5

SANITIZE_DEBUG = os.environ.get("NOTO_SANITIZE_DEBUG", "0") == "1"

def nuclear_sanitize(text: str) -> str:
    if not text:
        return ""
    original = text
    patterns = [
        r'<\|[^>]+\|>',
        r'```.+?```',
        r'(?i)\bto=(developer|assistant)\b.*',
        r'\[(?:INST|SYSTEM|MEMORY)[^\]]*\]',
        r'\(to developer:.*?\)',
        r'(?is)<think>.*?</think>',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*\{[^{}]{0,2000}\}\s*$', '', text.strip(), flags=re.DOTALL)
    text = re.sub(r'^\s*---[\s\S]{0,2000}?---\s*$', '', text.strip(), flags=re.DOTALL)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clean = " ".join(lines)
    if not clean or len(clean.split()) < 2:
        clean = ""
    if re.search(r'\b(as an ai|language model|system prompt|developer message)\b', clean, re.I):
        clean = ""
    if SANITIZE_DEBUG:
        print("[SANITIZE] cleaned :", clean[:300].replace("\n"," "))
    return clean

def is_important(text: str) -> bool:
    t = (text or "").lower()
    return (len(t.split()) > 8 or any(w in t for w in {"finally", "always", "never"}) or "?" in t or any(e in t for e in {"!", "..."}))

def get_conversation_history(user_id: int) -> List[Conversation]:
    return Conversation.query.filter_by(user_id=user_id).order_by(Conversation.timestamp.asc()).all()

def build_memory_prompt_from_history(history: List[Conversation]) -> str:
    important = [msg for msg in history if is_important(msg.user_message)]
    return "Key memories from recent messages:\n" + "\n".join(
        f"{msg.timestamp.date()}: {msg.user_message[:100]}..." for msg in important[-MAX_MEMORIES_FROM_HISTORY:]
    )

def _softmax(logits, temp: float = 1.0):
    mx = max(logits)
    exps = [math.exp((l - mx) / max(1e-6, temp)) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

def _pick_by_probs(items, probs):
    r = random.random()
    cum = 0.0
    for it, p in zip(items, probs):
        cum += p
        if r <= cum:
            return it
    return items[-1]

def _recent_penalty(style: str, recent: List[str]) -> float:
    if style in (recent or [])[-NOVELTY_WINDOW:]:
        return NOVELTY_DROP
    return 1.0

def _update_recent_styles(mem: Dict[str, Any], style: str, max_keep: int = 10):
    lst = mem.get("recent_styles") or []
    lst.append(style)
    if len(lst) > max_keep:
        lst = lst[-max_keep:]
    mem["recent_styles"] = lst

def effectiveness_from_reply(reply_text: str) -> Optional[str]:
    if not reply_text:
        return None
    t = reply_text.lower()
    long_enough = len(t.split()) >= 6
    positive = long_enough or any(w in t for w in POSITIVE_WORDS)
    negative = any(w in t for w in NEGATIVE_WORDS)
    if positive and not negative:
        return 'positive'
    if negative and not positive:
        return 'negative'
    return None

def evaluate_last_style(user_id: int, latest_user_text: str) -> None:
    meta = InteractionMeta.query.filter_by(user_id=user_id, effectiveness=None).order_by(InteractionMeta.id.desc()).first()
    if not meta: return
    eff = effectiveness_from_reply(latest_user_text)
    if not eff: return
    meta.effectiveness = eff
    db.session.add(meta)
    db.session.commit()
    mem = load_user_memory(user_id)
    style = meta.style_goal
    slot = mem.get("style_effectiveness", {}).get(style)
    if slot is None:
        slot = {"ema": 0.5, "positive": 0, "negative": 0}
        mem["style_effectiveness"][style] = slot
    if eff == "positive":
        slot["positive"] = slot.get("positive", 0) + 1
        target = 1.0
    else:
        slot["negative"] = slot.get("negative", 0) + 1
        target = 0.0
    prev = float(slot.get("ema", 0.5))
    slot["ema"] = max(0.1, min(0.9, (1 - EMA_ALPHA) * prev + EMA_ALPHA * target))
    save_user_memory(user_id, mem)

def decide_style(user_text: str, mem: Dict[str, Any], history: List[Conversation]) -> str:
    t = (user_text or "").lower()
    high_distress = any(x in t for x in ("empty", "no hope", "overwhelmed", "anxious", "sad"))
    low_moderate   = any(x in t for x in ("meh", "off", "down")) and not high_distress
    upbeat         = any(x in t for x in ("good", "great", "excited", "pumped"))
    low_energy     = any(x in t for x in ("meh", "don’t feel like", "dont feel like", "tired", "empty"))
    is_specific    = bool(re.search(r"because|due to|after|when|since|i .* (missed|failed|argued|quit|started)", t))
    early          = len(history) < 2
    unwilling      = any(x in t for x in NEGATIVE_WORDS)
    open_to_act    = any(x in t for x in POSITIVE_WORDS)

    logits = {STYLE_REFLECT: 0.0, STYLE_SINGLE: 0.0, STYLE_MULTI: 0.0}
    if high_distress or low_energy or unwilling:
        logits[STYLE_REFLECT] += 1.0
    elif upbeat:
        logits[STYLE_MULTI]  += 0.8
        logits[STYLE_SINGLE] += 0.4
    elif low_moderate:
        logits[STYLE_SINGLE] += 0.8
        logits[STYLE_REFLECT]+= 0.4

    if is_specific:
        logits[STYLE_SINGLE] += 0.6
    else:
        logits[STYLE_REFLECT]+= 0.4

    if early and not is_specific:
        logits[STYLE_REFLECT]+= 0.4
    else:
        logits[STYLE_SINGLE] += 0.2

    if open_to_act and not unwilling:
        logits[STYLE_SINGLE] += 0.5
        logits[STYLE_MULTI]  += 0.2

    if crisis_cooldown_active(mem):
        logits[STYLE_REFLECT] += 0.8

    eff = mem.get("style_effectiveness", {})
    for style in (STYLE_SINGLE, STYLE_MULTI, STYLE_REFLECT):
        ema = float((eff.get(style) or {}).get("ema", 0.5))
        logits[style] += PREF_GAIN * (ema - 0.5)

    recent = mem.get("recent_styles") or []
    for style in (STYLE_SINGLE, STYLE_MULTI, STYLE_REFLECT):
        logits[style] += math.log(_recent_penalty(style, recent))

    items = [STYLE_REFLECT, STYLE_SINGLE, STYLE_MULTI]
    probs = _softmax([logits[s] for s in items], temp=TEMP)
    return _pick_by_probs(items, probs)

def generate_noto_response(user_text: str, user_id: int) -> Dict[str, str]:
    evaluate_last_style(user_id, user_text)
    mem = load_user_memory(user_id)
    mem = apply_memory_updates(mem, infer_memory_updates(user_text))
    history = get_conversation_history(user_id)
    vitality = calculate_vitality(user_text)
    style_goal = decide_style(user_text, mem, history)
    _update_recent_styles(mem, style_goal, max_keep=10)
    save_user_memory(user_id, mem)

    base = load_base_prompt()
    memory_snapshot = summarize_memory_for_prompt(mem)

    messages = [
        {"role": "system", "content": base},
        {"role": "system", "content": f"Personalization snapshot (concise):\n{memory_snapshot}"},
        {"role": "system", "content": (
            f"Style goal for this reply: {style_goal}. "
            "If reflection: avoid action and keep it validating. "
            "If single_action: offer exactly one tiny, concrete step unless the user asks for options. "
            "If multiple_options: usually 2 light choices are enough; keep it simple and stop after the options."
        )},
    ]

    if history:
        messages.append({"role": "system", "content": build_memory_prompt_from_history(history)})

    for msg in history[-3:] if history else []:
        messages.append({"role": "user", "content": msg.user_message})
        if msg.ai_response:
            messages.append({"role": "assistant", "content": msg.ai_response})

    mode = "Warm/engaged" if vitality > 0.7 else "Neutral/listening"
    messages.append({"role": "system", "content": f"Current Mode: {mode}"})
    messages.append({"role": "user", "content": user_text})

    try:
        response = requests.post(
            LMSTUDIO_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 320,
                "stop": ["\nUser:", "\nNoto:"]
            },
            timeout=LMSTUDIO_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError("No choices in response")
        raw = data["choices"][0]["message"]["content"]
        clean = nuclear_sanitize(raw)
        if not clean:
            raise ValueError("Sanitization failed - empty after cleaning")
        return {"reply": clean, "style_goal": style_goal}
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return {"reply": fallback_rotator.get(), "style_goal": style_goal}

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------
@app.before_request
def verify_user():
    # Only used by /send; require a non-empty handle but don't restrict who can use it
    if request.endpoint == 'handle_message':
        payload = request.get_json(silent=True) or {}
        handle = (payload.get('u') or '').strip().lower()
        if not handle:
            abort(400, description="Missing handle")

@app.route("/")
def home():
    return render_template("index.html")

def _user_country(user: "User") -> Optional[str]:
    if not user:
        return None
    return (user.country or "").strip() or None

def _crisis_short_circuit_if_needed(user: "User", user_text: str):
    level = detect_crisis_level(user_text)
    if not level:
        return None
    reply = build_crisis_response(user_text, level, _user_country(user))
    mem = load_user_memory(user.id)
    start_crisis_cooldown(mem, minutes=CRISIS_COOLDOWN_MINUTES)
    save_user_memory(user.id, mem)
    convo = Conversation(user_id=user.id, user_message=user_text, ai_response=reply, vitality=0.0)
    db.session.add(convo)
    db.session.commit()
    return jsonify({"reply": reply})

@app.route("/send", methods=["POST"])
def handle_message():
    data = request.get_json() or {}
    user_text = (data.get("text") or "").strip()
    handle = (data.get("u") or "anon").lower().strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    user = User.query.filter_by(handle=handle).first()
    if not user:
        user = User(handle=handle)
        db.session.add(user)
        db.session.commit()

    history = get_conversation_history(user.id)
    if is_casual_greeting(user_text) and len(history) == 0 and not is_direct_question(user_text):
        reply = greeting_reply()
        convo = Conversation(user_id=user.id, user_message=user_text, ai_response=reply, vitality=0.5)
        db.session.add(convo); db.session.commit()
        return jsonify({"reply": reply})

    crisis_resp = _crisis_short_circuit_if_needed(user, user_text)
    if crisis_resp is not None:
        return crisis_resp

    result = generate_noto_response(user_text, user.id)
    ai_response = result["reply"]
    style_goal = result["style_goal"]

    convo = Conversation(
        user_id=user.id,
        user_message=user_text,
        ai_response=ai_response,
        vitality=calculate_vitality(user_text)
    )
    db.session.add(convo)
    db.session.commit()

    meta = InteractionMeta(user_id=user.id, conversation_id=convo.id, style_goal=style_goal, effectiveness=None)
    db.session.add(meta)
    db.session.commit()

    return jsonify({"reply": ai_response})

# ---- Safe debug endpoints (locked down) ----
@app.route("/debug/users", methods=["GET"])
@require_debug_auth
def debug_users():
    try:
        rows = db.session.execute(db.text(
            "SELECT id, handle, created_at, country FROM user ORDER BY id ASC"
        )).mappings().all()
        return jsonify({"users": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/memory", methods=["GET"])
@require_debug_auth
def debug_memory():
    handle = (request.args.get("u") or "").lower().strip()
    if not handle:
        return jsonify({"error": "missing ?u=<handle>"}), 400
    user = User.query.filter_by(handle=handle).first()
    if not user:
        return jsonify({"error": f"no such user '{handle}'"}), 404
    try:
        mem = load_user_memory(user.id)
        mem_copy = dict(mem)
        for k in ("goals", "wins", "moods", "facts", "recent_styles"):
            if isinstance(mem_copy.get(k), list) and len(mem_copy[k]) > 25:
                mem_copy[k] = mem_copy[k][-25:]
        return jsonify({
            "user": {"id": user.id, "handle": user.handle, "country": user.country},
            "memory": mem_copy
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- User memory endpoints ----
@app.route("/memory", methods=["GET"])
def view_memory():
    handle = request.args.get("u", "anon").lower()
    user = User.query.filter_by(handle=handle).first()
    if not user:
        return jsonify(DEFAULT_MEMORY)
    return jsonify(load_user_memory(user.id))

@app.route("/memory/reset", methods=["POST"])
def reset_memory():
    handle = request.args.get("u", "anon").lower()
    user = User.query.filter_by(handle=handle).first()
    if not user:
        return jsonify({"ok": True})
    save_user_memory(user.id, json.loads(json.dumps(DEFAULT_MEMORY)))
    return jsonify({"ok": True})

@app.route("/user/country", methods=["POST"])
def set_country():
    data = request.get_json(silent=True) or {}
    handle = (data.get("u") or "").lower().strip()
    country = (data.get("country") or "").strip()
    if not handle or handle not in ALLOWED_USERS:
        return jsonify({"error": "invalid or unauthorized user handle"}), 400
    user = User.query.filter_by(handle=handle).first()
    if not user:
        user = User(handle=handle)
        db.session.add(user)
        db.session.commit()
    user.country = country or None
    db.session.add(user)
    db.session.commit()
    return jsonify({"ok": True, "country": user.country})

# ------------------------------------------------------------------------------
# UTIL
# ------------------------------------------------------------------------------
def _ping_lm():
    try:
        response = requests.post(
            LMSTUDIO_URL,
            json={"messages": [{"role": "user", "content": "ping"}], "model": MODEL_NAME},
            timeout=5
        )
        if response.status_code == 200:
            print("[Noto] LM Studio connection successful")
        else:
            print(f"[Noto] LM Studio returned HTTP {response.status_code}")
    except Exception as e:
        print(f"[Noto] LM Studio connection failed: {str(e)}")
