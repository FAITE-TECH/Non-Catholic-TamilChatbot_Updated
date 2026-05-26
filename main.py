"""
Tamil Bible Chatbot — FastAPI + Chat UI
========================================
Visit http://127.0.0.1:8000  →  full chatbot interface
Visit http://127.0.0.1:8000/docs  →  Swagger API

Run:
  $env:GEMINI_API_KEY="your_key"   (PowerShell)
  uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_PATH   = os.getenv("JSONL_PATH",     "cleaned_tamil_bible_verses.jsonl")
BOOKS_DIR    = os.getenv("BOOKS_DIR",      "tamil_bible_books")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "YOUR KEY HERE")
EMBED_MODEL  = "distiluse-base-multilingual-cased-v1"
CHUNK_SIZE   = 5
GEMINI_MODEL = "gemini-3.5-flash"

ALL_BOOKS: List[str] = [
    "ஆதியாகமம்", "யாத்திராகமம்", "லேவியராகமம்", "எண்ணாகமம்", "உபாகமம்",
    "யோசுவா", "நியாயாதிபதிகள்", "ரூத்", "1 சாமுவேல்", "2 சாமுவேல்",
    "1 அரசர்கள்", "2 அரசர்கள்", "1 நாளாகமம்", "2 நாளாகமம்",
    "எஸ்றா", "நெகேமியா", "எஸ்தர்", "யோபு", "சங்கீதம்", "நீதிமொழிகள்",
    "பிரசங்கி", "உன்னதப்பாட்டு", "எசாயா", "எரேமியா", "விலாப்பாட்டு",
    "எசேக்கியேல்", "தானியேல்", "ஓசியா", "யோவேல்", "ஆமோஸ்", "ஒபதியா",
    "யோனா", "மீக்கா", "நாகூம்", "ஆபக்கூக்", "செப்பனியா", "ஆகாய்",
    "சகரியா", "மல்கி", "மத்தேயு", "மாற்கு", "லூக்கா", "யோவான்",
    "அப்போஸ்தலர்", "ரோமர்", "1 கொரிந்தியர்", "2 கொரிந்தியர்", "கலாத்தியர்",
    "எபேசியர்", "பிலிப்பியர்", "கொலோசெயர்", "1 தெசலோனிக்கேயர்",
    "2 தெசலோனிக்கேயர்", "1 தீமோத்தேயு", "2 தீமோத்தேயு", "தீத்து", "பிலேமோன்",
    "எபிரெயர்", "யாக்கோபு", "1 பேதுரு", "2 பேதுரு", "1 யோவான்",
    "2 யோவான்", "3 யோவான்", "யூதா", "வெளிப்படுத்தல்",
]

state: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_chunk_verses(filepath: str, chunk_size: int = CHUNK_SIZE):
    with open(filepath, "r", encoding="utf-8") as f:
        verses = [json.loads(line) for line in f if line.strip()]
    chunks = []
    for i in range(0, len(verses), chunk_size):
        batch = verses[i: i + chunk_size]
        chunks.append({
            "text":   " ".join(v["text"] for v in batch),
            "source": ", ".join(f'{v["book"]} {v["chapter"]}:{v["verse"]}' for v in batch),
        })
    verse_lookup   = {(v["book"], v["chapter"], v["verse"]): v["text"] for v in verses}
    chapter_index  = {}
    for v in verses:
        chapter_index.setdefault(v["book"], {}).setdefault(v["chapter"], []).append(v)
    return verses, chunks, verse_lookup, chapter_index


def create_faiss_index(chunks):
    texts   = [c["text"]   for c in chunks]
    sources = [c["source"] for c in chunks]
    model   = SentenceTransformer(EMBED_MODEL)
    embs    = model.encode(texts, show_progress_bar=True, batch_size=64)
    idx     = faiss.IndexFlatL2(embs.shape[1])
    idx.add(np.array(embs, dtype="float32"))
    return idx, model, texts, sources


def init_gemini():
    if not GEMINI_KEY:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        def _gen(prompt):
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
        print("✅  Gemini ready (google-genai).")
        return _gen
    except ImportError:
        pass
    try:
        import google.generativeai as g
        g.configure(api_key=GEMINI_KEY)
        m = g.GenerativeModel(GEMINI_MODEL)
        def _gen(prompt):
            return m.generate_content(prompt).text
        print("✅  Gemini ready (google-generativeai).")
        return _gen
    except Exception as e:
        print(f"⚠️  Gemini failed: {e}")
        return None


def build_prompt(context: str, question: str) -> str:
    return f"""நீ ஒரு தமிழ் வேதாகம உதவியாளர். பின்வரும் வேதாகம உரைகளைப் பயன்படுத்தி, பயனரின் கேள்விக்கு நட்பு மற்றும் ஆதரவாக பதிலளி.

📚 உரைகள்:
{context}

🙋‍♂️ பயனர்: {question}
🤖 பதில்:"""


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳  Loading dataset …")
    verses, chunks, verse_lookup, chapter_index = load_and_chunk_verses(JSONL_PATH)
    print(f"⏳  Encoding {len(chunks)} chunks …")
    index, embed_model, texts, sources = create_faiss_index(chunks)
    generate = init_gemini()
    if not generate:
        print("⚠️  Set GEMINI_API_KEY to enable AI answers.")
    state.update(
        verses=verses, chunks=chunks, texts=texts, sources=sources,
        embed_model=embed_model, index=index, generate=generate,
        verse_lookup=verse_lookup, chapter_index=chapter_index,
        book_set=sorted({v["book"] for v in verses}),
    )
    print(f"✅  Ready — {len(verses):,} verses · {len(chunks):,} chunks")
    yield
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="📖 Tamil Bible Chatbot API",
    description=(
        "Semantic search + Gemini RAG over the full Tamil Bible.\n\n"
        "**Chat UI:** `http://127.0.0.1:8000`\n\n"
        "Set `GEMINI_API_KEY` env var before starting."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Chat UI — served at /
# ---------------------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="ta">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>தமிழ் வேதாகம சாட்பாட்</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+Tamil:wght@400;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #0f0e0b;
    --surface:   #1a1814;
    --card:      #221f1a;
    --border:    #3a3428;
    --gold:      #c9a84c;
    --gold-dim:  #8a6f30;
    --cream:     #f0e6cc;
    --muted:     #7a6e5a;
    --user-bg:   #2a2318;
    --bot-bg:    #1e1c17;
    --accent:    #d4af60;
    --red:       #c0392b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--cream);
    font-family: 'Lora', Georgia, serif;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .logo {
    width: 42px; height: 42px;
    background: radial-gradient(circle at 40% 35%, #e8c96a, #8a6020);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 0 18px #c9a84c44;
  }
  header h1 {
    font-family: 'Noto Serif Tamil', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.03em;
  }
  header p {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 2px;
    font-style: italic;
  }
  .status-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #27ae60;
    margin-left: auto;
    box-shadow: 0 0 8px #27ae6088;
    flex-shrink: 0;
  }
  .status-dot.offline { background: var(--red); box-shadow: 0 0 8px #c0392b88; }

  /* ── Chat window ── */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 28px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    scroll-behavior: smooth;
  }
  #chat::-webkit-scrollbar { width: 5px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ── Messages ── */
  .msg { display: flex; gap: 12px; animation: fadeUp .35s ease; max-width: 820px; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-top: 2px;
  }
  .msg.user .avatar { background: var(--gold-dim); }
  .msg.bot  .avatar {
    background: radial-gradient(circle at 40% 35%, #e8c96a, #7a5a18);
    box-shadow: 0 0 10px #c9a84c33;
  }

  .bubble {
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.95rem;
    line-height: 1.75;
    max-width: 100%;
  }
  .msg.user .bubble {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-top-right-radius: 4px;
    font-family: 'Noto Serif Tamil', serif;
  }
  .msg.bot .bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
  }

  .sources {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 0.72rem;
    color: var(--gold-dim);
    font-style: italic;
    font-family: 'Lora', serif;
  }
  .sources span { color: var(--gold); font-weight: 600; }

  /* ── Typing indicator ── */
  .typing { display: flex; gap: 5px; padding: 6px 4px; align-items: center; }
  .typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--gold-dim);
    animation: blink 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes blink {
    0%,80%,100% { opacity: .3; transform: scale(1); }
    40%         { opacity: 1;  transform: scale(1.3); }
  }

  /* ── Welcome card ── */
  .welcome {
    text-align: center;
    padding: 48px 20px 32px;
    color: var(--muted);
  }
  .welcome .cross { font-size: 2.5rem; margin-bottom: 14px; }
  .welcome h2 {
    font-family: 'Noto Serif Tamil', serif;
    font-size: 1.4rem;
    color: var(--gold);
    margin-bottom: 8px;
  }
  .welcome p { font-size: 0.85rem; line-height: 1.8; max-width: 400px; margin: 0 auto; }
  .chips {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-top: 20px;
  }
  .chip {
    background: var(--card); border: 1px solid var(--border);
    color: var(--cream); font-size: 0.78rem;
    padding: 7px 14px; border-radius: 20px; cursor: pointer;
    font-family: 'Noto Serif Tamil', serif;
    transition: border-color .2s, color .2s;
  }
  .chip:hover { border-color: var(--gold); color: var(--gold); }

  /* ── Input bar ── */
  footer {
    padding: 16px 20px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .input-row {
    display: flex; gap: 10px; align-items: flex-end;
    max-width: 820px; margin: 0 auto;
  }
  textarea {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--cream);
    font-family: 'Noto Serif Tamil', serif;
    font-size: 0.92rem;
    padding: 12px 16px;
    resize: none;
    min-height: 48px;
    max-height: 140px;
    outline: none;
    transition: border-color .2s;
    overflow-y: auto;
    line-height: 1.6;
  }
  textarea:focus { border-color: var(--gold-dim); }
  textarea::placeholder { color: var(--muted); }

  button#send {
    width: 48px; height: 48px; flex-shrink: 0;
    background: var(--gold);
    border: none; border-radius: 12px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .2s, transform .1s;
  }
  button#send:hover   { background: var(--accent); }
  button#send:active  { transform: scale(.93); }
  button#send:disabled { background: var(--border); cursor: not-allowed; }
  button#send svg { fill: #0f0e0b; width: 20px; height: 20px; }

  .hint {
    text-align: center; font-size: 0.7rem;
    color: var(--muted); margin-top: 8px; font-style: italic;
  }
</style>
</head>
<body>

<header>
  <div class="logo">✝</div>
  <div>
    <h1>தமிழ் வேதாகம சாட்பாட்</h1>
    <p>Tamil Bible · 66 Books · Gemini RAG</p>
  </div>
  <div class="status-dot" id="dot"></div>
</header>

<div id="chat">
  <div class="welcome" id="welcome">
    <div class="cross">📖</div>
    <h2>வணக்கம்! தேவன் உங்களை நேசிக்கிறார்.</h2>
    <p>கேள்விகளை தமிழிலோ ஆங்கிலத்திலோ கேளுங்கள். வேதாகம வசனங்களின் அடிப்படையில் பதில் வழங்கப்படும்.</p>
    <div class="chips">
      <div class="chip" onclick="ask(this)">யோவான் 3:16 என்ன சொல்கிறது?</div>
      <div class="chip" onclick="ask(this)">சங்கீதம் 23ஆம் அதிகாரம் என்ன?</div>
      <div class="chip" onclick="ask(this)">நோவாவின் கப்பல் பற்றி வேதம் என்ன சொல்கிறது?</div>
      <div class="chip" onclick="ask(this)">தாவீதும் கோலியாத்தும் — என்ன நடந்தது?</div>
      <div class="chip" onclick="ask(this)">இயேசு செய்த அற்புதங்கள் என்னென்ன?</div>
      <div class="chip" onclick="ask(this)">பத்து கட்டளைகள் என்னென்ன?</div>
    </div>
  </div>
</div>

<footer>
  <div class="input-row">
    <textarea id="inp" rows="1" placeholder="கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்…" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="send" onclick="sendMsg()" title="அனுப்பு">
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
      </svg>
    </button>
  </div>
  <p class="hint">Enter to send · Shift+Enter for new line</p>
</footer>

<script>
const chat  = document.getElementById('chat');
const inp   = document.getElementById('inp');
const btn   = document.getElementById('send');
const dot   = document.getElementById('dot');
const TOP_K = 3;

// Check server health on load
async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    dot.className = 'status-dot';
    dot.title = `${d.total_verses.toLocaleString()} verses · Gemini: ${d.gemini_available ? 'ON' : 'OFF'}`;
  } catch {
    dot.className = 'status-dot offline';
    dot.title = 'Server unreachable';
  }
}
checkHealth();

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
}

function ask(el) {
  inp.value = el.textContent;
  autoResize(inp);
  sendMsg();
}

function appendMsg(role, html, sources) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '🙋' : '✝';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;

  if (sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.innerHTML = `<span>📚 மேற்கோள்கள்:</span> ${sources.join(' · ')}`;
    bubble.appendChild(s);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function showTyping() {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  msg.id = 'typing-msg';
  msg.innerHTML = `
    <div class="avatar">✝</div>
    <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-msg');
  if (t) t.remove();
}

async function sendMsg() {
  const q = inp.value.trim();
  if (!q) return;

  inp.value = '';
  inp.style.height = 'auto';
  btn.disabled = true;

  appendMsg('user', escHtml(q));
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: TOP_K }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    removeTyping();
    const formatted = data.answer
      .replace(/\\n/g, '<br>')
      .replace(/📖/g, '<span style="color:var(--gold)">📖</span>');
    appendMsg('bot', formatted, data.sources);

  } catch (err) {
    removeTyping();
    appendMsg('bot', `<span style="color:#e74c3c">⚠️ பிழை: ${escHtml(err.message)}</span>`);
  }

  btn.disabled = false;
  inp.focus();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui():
    return CHAT_HTML


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., examples=["இன்று நான் சோகமாக இருக்கிறேன். தேவன் என்ன சொல்கிறார்?"])
    top_k: int    = Field(default=3, ge=1, le=10)

class SearchRequest(BaseModel):
    query: str = Field(..., examples=["இயேசு அன்பு"])
    top_k: int = Field(default=5, ge=1, le=20)

class SearchResult(BaseModel):
    rank: int; source: str; text: str; score: float

class SearchResponse(BaseModel):
    query: str; results: List[SearchResult]

class ChatResponse(BaseModel):
    question: str; answer: str; sources: List[str]; gemini_used: bool

class VerseResponse(BaseModel):
    book: str; chapter: int; verse: int; text: str

class ChapterResponse(BaseModel):
    book: str; chapter: int; verses: List[VerseResponse]

class HealthResponse(BaseModel):
    status: str; total_verses: int; total_chunks: int
    faiss_vectors: int; gemini_available: bool; books_count: int


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        total_verses=len(state["verses"]),
        total_chunks=len(state["chunks"]),
        faiss_vectors=state["index"].ntotal,
        gemini_available=state["generate"] is not None,
        books_count=len(state["book_set"]),
    )

@app.get("/books", response_model=List[str], tags=["Bible"], summary="All 66 book names")
def list_books():
    return ALL_BOOKS

@app.get("/books/{book_name}/raw", tags=["Bible"], summary="Raw text from tamil_bible_books/")
def get_raw_book(book_name: str):
    p = Path(BOOKS_DIR) / f"{book_name}.txt"
    if not p.exists():
        raise HTTPException(404, detail=f"File not found: {p}")
    return {"book": book_name, "content": p.read_text(encoding="utf-8")}

@app.get("/verse", response_model=VerseResponse, tags=["Bible"], summary="Fetch a single verse")
def get_verse(
    book:    str = Query(..., examples=["யோவான்"]),
    chapter: int = Query(..., ge=1, examples=[3]),
    verse:   int = Query(..., ge=1, examples=[16]),
):
    text = state["verse_lookup"].get((book, chapter, verse))
    if text is None:
        raise HTTPException(404, detail=f"Not found: {book} {chapter}:{verse}")
    return VerseResponse(book=book, chapter=chapter, verse=verse, text=text)

@app.get("/chapter", response_model=ChapterResponse, tags=["Bible"], summary="All verses in a chapter")
def get_chapter(
    book:    str = Query(..., examples=["யோவான்"]),
    chapter: int = Query(..., ge=1, examples=[3]),
):
    verses = state["chapter_index"].get(book, {}).get(chapter)
    if not verses:
        raise HTTPException(404, detail=f"Not found: {book} chapter {chapter}")
    return ChapterResponse(
        book=book, chapter=chapter,
        verses=[VerseResponse(**{k: v[k] for k in ("book","chapter","verse","text")}) for v in verses],
    )

@app.post("/search", response_model=SearchResponse, tags=["Search"], summary="Semantic verse search")
def semantic_search(body: SearchRequest):
    q = state["embed_model"].encode([body.query]).astype("float32")
    D, I = state["index"].search(q, body.top_k)
    return SearchResponse(query=body.query, results=[
        SearchResult(rank=r+1, source=state["sources"][i], text=state["texts"][i], score=float(d))
        for r, (d, i) in enumerate(zip(D[0], I[0]))
    ])

@app.post("/chat", response_model=ChatResponse, tags=["Chat"], summary="Tamil Bible Q&A (Gemini RAG)")
def chat(body: ChatRequest):
    q = state["embed_model"].encode([body.question]).astype("float32")
    D, I = state["index"].search(q, body.top_k)
    ret_texts   = [state["texts"][i]   for i in I[0]]
    ret_sources = [state["sources"][i] for i in I[0]]
    context     = "\n\n".join(f"📖 {t}" for t in ret_texts)
    generate    = state["generate"]

    if generate is None:
        return ChatResponse(
            question=body.question,
            answer="⚠️ GEMINI_API_KEY not set.\n\nமிகவும் பொருத்தமான வசனங்கள்:\n\n" + context,
            sources=ret_sources, gemini_used=False,
        )
    try:
        answer      = generate(build_prompt(context, body.question))
        gemini_used = True
    except Exception as e:
        answer      = f"Gemini error: {e}\n\n{context}"
        gemini_used = False

    return ChatResponse(question=body.question, answer=answer, sources=ret_sources, gemini_used=gemini_used)
