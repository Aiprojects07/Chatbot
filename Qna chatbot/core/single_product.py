import os, re, json, time, logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import anthropic

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("single_product_core")

# ===== Config =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "kult-beauty-jam")
EMBED_MODEL      = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL", "anthropic")  # "anthropic" or "openai"
ANTHROPIC_MODEL  = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
OPENAI_CHAT_MODEL= os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

PRODUCT_NAME     = "Glossier Generation G – Jam"

# External system prompt file (optional override)
PROMPT_FILE_PATH = (Path(__file__).resolve().parent.parent.parent / "data" / "Chatbot system message prompt.txt")

# ===== Clients =====
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anth = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ===== CLI mode =====

# ===== Utilities =====
SPEC_WORDS = [
    "transfer", "mask-proof", "mask proof", "wear time", "longevity", "opacity",
    "finish", "price", "dupe", "best for", "skip if", "pros", "cons"
]
GLOSSARY_WORDS = [
    "mac", "nc15", "nc20", "nc25", "nc30", "nc35", "nc37", "nc40", "nc42",
    "nc44", "nc45", "nc47", "nc50", "wheatish", "golden brown", "dusky", "undertone"
]
VERDICT_HINTS = ["pros", "cons", "best for", "skip if", "who should buy", "who should skip"]

# Default minimal system prompt (used only if file read fails)
_DEFAULT_SYSTEM_PROMPT = f"""You are a beauty expert chatbot for one product: {PRODUCT_NAME}.
Keep answers concise and grounded in provided context. Provide helpful, accurate information in plain text format without markdown formatting or citations."""

# Load system prompt from external file if available
try:
    SYSTEM_PROMPT = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip() or _DEFAULT_SYSTEM_PROMPT
except Exception:
    SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT

def embed_query(q: str) -> List[float]:
    return oai.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

def soft_intents(q: str) -> Dict[str, bool]:
    ql = q.lower()
    return {
        "wants_specs": any(w in ql for w in SPEC_WORDS),
        "wants_verdict": any(w in ql for w in VERDICT_HINTS),
        "wants_glossary": any(w in ql for w in GLOSSARY_WORDS),
    }

def retrieve(q: str, k: int = 10) -> List[Dict[str, Any]]:
    vec = embed_query(q)
    res = index.query(vector=vec, top_k=k, include_metadata=True)
    hits = []
    for m in res.matches:
        md = m.metadata or {}
        doc_type = md.get("doc_type","")
        # Map to display labels
        label = {
            "faq": "[FAQ]",
            "snapshot": "[Snapshot]",
            "atomic_fact": "[Snapshot]",
            "glossary": "[Glossary]",
        }.get(doc_type, "[Snapshot]")
        # Special label for section_4 verdict bullets
        if doc_type == "snapshot" and md.get("section") == "section_4":
            label = "[Verdict]"
        hits.append({
            "score": float(m.score),
            "text": md.get("text",""),
            "metadata": md,
            "label": label
        })
    return hits

def boost(hits: List[Dict[str, Any]], q: str) -> List[Dict[str, Any]]:
    """Heuristic re-ranking: promote specs, verdict bullets, or glossary when relevant."""
    intents = soft_intents(q)
    def key(h):
        bonus = 0.0
        dt = h["metadata"].get("doc_type")
        section = h["metadata"].get("section","")
        topics = [t.lower() for t in h["metadata"].get("topic",[])]
        txt = h["text"].lower()

        # Spec boost: snapshot/atomic first
        if intents["wants_specs"] and dt in ("atomic_fact","snapshot"):
            bonus += 0.25
        # Verdict boost for pros/cons/best/skip
        if intents["wants_verdict"] and section == "section_4":
            bonus += 0.25
        # Glossary boost for MAC/Indian terms
        if intents["wants_glossary"] and dt == "glossary":
            bonus += 0.20
        # Tiny bonus if the query words appear verbatim
        if any(w in txt for w in re.findall(r"\w+", q.lower())):
            bonus += 0.05
        return -(h["score"] + bonus)
    return sorted(hits, key=key)

def build_context(hits: List[Dict[str, Any]], limit: int = 5) -> str:
    out = []
    for h in hits[:limit]:
        # Remove the label prefix - just use the text content
        out.append(h["text"])
    return "\n\n".join(out)

def answer_with_openai(question: str, context: str) -> str:
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
    ]
    resp = oai.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=450
    )
    return resp.choices[0].message.content

def answer_with_anthropic(question: str, context: str) -> str:
    msg = anth.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=450,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}]
    )
    return msg.content[0].text

def generate_answer(question: str, context: str) -> str:
    try:
        if PRIMARY_MODEL == "anthropic":
            return answer_with_anthropic(question, context)
        else:
            return answer_with_openai(question, context)
    except Exception:
        # fallback to the other provider
        try:
            if PRIMARY_MODEL == "anthropic":
                return answer_with_openai(question, context)
            else:
                return answer_with_anthropic(question, context)
        except Exception:
            # final fallback: extractive dump
            return ("Here’s what I found:\n\n" + context[:1000]) if context else \
                   "I couldn’t find enough context to answer confidently."

def answer(question: str, top_k: int = 10, limit: int = 5, provider: Optional[str] = None) -> str:
    """Single entry point used by both CLI and Netlify Function."""
    global PRIMARY_MODEL
    if provider:
        PRIMARY_MODEL = provider

    hits = retrieve(question, k=top_k)
    if not hits:
        return "I couldn't find anything relevant for this product."

    hits = boost(hits, question)
    ctx  = build_context(hits, limit=limit)
    ans  = generate_answer(question, ctx)
    return ans
