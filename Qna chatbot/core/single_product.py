import os, re, json, time, logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import anthropic
from datetime import datetime

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

# ===== Conversation Memory =====
class ConversationMemory:
    """Manages conversation history for context-aware responses."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
    
    def add_exchange(self, user_message: str, bot_response: str, timestamp: Optional[str] = None):
        """Add a user-bot exchange to the conversation history."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        exchange = {
            "timestamp": timestamp,
            "user": user_message,
            "bot": bot_response
        }
        
        self.conversation_history.append(exchange)
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self, include_last_n: int = 5) -> str:
        """Get formatted conversation history for LLM context."""
        if not self.conversation_history:
            return ""
        
        # Get the last N exchanges
        recent_history = self.conversation_history[-include_last_n:]
        
        context_parts = ["Previous conversation:"]
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['bot']}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history = []
    
    def get_history_as_list(self) -> List[Dict[str, Any]]:
        """Return the conversation history as a list of dictionaries."""
        return self.conversation_history.copy()
    
    def load_history(self, history: List[Dict[str, Any]]):
        """Load conversation history from a list of dictionaries."""
        self.conversation_history = history[-self.max_history:] if history else []

# Global conversation memory instance
conversation_memory = ConversationMemory()

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

def answer_with_openai(question: str, context: str, conversation_context: str = "") -> str:
    # Build the user message with both product context and conversation history
    user_content = f"Question: {question}\n\nProduct Context:\n{context}"
    if conversation_context:
        user_content += f"\n\n{conversation_context}"
    user_content += "\n\nAnswer:"
    
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": user_content}
    ]
    resp = oai.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=450
    )
    return resp.choices[0].message.content

def answer_with_anthropic(question: str, context: str, conversation_context: str = "") -> str:
    # Build the user message with both product context and conversation history
    user_content = f"Question: {question}\n\nProduct Context:\n{context}"
    if conversation_context:
        user_content += f"\n\n{conversation_context}"
    user_content += "\n\nAnswer:"
    
    msg = anth.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=450,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content": user_content}]
    )
    return msg.content[0].text

def generate_answer(question: str, context: str, conversation_context: str = "") -> str:
    try:
        if PRIMARY_MODEL == "anthropic":
            return answer_with_anthropic(question, context, conversation_context)
        else:
            return answer_with_openai(question, context, conversation_context)
    except Exception:
        # fallback to the other provider
        try:
            if PRIMARY_MODEL == "anthropic":
                return answer_with_openai(question, context, conversation_context)
            else:
                return answer_with_anthropic(question, context, conversation_context)
        except Exception:
            # final fallback: extractive dump
            return ("Here's what I found:\n\n" + context[:1000]) if context else \
                   "I couldn't find enough context to answer confidently."

def answer(question: str, top_k: int = 10, limit: int = 5, provider: Optional[str] = None, 
          use_memory: bool = True, memory_instance: Optional[ConversationMemory] = None) -> str:
    """Single entry point used by both CLI and Netlify Function."""
    global PRIMARY_MODEL, conversation_memory
    if provider:
        PRIMARY_MODEL = provider

    # Use provided memory instance or global one
    memory = memory_instance if memory_instance else conversation_memory

    hits = retrieve(question, k=top_k)
    if not hits:
        return "I couldn't find anything relevant for this product."

    hits = boost(hits, question)
    ctx = build_context(hits, limit=limit)
    
    # Get conversation context if memory is enabled
    conversation_ctx = ""
    if use_memory and memory:
        conversation_ctx = memory.get_conversation_context()
    
    ans = generate_answer(question, ctx, conversation_ctx)
    
    # Add this exchange to memory if enabled
    if use_memory and memory:
        memory.add_exchange(question, ans)
    
    return ans

# ===== Convenience Functions for Memory Management =====

def get_conversation_history() -> List[Dict[str, Any]]:
    """Get the current conversation history as a list of dictionaries."""
    return conversation_memory.get_history_as_list()

def clear_conversation_history():
    """Clear the conversation history."""
    conversation_memory.clear_history()

def load_conversation_history(history: List[Dict[str, Any]]):
    """Load conversation history from a list of dictionaries."""
    conversation_memory.load_history(history)

def answer_with_custom_memory(question: str, memory_history: List[Dict[str, Any]], 
                             top_k: int = 10, limit: int = 5, provider: Optional[str] = None) -> str:
    """Answer a question using a custom conversation history."""
    custom_memory = ConversationMemory()
    custom_memory.load_history(memory_history)
    return answer(question, top_k, limit, provider, use_memory=True, memory_instance=custom_memory)

# ===== Example Usage =====
def example_conversation():
    """Example of how to use the conversation memory system."""
    print("=== Conversation Memory Example ===")
    
    # Clear any existing history
    clear_conversation_history()
    
    # First question
    q1 = "What is the price of this lipstick?"
    a1 = answer(q1)
    print(f"Q: {q1}")
    print(f"A: {a1}\n")
    
    # Follow-up question that references previous context
    q2 = "Is it worth the price?"
    a2 = answer(q2)
    print(f"Q: {q2}")
    print(f"A: {a2}\n")
    
    # Another follow-up
    q3 = "What about the longevity we discussed?"
    a3 = answer(q3)
    print(f"Q: {q3}")
    print(f"A: {a3}\n")
    
    # Show conversation history
    history = get_conversation_history()
    print("=== Conversation History ===")
    for i, exchange in enumerate(history, 1):
        print(f"Exchange {i}:")
        print(f"  User: {exchange['user']}")
        print(f"  Bot: {exchange['bot'][:100]}...")
        print(f"  Time: {exchange['timestamp']}\n")

if __name__ == "__main__":
    # Run example if script is executed directly
    example_conversation()
