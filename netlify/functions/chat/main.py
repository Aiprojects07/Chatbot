# netlify/functions/chat/main.py
import json
import os
import sys
from pathlib import Path

# Locate and add the 'Qna chatbot' directory to sys.path in a way that works
# both locally (repo layout) and in the Netlify bundle.
here = Path(__file__).resolve()
candidates = [
    here.parent,                 # .../chat
    here.parent.parent,          # .../functions  (Netlify bundle usually has Qna chatbot at parents[1])
    here.parent.parent.parent,   # .../netlify
    here.parent.parent.parent.parent,  # project root in local repo
]
for base in candidates:
    qna_dir = base / "Qna chatbot"
    if qna_dir.exists():
        sys.path.append(str(qna_dir))
        break

from core.single_product import answer, answer_with_custom_memory

def _cors_headers():
    return {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET, POST, OPTIONS",
        # Be permissive for local testing; adjust as needed
        "access-control-allow-headers": "*",
        "access-control-max-age": "86400",
        "content-type": "application/json",
        "vary": "Origin",
    }

def _json_response(status, payload):
    return {
        "statusCode": status,
        "headers": _cors_headers(),
        "body": json.dumps(payload, ensure_ascii=False)
    }

def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        # CORS preflight
        return {
            "statusCode": 204,
            "headers": _cors_headers(),
            "body": ""
        }

    if event.get("httpMethod") != "POST":
        return _json_response(405, {"error": "Method not allowed. Use POST."})

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        return _json_response(400, {"error": "Invalid JSON body"})

    q = (body.get("question") or "").strip()
    if not q:
        return _json_response(400, {"error": "Missing 'question' in body"})

    top_k  = int(body.get("top_k", 10))
    limit  = int(body.get("limit", 5))
    prov   = body.get("provider")  # "anthropic" | "openai" | None
    # Optional: client-managed session and memory
    session_id = body.get("sessionId")
    history = body.get("history")

    try:
        # If client passes a valid history list of dicts, use it
        if isinstance(history, list) and all(isinstance(x, dict) for x in history):
            ans = answer_with_custom_memory(q, history, top_k=top_k, limit=limit, provider=prov)
        else:
            ans = answer(q, top_k=top_k, limit=limit, provider=prov)
        # Echo sessionId if provided (useful for clients)
        return _json_response(200, {"answer": ans, "sessionId": session_id})
    except Exception as e:
        # Optional: log e
        return _json_response(500, {"error": "Failed to answer", "detail": str(e)})
