# netlify/functions/chat/main.py
import json
import os
import sys
from pathlib import Path

# Add the path to import from Qna chatbot directory
# __file__ -> .../netlify/functions/chat/main.py
# parent (chat) -> parent (functions) -> parent (netlify) -> parent (project root)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "Qna chatbot"))

from core.single_product import answer, answer_with_custom_memory

def _json_response(status, payload):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",  # allow simple CORS for testing/frontends
        },
        "body": json.dumps(payload, ensure_ascii=False)
    }

def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        # CORS preflight (if you build a frontend that calls this)
        return {
            "statusCode": 204,
            "headers": {
                "access-control-allow-origin": "*",
                "access-control-allow-methods": "POST, OPTIONS",
                "access-control-allow-headers": "content-type, authorization",
            },
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
