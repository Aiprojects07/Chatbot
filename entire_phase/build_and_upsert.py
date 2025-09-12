import os, json, uuid
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
FILES = {
    # Use the structured ULAS (Ultimate Lipstick Attribute System) JSON instead of the markdown file
    "attributes": DATA_DIR/"Ultimate_Lipstick_Attribute_System_Indian.json",
    "qa":         DATA_DIR/"Q&A Glossier Generation G Lipstick in Jam.json",
    "snapshot":   DATA_DIR/"Snapshot Glossier Generation G Lipstick in Jam.json",
}

PRODUCT_ID = "glossier_generation_g_jam"

# === Config ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "kult-beauty-jam")
EMBED_MODEL      = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072 dims

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc  = Pinecone(api_key=PINECONE_API_KEY)

# === Helpers ===
def load_any(path: Path):
    """Load JSON if possible; otherwise return raw text (Markdown) wrapped.
    Returns dict/list for JSON; else {"_markdown": <text>}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    stripped = data.lstrip()
    if stripped and stripped[0] in "{[":
        try:
            return json.loads(data)
        except Exception:
            pass
    return {"_markdown": data}

# Loader for the new attributes JSON
def load_ulais_json(path: Path):
    """Load the 'Ultimate Lipstick Attribute System - Indian Market Focus' JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mk_id(prefix="c"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def atomic_fact(field, value, aliases=None, source_path=None):
    aliases = aliases or []
    text = f"DocType: AtomicFact | Product: Glossier Generation G – Jam | Field: {field} | Value: {value}"
    if aliases:
        text += f" | Aliases: {aliases}"
    md = {
        "product_id": PRODUCT_ID,
        "doc_type": "atomic_fact",
        "topic": [str(field).lower().replace(" ","_")],
        "source": "snapshot"
    }
    if source_path:
        md["source_path"] = source_path
    return {
        "id": mk_id("fact"),
        "text": text,
        "metadata": md
    }

# ---- Snapshot (section_6) scalar/list/dict walker ----
def chunk_snapshot_section6(snapshot_dict):
    chunks = []

    def emit_atomic(path, value, aliases=None):
        field = " / ".join(path)
        chunks.append(atomic_fact(field, value, aliases=aliases, source_path="/".join(path)))

    def walk(obj, path):
        # Scalar
        if isinstance(obj, (str, int, float)) and str(obj).strip():
            aliases = None
            last = (path[-1] if path else "").lower()
            if "transfer" in last and "high" in str(obj).lower():
                aliases = ["not mask-proof", "not kiss-proof"]
            emit_atomic(path, str(obj), aliases)
            return

        # List
        if isinstance(obj, list):
            for idx, item in enumerate(obj, 1):
                if isinstance(item, (str, int, float)):
                    emit_atomic(path + [f"item_{idx}"], str(item))
                elif isinstance(item, dict):
                    walk(item, path + [f"item_{idx}"])
            return

        # Dict
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, path + [str(k).replace(" ", "_")])

    walk(snapshot_dict, ["Snapshot", "Quick_Reference"])
    # One compact summary card (safe even if some fields missing)
    desc = (
        "DocType: SnapshotSummary | Product: Glossier Generation G – Jam | "
        f"Finish: {snapshot_dict.get('Finish','')} | "
        f"Opacity: {snapshot_dict.get('Opacity','')} | "
        f"Wear Time: {snapshot_dict.get('Wear_Time','')} | "
        f"Transfer Level: {snapshot_dict.get('Transfer_Level','')} | "
        "Includes BEST_FOR, SKIP_IF, Pros/Cons if present."
    )
    chunks.append({
        "id": mk_id("snap"),
        "text": desc,
        "metadata": {
            "product_id": PRODUCT_ID,
            "doc_type": "snapshot",
            "topic": ["summary"],
            "source_path": "Snapshot/Quick_Reference"
        }
    })
    return chunks

# ---- Snapshot (sections 4–7) full coverage ----
def chunk_snapshot_all(snapshot_doc):
    chunks = []

    # §4: Pros/Cons/Best/Skip (lists → one mini-chunk per bullet)
    sec4 = snapshot_doc.get("section_4", {}).get("format", {})
    for label, items in (sec4.items() if isinstance(sec4, dict) else []):
        if isinstance(items, list):
            for idx, item in enumerate(items, 1):
                chunks.append({
                    "id": mk_id("snap4"),
                    "text": f"DocType: Verdict | Category: {label} | Item: {item}",
                    "metadata": {
                        "product_id": PRODUCT_ID,
                        "doc_type": "snapshot",
                        "topic": [label.lower()],
                        "section": "section_4",
                        "source_path": f"Snapshot/section_4/{label}/item_{idx}"
                    }
                })

    # §5: User Consensus (paragraph → one chunk)
    consensus = snapshot_doc.get("section_5", {}).get("content")
    if consensus:
        chunks.append({
            "id": mk_id("snap5"),
            "text": f"DocType: UserConsensus | Content: {consensus}",
            "metadata": {
                "product_id": PRODUCT_ID,
                "doc_type": "snapshot",
                "topic": ["consensus"],
                "section": "section_5",
                "source_path": "Snapshot/section_5/content"
            }
        })

    # §6: Quick Reference (scalar/list/dict walker + summary)
    sec6 = snapshot_doc.get("section_6", {}).get("snapshot", {})
    if isinstance(sec6, dict) and sec6:
        chunks += chunk_snapshot_section6(sec6)

    # §7: Research Sources (list → one source chunk per item)
    sources = snapshot_doc.get("section_7", {}).get("sources", [])
    if isinstance(sources, list):
        for i, src in enumerate(sources, 1):
            chunks.append({
                "id": mk_id("snap7"),
                "text": f"DocType: ResearchSource | Source: {src}",
                "metadata": {
                    "product_id": PRODUCT_ID,
                    "doc_type": "snapshot",
                    "topic": ["sources"],
                    "section": "section_7",
                    "source_path": f"Snapshot/section_7/sources/item_{i}"
                }
            })

    return chunks

# ---- Q&A (dynamic over all sections) ----
def chunk_faq(qa_json):
    chunks = []
    root = qa_json.get("glossier_generation_g_jam_complete_qa", qa_json)

    def add(item, topic="general", section="general"):
        q = (item.get("Q","") or "").strip()
        a = (item.get("A","") or "").strip()
        if not (q or a):
            return
        why = item.get("WHY")
        sol = item.get("SOLUTION")
        header = "DocType: FAQ | Product: Glossier Generation G – Jam"
        body = f"Q: {q}\nA: {a}"
        if why: body += f"\nWHY: {why}"
        if sol: body += f"\nSOLUTION: {sol}"
        text = f"{header} | Section: {section} | Topic: {topic}\n{body}"
        chunks.append({
            "id": mk_id("faq"),
            "text": text,
            "metadata": {
                "product_id": PRODUCT_ID,
                "doc_type":"faq",
                "topic":[topic],
                "section":section,
                "source_path": f"Q&A/{section}"
            }
        })

    for sec_key, sec_val in root.items():
        section = str(sec_key)
        if isinstance(sec_val, list):
            for item in sec_val:
                if isinstance(item, dict):
                    add(item, topic=section, section=section)
        elif isinstance(sec_val, dict):
            for sub_key, sub_list in sec_val.items():
                sub_section = f"{section}/{sub_key}"
                if isinstance(sub_list, list):
                    for item in sub_list:
                        if isinstance(item, dict):
                            add(item, topic=sub_section, section=sub_section)
    return chunks

import re
from collections import Counter

# ---- tiny helpers ----
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokens(s: str):
    return [w.lower() for w in _WORD_RE.findall(str(s or "")) if w]

def _top_keywords(obj, limit=20):
    """Collect frequent words from a subtree for weak topic signals."""
    bag = Counter()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                bag.update(_tokens(k))
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
        else:
            bag.update(_tokens(o))
    walk(obj)
    stop = {
        "and","or","of","for","the","to","in","on","with","by","at","an","a","vs","vs.",
        "item","list","data","info","section","type"
    }
    return [w for w,_ in bag.most_common() if w not in stop and len(w) >= 3][:limit]

def _pretty(k: str) -> str:
    return str(k).replace("_"," ").replace("-"," ").strip().title()

# ---- GENERALIST ULAS JSON GLOSSARY CHUNKER ----
def chunk_glossary(ulas_json: dict, topic_overrides: dict | None = None):
    """
    Parse the 'Ultimate Lipstick Attribute System - Indian Market Focus' JSON into glossary chunks.
    - Walks ALL dicts/lists/scalars (no data skipped)
    - Auto-generates topic tags from section key + subtree content (no hard-coded buckets)
    - Preserves a precise source_path using ORIGINAL keys
    - Produces compact, readable headings from prettified keys
    """
    chunks = []
    seen = set()  # to avoid duplicate (source_path, text)

    topic_overrides = topic_overrides or {}

    # Build an automatic per-section topic map (top-level keys only)
    def build_topic_map_auto(root: dict, max_tags_per_section=6):
        topic_map = {}
        if not isinstance(root, dict): 
            return topic_map
        for key, node in root.items():
            if key in ("title","note","_topic_aliases"):
                continue
            # base: tokens from key
            tags = list(dict.fromkeys(_tokens(key)))
            # content signals
            tags += _top_keywords(node, limit=15)
            # clean + trim
            seen_local = set()
            clean = []
            for t in tags:
                if t not in seen_local and len(t) > 2:
                    clean.append(t); seen_local.add(t)
            topic_map[key] = clean[:max_tags_per_section]
        # merge explicit overrides if present in JSON or passed in
        json_overrides = root.get("_topic_aliases") if isinstance(root, dict) else None
        for k, aliases in (json_overrides or {}).items():
            topic_map[k] = list(dict.fromkeys(aliases))[:max_tags_per_section]
        for k, aliases in topic_overrides.items():
            topic_map[k] = list(dict.fromkeys(aliases))[:max_tags_per_section]
        return topic_map

    topic_map = build_topic_map_auto(ulas_json)

    def auto_topics(top_key: str, path_tokens: list[str], leaf_text: str):
        """Pick tags from: overrides/auto-map + path/leaf weak signals."""
        base = list(topic_map.get(top_key, []))  # start from auto map for that section
        # add path tokens + leaf tokens (lightly)
        bag = Counter(_tokens(" ".join(path_tokens) + " " + str(leaf_text)))
        for w, _ in bag.most_common(6):
            if w not in base and len(base) < 8 and len(w) >= 3:
                base.append(w)
        # trim for sanity
        return base[:6] if len(base) > 6 else base

    def emit(heading_path_human: str, note_text: str, top_key: str, source_path_keys: list[str]):
        text = str(note_text).strip()
        if not text:
            return
        spath = "/".join(source_path_keys)
        key = (spath, text)
        if key in seen:
            return
        seen.add(key)

        topics = auto_topics(top_key, source_path_keys, text)
        chunks.append({
            "id": mk_id("gloss"),
            "text": f"DocType: Glossary | Heading: {heading_path_human or 'Attributes'} | Notes: {text}",
            "metadata": {
                "product_id": PRODUCT_ID,
                "doc_type": "glossary",
                "topic": topics or ["attributes"],
                "source_path": spath
            }
        })

    # Walk everything
    def walk(obj, raw_path: list[str], human_h1=None, human_h2=None, human_h3=None, top_key="attributes"):
        # human heading (nice for UI)
        heading_parts = [p for p in [human_h1, human_h2, human_h3] if p]
        heading_human = " / ".join(heading_parts)

        if isinstance(obj, dict):
            for k, v in obj.items():
                # skip meta keys anywhere
                if k in ("_topic_aliases",):
                    continue
                # keep 'title'/'note' only if they are the ENTIRE root; otherwise they are meta
                if k in ("title","note") and len(raw_path) == 0:
                    continue

                new_raw = raw_path + [str(k)]
                # identify top section key at depth 1 (under root)
                if len(raw_path) == 0:
                    walk(v, new_raw, _pretty(k), None, None, top_key=k)
                elif human_h2 is None:
                    walk(v, new_raw, human_h1, _pretty(k), None, top_key=top_key)
                else:
                    walk(v, new_raw, human_h1, human_h2, _pretty(k), top_key=top_key)
            return

        if isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                new_raw = raw_path + [f"item_{i}"]
                if isinstance(item, (str, int, float, bool)) or item is None:
                    emit(heading_human, item, top_key, new_raw)
                else:
                    walk(item, new_raw, human_h1, human_h2, human_h3, top_key=top_key)
            return

        # scalar leaf
        emit(heading_human, obj, top_key, raw_path)

    # start at root
    walk(ulas_json, [], top_key="attributes")

    return chunks

# Back-compat alias to match requested name
chunk_glossary_json = chunk_glossary

# --- new: simple Markdown -> glossary chunker ---
def chunk_glossary_md(md_text: str, max_items=2000):
    """
    Turn a Markdown taxonomy (headings + bullet lists) into glossary chunks.
    Each bullet becomes one compact chunk under its nearest heading path.
    """
    chunks = []
    lines = md_text.splitlines()

    h1 = h2 = h3 = h4 = h5 = h6 = None
    items = 0

    def heading_path():
        return " / ".join([p for p in [h1, h2, h3, h4, h5, h6] if p])

    def emit_note(heading, note, path_extra=None):
        nonlocal items
        if items >= max_items:
            return
        hp = heading_path()
        if path_extra:
            hp = f"{hp}/{path_extra}" if hp else path_extra
        chunks.append({
            "id": mk_id("gloss"),
            "text": f"DocType: Glossary | Heading: {heading or (hp or 'General')} | Notes: {note}",
            "metadata": {
                "product_id": PRODUCT_ID,
                "doc_type": "glossary",
                "topic": ["attributes","taxonomy"],
                "source_path": f"AttributesMD/{hp}"
            }
        })
        items += 1

    import re
    # Bullets: '-', '*', '+' and numbered '1.' or '1)'
    bullet_re = re.compile(r"^\s*(?:[-*+]|\d+[\.)])\s+(.*)$")
    h1_re = re.compile(r"^\s*#\s+(.*)$")
    h2_re = re.compile(r"^\s*##\s+(.*)$")
    h3_re = re.compile(r"^\s*###\s+(.*)$")
    h4_re = re.compile(r"^\s*####\s+(.*)$")
    h5_re = re.compile(r"^\s*#####\s+(.*)$")
    h6_re = re.compile(r"^\s*######\s+(.*)$")

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        if h1_re.match(line):
            h1 = h1_re.match(line).group(1).strip()
            h2 = h3 = h4 = h5 = h6 = None
            continue
        if h2_re.match(line):
            h2 = h2_re.match(line).group(1).strip()
            h3 = h4 = h5 = h6 = None
            continue
        if h3_re.match(line):
            h3 = h3_re.match(line).group(1).strip()
            h4 = h5 = h6 = None
            continue
        if h4_re.match(line):
            h4 = h4_re.match(line).group(1).strip()
            h5 = h6 = None
            continue
        if h5_re.match(line):
            h5 = h5_re.match(line).group(1).strip()
            h6 = None
            continue
        if h6_re.match(line):
            h6 = h6_re.match(line).group(1).strip()
            continue

        m = bullet_re.match(line)
        if m:
            bullet = m.group(1).strip()
            # Use the deepest heading as the "Heading", bullet as "Notes"
            emit_note(heading=h3 or h2 or h1, note=bullet, path_extra="bullet")
            continue

        # Plain paragraph lines under a heading (or before any heading → General)
        if any([h1,h2,h3,h4,h5,h6]):
            emit_note(heading=h6 or h5 or h4 or h3 or h2 or h1, note=line, path_extra="text")
        else:
            # Lines before first heading
            emit_note(heading="General", note=line, path_extra="text")

    # Small curated helpers if present in text
    text_lower = md_text.lower()
    if "mac" in text_lower and "nc" in text_lower:
        chunks.append({
            "id": mk_id("gloss"),
            "text": "DocType: Glossary | Heading: Indian Skin Tone Mapping | Notes: MAC NC15–NC50 mapping for Indian regions; terms like wheatish/golden-brown.",
            "metadata": {"product_id": PRODUCT_ID, "doc_type":"glossary",
                         "topic":["skin_tone","mac_mapping"], "source_path":"AttributesMD/skin_tone_mapping"}
        })

    return chunks

# ---- Build corpus ----
def build_corpus():
    # Prefer structured ULAS JSON for attributes
    attrs   = load_ulais_json(FILES["attributes"])   
    qa      = load_any(FILES["qa"])           # robust JSON loader
    snapdoc = load_any(FILES["snapshot"])     # robust JSON loader

    corpus = []
    # Q&A
    qa_obj = qa if isinstance(qa, dict) else {}
    corpus += chunk_faq(qa_obj)

    # Snapshot (sections 4–7 + 6 walker)
    snap_obj = snapdoc if isinstance(snapdoc, dict) else {}
    corpus += chunk_snapshot_all(snap_obj)

    # Attributes: structured ULAS JSON
    corpus += chunk_glossary_json(attrs)

    return corpus

def embed_texts(texts):
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ensure_index():
    # Create if missing
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=3072,            # text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def upsert_chunks(chunks, batch_size=100):
    index = pc.Index(PINECONE_INDEX)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = embed_texts([c["text"] for c in batch])
        vectors = [{
            "id": batch[j]["id"],
            "values": embeddings[j],
            "metadata": {
                **batch[j]["metadata"],
                "text": batch[j]["text"]  # keep full text in metadata for quick render
            }
        } for j in range(len(batch))]
        index.upsert(vectors=vectors)
        print(f"Upserted {i+len(batch)}/{len(chunks)}")

if __name__ == "__main__":
    print("Building chunks…")
    corpus = build_corpus()
    print(f"Chunks: {len(corpus)}")
    from collections import Counter
    print("Doc types:", Counter([c["metadata"]["doc_type"] for c in corpus]))
    ensure_index()
    upsert_chunks(corpus)
    print("Done.")
