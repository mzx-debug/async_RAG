#!/usr/bin/env python3
"""
Generate a well-distributed query set from msmarco_2k.jsonl corpus.

Strategy:
- Filter docs to those with clean noun-phrase titles (strict rules)
- Use the title directly as the topic — no text parsing
- Generate one question per doc, sampling templates to hit target distribution
- Target 512 queries: 50% short (≤20 words), 35% mid (21-60), 15% long (61+)
- Output: data/queries_generated.jsonl
"""

import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CORPUS_PATH = Path("E:/R1/async_rag_pipeline_v0/data/msmarco_2k.jsonl")
OUTPUT_PATH = Path("E:/R1/async_rag_pipeline_v0/data/queries_generated.jsonl")

# ── question templates ────────────────────────────────────────────────────────

SHORT_Q = [
    "What is {t}?",
    "What does {t} mean?",
    "Define {t}.",
    "What is {t} used for?",
    "What is the main purpose of {t}?",
    "How does {t} work?",
    "What causes {t}?",
    "What are the effects of {t}?",
    "Who or what is {t}?",
    "What role does {t} play?",
    "What is a key characteristic of {t}?",
    "What is the origin of {t}?",
    "Where does {t} come from?",
    "When was {t} first established?",
    "What distinguishes {t} from similar things?",
]

MID_Q = [
    "Explain the significance of {t} and describe its main characteristics, including how it works and why it matters.",
    "How does {t} function in practice, and what are its most important real-world applications or consequences?",
    "What are the key components of {t}, how do they interact with each other, and what outcomes do they produce?",
    "Describe the history and development of {t}, highlighting the major milestones and how understanding has evolved over time.",
    "What are the main advantages and disadvantages of {t}, and under what conditions does each outweigh the other?",
    "Compare {t} with closely related concepts, explain the key differences, and clarify when each is most relevant.",
    "What factors most strongly influence {t}, why does it matter, and how is it typically studied or measured?",
    "Explain the process involved in {t} step by step, and identify the most critical points where things can go wrong.",
    "What are the most important things to understand about {t}, and what common mistakes do people make when dealing with it?",
    "How has understanding of {t} changed over time, what is its current state, and what trends are shaping its future?",
    "What are the most common misconceptions about {t}, what does the evidence actually show, and why do the myths persist?",
    "Describe the relationship between {t} and its broader context, explaining how external factors shape and are shaped by it.",
    "What evidence best supports the importance of {t}, and how do experts evaluate or interpret that evidence today?",
    "How is {t} measured or evaluated in practice, what metrics are used, and what are the known limitations of those approaches?",
    "What challenges are most commonly associated with {t}, how have practitioners addressed them, and what remains unresolved?",
]

LONG_Q = [
    (
        "Provide a comprehensive overview of {t}, including its precise definition, "
        "historical background, key mechanisms or components, practical applications in "
        "real-world settings, and its current relevance or status. Where applicable, "
        "discuss any major controversies, open research questions, or competing "
        "interpretations that experts hold about it, and explain what evidence or "
        "reasoning supports the mainstream view."
    ),
    (
        "Write a detailed analytical essay on {t}. Begin with a clear definition "
        "and historical context, then explain how it functions or operates in practice. "
        "Identify who or what is most affected by it, describe the major debates or "
        "disagreements in the field, and summarize what leading experts currently "
        "recommend or believe. Conclude with the most important practical takeaways "
        "for someone encountering {t} for the first time."
    ),
    (
        "Explain {t} in depth across multiple dimensions: what it is and why it "
        "matters, how it developed over time, what its main components or stages are, "
        "what the available evidence says about its effectiveness or impact, and what "
        "practical implications follow from a thorough understanding of it. Also "
        "address common misconceptions and clarify what distinguishes {t} from "
        "closely related concepts."
    ),
    (
        "Give a thorough account of {t}, systematically addressing its definition "
        "and scope, its causes or origins, its effects or outcomes on individuals and "
        "broader systems, the methods commonly used to study or manage it, the key "
        "lessons learned from documented real-world examples, and any unresolved "
        "challenges that researchers or practitioners still face when dealing with it."
    ),
    (
        "Discuss {t} comprehensively: trace its historical development from early "
        "origins to the present day, explain the underlying principles or theories "
        "that govern it, describe how different stakeholders or communities are "
        "affected, outline the main approaches or strategies that have been taken to "
        "understand or address it, evaluate their relative effectiveness based on "
        "available evidence, and identify the most promising directions for future "
        "work or policy in this area."
    ),
]

# ── title filtering ───────────────────────────────────────────────────────────

_QUESTION_START = re.compile(
    r"^(how|what|where|when|why|who|which|can|does|is|are|do|will|should|was|were"
    r"|define|list|name|give|find|tell|show|explain|describe|compare|calculate"
    r"|difference|advantages|disadvantages|benefits|types|examples)\b",
    re.IGNORECASE,
)
_DANGLING_END = frozenset(
    "the a an of in on at to and or for with by from into about near "
    "between among through during after before since".split()
)
_BAD_CHARS = re.compile(r"[#@\[\]{}<>|\\^~`\xe2]")
_DIGITS = re.compile(r"\d")
_GENERIC_TITLES = frozenset(
    "overview introduction biography anatomy summary conclusion "
    "background history references notes contents abstract".split()
)


def clean_title(raw: str) -> Optional[str]:
    """
    Return a cleaned title string if it passes all quality checks, else None.
    """
    t = unicodedata.normalize("NFKC", raw).strip()
    if not t or t == "-":
        return None

    # Fix common mojibake (Windows-1252 mis-decoded as Latin-1)
    t = t.replace("\u00e2\u0080\u0099", "'").replace("\u00e2\u0080\u009c", '"') \
         .replace("\u00e2\u0080\u009d", '"').replace("\u00e2\u0080\u0093", "-") \
         .replace("\u00e2\u0080\u0094", "-")
    # Drop any remaining non-ASCII-printable characters
    t = re.sub(r"[^\x20-\x7e]", "", t).strip()
    t = re.sub(r"\s*[\(\[].*", "", t).strip(" :-–?!.,;")
    if not t:
        return None

    # Reject question/imperative starts
    if _QUESTION_START.match(t):
        return None

    # Reject titles with digits (ZIP codes, IDs, version numbers, etc.)
    if _DIGITS.search(t):
        return None

    # Reject titles with special characters
    if _BAD_CHARS.search(t):
        return None

    # Reject URLs
    if "http" in t or "www." in t:
        return None

    words = t.split()

    # Must be 1–7 words
    if len(words) < 1 or len(words) > 7:
        return None

    # Reject single generic words
    if len(words) == 1 and words[0].lower() in _GENERIC_TITLES:
        return None

    # Reject if last word is a dangling function word
    if words[-1].lower() in _DANGLING_END:
        return None

    # Reject if it looks like a sentence (contains a conjugated verb mid-phrase)
    _VERB_MID = re.compile(
        r"\b(is|was|are|were|has|have|had|does|do|did|will|would|can|could"
        r"|should|may|might|must|shall|need|dare|used to)\b",
        re.IGNORECASE,
    )
    if _VERB_MID.search(" ".join(words[1:])):  # allow first word to be a noun
        return None

    return t


# ── bucket / word count ───────────────────────────────────────────────────────

def word_count(s: str) -> int:
    return len(s.split())


def bucket_of(q: str) -> str:
    wc = word_count(q)
    if wc <= 20:
        return "short"
    if wc <= 60:
        return "mid"
    return "long"


# ── main ──────────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> List[Dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def generate_queries(
    docs: List[Dict],
    target: int = 512,
    seed: int = 42,
    short_ratio: float = 0.50,
    mid_ratio: float = 0.35,
    long_ratio: float = 0.15,
) -> List[Dict]:
    rng = random.Random(seed)

    n_short = int(target * short_ratio)
    n_mid   = int(target * mid_ratio)
    n_long  = target - n_short - n_mid
    targets = {"short": n_short, "mid": n_mid, "long": n_long}

    # Build candidate list: (clean_title, doc_idx)
    candidates = []
    for i, doc in enumerate(docs):
        title = clean_title(doc.get("title", ""))
        if title:
            candidates.append((title, i))

    print(f"  Docs with clean title: {len(candidates)} / {len(docs)}")

    if len(candidates) < target:
        print(f"  WARNING: only {len(candidates)} clean titles, target={target}. "
              f"Will repeat pool.")

    # Repeat pool enough times to fill target
    pool = candidates * max(1, (target * 2 // max(len(candidates), 1)) + 1)
    rng.shuffle(pool)

    counters: Dict[str, int] = {"short": 0, "mid": 0, "long": 0}
    queries: List[Dict] = []
    seen: set = set()

    for title, doc_idx in pool:
        if all(counters[b] >= targets[b] for b in targets):
            break

        remaining = {b: max(0, targets[b] - counters[b]) for b in targets}
        weights = [remaining["short"], remaining["mid"], remaining["long"]]
        if sum(weights) == 0:
            break

        chosen = rng.choices(["short", "mid", "long"], weights=weights, k=1)[0]

        if chosen == "short":
            tmpl = rng.choice(SHORT_Q)
        elif chosen == "mid":
            tmpl = rng.choice(MID_Q)
        else:
            tmpl = rng.choice(LONG_Q)

        question = tmpl.format(t=title)

        if bucket_of(question) != chosen:
            continue

        key = question.lower()
        if key in seen:
            continue
        seen.add(key)

        queries.append({
            "id": f"{chosen}_{counters[chosen]:04d}",
            "question": question,
            "bucket_hint": chosen,
            "source_doc_idx": doc_idx,
            "topic": title,
        })
        counters[chosen] += 1

    # Shuffle and re-assign IDs
    rng.shuffle(queries)
    id_counters: Dict[str, int] = {"short": 0, "mid": 0, "long": 0}
    for q in queries:
        b = q["bucket_hint"]
        q["id"] = f"{b}_{id_counters[b]:04d}"
        id_counters[b] += 1

    return queries


def main() -> None:
    print(f"Loading corpus from {CORPUS_PATH} ...")
    docs = load_corpus(CORPUS_PATH)
    print(f"  {len(docs)} documents loaded.")

    queries = generate_queries(docs, target=512)

    from collections import Counter
    dist = Counter(q["bucket_hint"] for q in queries)
    total = len(queries)
    print(f"\nGenerated {total} queries:")
    for b in ("short", "mid", "long"):
        print(f"  {b:5s} : {dist[b]:4d}  ({dist[b]/total*100:.1f}%)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print 3 samples per bucket
    by_bucket: Dict[str, List] = defaultdict(list)
    for q in queries:
        by_bucket[q["bucket_hint"]].append(q)

    print("\n── Sample queries ──")
    for b in ("short", "mid", "long"):
        print(f"\n[{b}]")
        for q in by_bucket[b][:4]:
            print(f"  [{q['id']}] {q['question']}")


if __name__ == "__main__":
    main()
