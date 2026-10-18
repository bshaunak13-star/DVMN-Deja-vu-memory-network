"""
DVMN-0.1 — Déjà Vu Memory Network
Python 3.14-compatible, standard-library-only edition.

No PyTorch, NumPy, or sentence-transformers required.
Uses a transparent lexical similarity score as a baseline research prototype.
Optional local generation through Ollama is supported if Ollama is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "dvmn_memory.db"
DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "as",
    "is", "am", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "my", "our", "your", "their", "me", "us", "them", "do", "does", "did",
    "can", "could", "would", "should", "will", "may", "might", "have", "has",
    "had", "not", "no", "yes", "what", "when", "where", "why", "how", "which",
    "who", "about", "into", "through", "using", "use", "make", "made", "want",
    "need", "like", "also", "very", "more", "most", "some", "any", "each",
    "new", "old", "help", "please", "explain", "tell", "give", "show",
}

CONCEPT_GROUPS = {
    "memory": {"memory", "remember", "recall", "retrieval", "episodic", "context", "forget"},
    "language_model": {"llm", "language", "model", "transformer", "prompt", "token", "generation"},
    "neural_network": {"neural", "neuron", "neurons", "network", "synapse", "spiking", "snn"},
    "learning": {"learn", "learning", "training", "train", "optimization", "optimize", "gradient"},
    "robotics": {"robot", "robotics", "navigation", "autonomous", "sensor", "motor"},
    "vision": {"vision", "image", "visual", "camera", "object", "detection", "opencv", "yolo"},
    "reasoning": {"reason", "reasoning", "logic", "problem", "solve", "inference", "deduction"},
    "similarity": {"similar", "similarity", "semantic", "embedding", "vector", "cosine", "familiarity"},
    "database": {"database", "sqlite", "sql", "storage", "store", "record", "index"},
    "evaluation": {"evaluation", "evaluate", "test", "testing", "benchmark", "metric", "accuracy"},
    "biology": {"biological", "biology", "brain", "fly", "fruit", "cell", "cortical"},
    "reinforcement_learning": {"reinforcement", "reward", "policy", "rlvr", "rlhf"},
}

RELATION_PATTERNS = [
    (r"\b(\w+)\s+(?:helps|help|improves|improve|supports|support)\s+(\w+)", "helps"),
    (r"\b(\w+)\s+(?:uses|use|utilizes|utilize)\s+(\w+)", "uses"),
    (r"\b(\w+)\s+(?:stores|store|remembers|remember|retrieves|retrieve)\s+(\w+)", "handles"),
    (r"\b(\w+)\s+(?:causes|cause|creates|create|produces|produce)\s+(\w+)", "produces"),
    (r"\b(\w+)\s+(?:is|are)\s+(?:based on|built on)\s+(\w+(?:\s+\w+)?)", "based_on"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tokenize(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def extract_concepts(text: str) -> set[str]:
    tokens = set(tokenize(text))
    concepts = {f"term:{t}" for t in tokens}
    for concept, words in CONCEPT_GROUPS.items():
        if tokens & words:
            concepts.add(concept)
    return concepts


def extract_relations(text: str) -> set[str]:
    found: set[str] = set()
    lowered = text.lower()
    for pattern, relation in RELATION_PATTERNS:
        for left, right in re.findall(pattern, lowered):
            left, right = left.strip(), right.strip()
            if left not in STOP_WORDS and right not in STOP_WORDS:
                found.add(f"{left}:{relation}:{right}")
    # Preserve common conceptual links even when phrased without a pattern.
    tokens = set(tokenize(text))
    if {"memory", "context"} <= tokens:
        found.add("memory:relates_to:context")
    if {"neuron", "robot"} <= tokens or {"neurons", "robot"} <= tokens:
        found.add("neural_network:relates_to:robotics")
    if {"embedding", "similarity"} <= tokens:
        found.add("embedding:measures:similarity")
    return found


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_cosine(a: str, b: str) -> float:
    """Cosine similarity over token-count vectors; no third-party packages."""
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) | set(cb)
    dot = sum(ca[t] * cb[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def familiarity(query: str, episode: dict[str, Any]) -> dict[str, float]:
    sem = lexical_cosine(query, episode["content"])
    concept = jaccard(extract_concepts(query), set(episode["concepts"]))
    relation = jaccard(extract_relations(query), set(episode["relations"]))
    score = 0.50 * sem + 0.30 * concept + 0.20 * relation
    return {"score": score, "lexical": sem, "concept": concept, "relation": relation}


class DVMN:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _initialize(self) -> None:
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    concepts_json TEXT NOT NULL,
                    relations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user'
                )
            """)

    def add(self, title: str, content: str, source: str = "user") -> int:
        title, content = title.strip(), content.strip()
        if not content:
            raise ValueError("Episode content cannot be empty.")
        concepts = sorted(extract_concepts(title + " " + content))
        relations = sorted(extract_relations(title + " " + content))
        with self.connect() as con:
            cur = con.execute(
                """INSERT INTO episodes
                   (title, content, concepts_json, relations_json, created_at, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (title or "Untitled episode", content, json.dumps(concepts),
                 json.dumps(relations), now_iso(), source),
            )
            return int(cur.lastrowid)

    def episodes(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM episodes ORDER BY id DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["concepts"] = json.loads(item.pop("concepts_json"))
            item["relations"] = json.loads(item.pop("relations_json"))
            result.append(item)
        return result

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        ranked = []
        for episode in self.episodes():
            scores = familiarity(query, episode)
            ranked.append({**episode, **scores})
        ranked.sort(key=lambda item: (item["score"], item["id"]), reverse=True)
        return ranked[:max(1, limit)]

    def seed_demo(self) -> int:
        demos = [
            ("Fly vision and neural circuits",
             "A fruit fly uses compact neural circuits to process visual motion while flying. "
             "Neural activity can connect sensory input with navigation and motor responses."),
            ("Long-term language-model memory",
             "An AI memory system stores conversation episodes and retrieves relevant context "
             "when a new discussion activates related concepts. Similarity and relations help recall."),
            ("Robot perception with spiking neurons",
             "A spiking neural network may process camera signals for robot vision and navigation. "
             "Neural spikes can encode events and support low-latency decisions."),
            ("Evaluating memory retrieval",
             "A benchmark tests whether a memory system retrieves the right past episode. "
             "Measure precision at k, recall at k, and ranking quality on labelled queries."),
        ]
        count = 0
        existing = {e["title"] for e in self.episodes()}
        for title, content in demos:
            if title not in existing:
                self.add(title, content, source="demo")
                count += 1
        return count

    def chat(self, prompt: str, limit: int = 3, model: str = DEFAULT_MODEL) -> str:
        memories = self.recall(prompt, limit)
        useful = [m for m in memories if m["score"] >= 0.08]
        context = "\n\n".join(
            f"[Episode {m['id']}: {m['title']} | familiarity={m['score']:.3f}]\n{m['content']}"
            for m in useful
        ) or "No strongly matching past episode was found."
        system_prompt = (
            "You are the reasoning component of DVMN, a memory research prototype. "
            "Answer the user's question clearly. Use retrieved episodes only when relevant. "
            "Distinguish recalled information from new reasoning; do not claim that a low "
            "familiarity score proves a human-like deja-vu experience.\n\n"
            f"Retrieved past episodes:\n{context}\n\nUser message: {prompt}"
        )
        try:
            payload = json.dumps({
                "model": model, "prompt": system_prompt, "stream": False
            }).encode("utf-8")
            request = urllib.request.Request(
                OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "").strip() or "Ollama returned an empty response."
        except urllib.error.URLError as exc:
            return (
                "Could not connect to Ollama. Start Ollama and run "
                f"`ollama pull {model}` first.\n\n"
                f"Retrieved-memory context:\n{context}\n\n"
                f"Your message: {prompt}\n\nConnection detail: {exc}"
            )


def command_init(dvmn: DVMN, _args: argparse.Namespace) -> None:
    print(f"Database ready: {dvmn.db_path}")


def command_seed(dvmn: DVMN, _args: argparse.Namespace) -> None:
    count = dvmn.seed_demo()
    print(f"Added {count} demo episode(s). Total episodes: {len(dvmn.episodes())}")


def command_add(dvmn: DVMN, args: argparse.Namespace) -> None:
    title = args.title or input("Episode title: ").strip()
    content = args.content
    if content is None:
        print("Enter episode content. Finish with a line containing only .END")
        lines = []
        while True:
            line = input()
            if line.strip() == ".END":
                break
            lines.append(line)
        content = "\n".join(lines)
    episode_id = dvmn.add(title, content)
    print(f"Saved episode #{episode_id}: {title or 'Untitled episode'}")


def command_list(dvmn: DVMN, _args: argparse.Namespace) -> None:
    episodes = dvmn.episodes()
    if not episodes:
        print("No memories yet. Run: python main.py seed-demo")
        return
    for e in sorted(episodes, key=lambda item: item["id"]):
        print(f"\n#{e['id']} — {e['title']} [{e['source']}]")
        print(f"  Created: {e['created_at']}")
        print(f"  Concepts: {', '.join(e['concepts'][:12]) or '(none)'}")
        print(f"  Content: {e['content'][:240]}")


def command_recall(dvmn: DVMN, args: argparse.Namespace) -> None:
    results = dvmn.recall(args.query, args.limit)
    if not results:
        print("No episodes stored.")
        return
    print(f"\nQuery: {args.query}\n")
    for rank, item in enumerate(results, 1):
        print(f"{rank}. [{item['score']:.3f}] {item['title']} (episode #{item['id']})")
        print(
            f"   lexical={item['lexical']:.3f} concept={item['concept']:.3f} "
            f"relation={item['relation']:.3f}"
        )
        print(f"   {item['content'][:350]}\n")


def command_chat(dvmn: DVMN, args: argparse.Namespace) -> None:
    prompt = args.prompt or input("You: ").strip()
    print("\nDVMN:\n" + dvmn.chat(prompt, args.limit, args.model))


def command_evaluate(dvmn: DVMN, _args: argparse.Namespace) -> None:
    """Small smoke test, not a scientific benchmark."""
    if not dvmn.episodes():
        dvmn.seed_demo()
    cases = [
        ("fruit fly visual motion and flying", "Fly vision and neural circuits"),
        ("language model retrieves old conversation context", "Long-term language-model memory"),
        ("spiking network for robot camera vision", "Robot perception with spiking neurons"),
        ("measure retrieval ranking using precision and recall", "Evaluating memory retrieval"),
    ]
    correct = 0
    print("DVMN smoke evaluation (four hand-labelled queries)")
    for query, expected in cases:
        results = dvmn.recall(query, 1)
        predicted = results[0]["title"] if results else "(none)"
        ok = predicted == expected
        correct += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | expected: {expected} | predicted: {predicted}")
    print(f"Top-1 accuracy: {correct}/{len(cases)} = {correct / len(cases):.1%}")
    print("Note: this tiny hand-written test is a smoke test, not research-grade evidence.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DVMN-0.1: lightweight episodic memory and conceptual familiarity prototype"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create/check the SQLite database")
    p.set_defaults(func=command_init)

    p = sub.add_parser("seed-demo", help="add example memory episodes")
    p.set_defaults(func=command_seed)

    p = sub.add_parser("add", help="save a new memory episode")
    p.add_argument("--title", help="short episode title")
    p.add_argument("--content", help="episode content; omit to enter it interactively")
    p.set_defaults(func=command_add)

    p = sub.add_parser("list", help="list stored memory episodes")
    p.set_defaults(func=command_list)

    p = sub.add_parser("recall", help="retrieve past episodes related to a query")
    p.add_argument("query", help="new question or concept")
    p.add_argument("--limit", type=int, default=5, help="number of memories to show")
    p.set_defaults(func=command_recall)

    p = sub.add_parser("chat", help="retrieve memories and ask a local Ollama model")
    p.add_argument("prompt", nargs="?", help="message to send")
    p.add_argument("--limit", type=int, default=3, help="number of memories to include")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.set_defaults(func=command_chat)

    p = sub.add_parser("evaluate", help="run a small hand-labelled smoke test")
    p.set_defaults(func=command_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        dvmn = DVMN()
        args.func(dvmn, args)
        return 0
    except (ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())