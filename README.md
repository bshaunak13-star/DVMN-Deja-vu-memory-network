# DVMN 0.4 — Trainable Déjà Vu Memory Network

DVMN 0.4 is a research prototype for **episodic memory retrieval**. It stores conversations/documents, imports JSONL corpora, trains a small PyTorch relevance network, and can optionally pass retrieved memories to a locally running Llama-family model through Ollama.

## Important reality check

- `train` genuinely optimizes a neural network, but the first training labels are **weak labels** generated from each document's own words versus other documents. It is not equivalent to training a foundation model, and the network does not automatically learn human-like déjà vu.
- Large corpora improve coverage, not necessarily accuracy. Measure with a separate, manually labelled test set. The reported validation accuracy is on generated weak labels and must not be treated as real-world retrieval accuracy.
- Ollama/Llama is optional and **not trained by this project**. DVMN retrieves memories; the local Llama model reasons over the retrieved text.
- Keep private conversations and licensed datasets private; check dataset terms before importing.

## Windows / Python 3.14 setup

Use a **new folder**; do not overwrite your DVMN 0.1 folder or its database. PyTorch and scikit-learn wheels can lag new Python releases. If installation fails on Python 3.14, use Python 3.12 or 3.13 in a separate virtual environment for this project.

```powershell
cd "$HOME\Downloads\DVMN-0.4"
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py init
python main.py add "A robot uses camera frames to navigate an unfamiliar room." --title "Visual navigation"
python main.py add "Memory systems retrieve past episodes when a new query shares concepts." --title "Episodic retrieval"
python main.py add "A fruit fly uses visual motion cues to stabilize flight." --title "Fly vision"
python main.py add "A language model can use retrieved context to answer questions." --title "LLM memory"
python main.py train --epochs 12
python main.py recall "How can a robot use vision to move through a room?" --threshold 0.18
```

If pip says no compatible PyTorch or scikit-learn wheel exists for Python 3.14, those dependencies are not available for your exact interpreter build yet. Since the neural-training feature requires them, use a supported Python version when one is available; do not delete an existing database to fix a dependency issue.

## Import a larger dataset

Create a UTF-8 JSON Lines file, one JSON object per line, with `title` and `content` (or `text`):

```json
{"title":"Visual navigation","content":"A robot estimates movement from camera frames."}
{"title":"Episodic memory","content":"A system stores past conversations and retrieves related episodes."}
```

Then run:

```powershell
python main.py import-jsonl .\my_dataset.jsonl
python main.py train --epochs 20 --pairs-per-episode 30
```

For larger datasets, import in batches and start with `--max-features 30000`; TF-IDF fitting and pair generation use RAM. Don't import a raw multi-gigabyte corpus without cleaning/deduplication first.

## Optional local Llama reasoning with Ollama

1. Install Ollama for Windows and download a model supported by your machine, e.g. run `ollama run llama3.2` once.
2. Leave Ollama running.
3. In PowerShell:

```powershell
$env:DVMN_OLLAMA_MODEL="llama3.2"
python main.py ask "What did my past notes say about visual navigation?"
```

The `ask` command calls `http://localhost:11434` only. No OpenRouter API is used.

## CLI

- `python main.py init`
- `python main.py add "episode text" --title "title"`
- `python main.py import-jsonl path.jsonl`
- `python main.py list --limit 20`
- `python main.py train --epochs 12`
- `python main.py recall "query" --threshold 0.18 --limit 5`
- `python main.py ask "query"` (requires Ollama)

## Evaluation checklist before claiming accuracy

Create query examples with known relevant episode IDs and irrelevant queries, split by episode/topic into train/validation/test, and report precision, recall, F1, MRR/Recall@k, and false-positive rate. Do not tune the threshold on the final test set. The built-in weak-label validation is a training diagnostic only.
