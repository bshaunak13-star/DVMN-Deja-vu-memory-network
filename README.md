DVMN 0.5 — Déjà Vu Memory Network by shaunak bellubbi

A Research Prototype for Concept-Activated Episodic Memory Retrieval

DVMN (Déjà Vu Memory Network) is an experimental memory-retrieval system designed to help a language-model-based application reconnect new queries with relevant past experiences. Instead of relying exclusively on keyword overlap, DVMN combines lexical similarity, concept matching, relational signals, and spreading activation to rank stored episodes.

The project explores a central research question:

«Can concept-activated memory retrieval improve the relevance of recalled experiences compared with a keyword-based baseline?»

DVMN 0.5 is a measurable, locally runnable research prototype built with Python and SQLite, with an optional Ollama integration for language-model-assisted responses.

---

Table of Contents

- "Overview" (#overview)
- "Research Objectives" (#research-objectives)
- "Key Features" (#key-features)
- "How DVMN Works" (#how-dvmn-works)
- "Architecture" (#architecture)
- "Technology Stack" (#technology-stack)
- "Project Structure" (#project-structure)
- "Installation" (#installation)
- "Quick Start" (#quick-start)
- "Ollama Integration" (#ollama-integration)
- "Training the Retrieval Model" (#training-the-retrieval-model)
- "Evaluation and Results" (#evaluation-and-results)
- "Limitations" (#limitations)
- "Future Work" (#future-work)
- "Reproducibility" (#reproducibility)
- "License" (#Apache License 2.0)

---

Overview

Traditional keyword-based retrieval can struggle when a new query describes an earlier experience using different words. DVMN investigates whether relationships between concepts and stored episodes can help recover those connections.

For example, a query about fly-inspired intelligence may be connected to earlier episodes concerning fruit-fly vision, neural circuits, or biologically inspired robotics—even when the wording differs.

DVMN represents past experiences as episodes, extracts concepts from them, builds a concept co-activation graph, and computes a relevance score for candidate memories.

The system ranks memories from stronger to weaker matches. An optional trained scoring model adjusts how retrieval features contribute to the final ranking.

Important distinction: DVMN is a research prototype for concept-aware memory retrieval. It is not a complete simulation of biological memory, and its current implementation does not establish that it reproduces the human déjà vu phenomenon.

Research Objectives

1. Episodic memory: Store past experiences as retrievable records.
2. Concept activation: Identify concepts that connect a new query to previous episodes.
3. Associative retrieval: Use a graph of co-activated concepts to help discover related memories.
4. Learned relevance scoring: Train a lightweight model using human-labelled query–episode pairs.
5. Baseline comparison: Compare DVMN retrieval against a keyword-based system.
6. Measurable evaluation: Assess precision, recall, F1, and ranking quality on labelled queries.

Key Features

- Episodic storage: Maintain memory records in a local SQLite database.
- Multi-feature retrieval: Combine seven retrieval signals.
- Concept association graph: Represent co-activation relationships between concepts.
- Spreading activation: Propagate activation through connected concepts to support associative recall.
- Learned scorer: Adjust feature contributions using labelled relevance examples.
- Keyword baseline: Provide a reference system for retrieval comparisons.
- Human-in-the-loop feedback: Record relevance judgments for training and evaluation.
- Command-line interface: Add, list, import, export, retrieve, train, and evaluate memories.
- Optional local LLM integration: Connect retrieved context to a compatible Ollama model.
- Reproducible experiments: Use CSV labels, JSON model files, and evaluation scripts.

How DVMN Works

The retrieval pipeline has five main stages.

1. Episodic memory

Past information is stored as episodes in SQLite. An episode can contain a title, text, extracted concepts, and associated metadata.

2. Feature extraction

For a query and each candidate episode, DVMN computes seven signals:

Feature| Purpose
"lexical"| Measures word-level overlap
"concept"| Measures concept correspondence
"relation"| Captures relationships between matched concepts
"spreading"| Measures activation propagated through the concept graph
"title_hint"| Rewards useful matches in episode titles
"temporal"| Incorporates temporal information
"concept_density"| Represents the concentration of relevant concepts

3. Associative activation

DVMN constructs a graph from concept co-activation in stored episodes. When a query activates a concept, the spreading-activation component uses graph connections to help identify related concepts and memories.

This graph-based mechanism is a central design element of the prototype. It is not a vector database or a learned neural memory network in the biological sense.

4. Relevance scoring

The extracted features are combined into a relevance score. DVMN can use default feature weights or a trained scoring model learned from labelled query–episode pairs.

Candidate episodes are ranked by their resulting scores, and the system returns the highest-ranked memories that satisfy the configured retrieval settings.

5. Contextual response generation

When configured with a compatible Ollama model, DVMN can pass retrieved memories to a language model to generate a response grounded in the recalled context.

The retrieval component and the language model have separate roles: DVMN selects memories; the LLM generates the response.

Architecture

                  User Query
                      |
                      v
              Query Processing
                      |
                      v
              Feature Extraction
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Lexical      Concept    Relations
       Matching     Matching
          |           |           |
          +-----------+-----------+
                      |
                      v
             Spreading Activation
                Concept Graph
                      |
                      v
              Relevance Scoring
                      |
              +-------+-------+
              |               |
              v               v
         Default Model    Trained Model
              |               |
              +-------+-------+
                      |
                      v
              Ranked Memories
                      |
                      v
              Optional Ollama
                      |
                      v
                Final Response

Technology Stack

- Python — application logic and CLI
- SQLite — episodic memory storage
- JSON — retrieval model configuration and weights
- CSV — relevance labels and evaluation rankings
- Ollama — optional local or cloud-connected model interface, depending on the selected model
- PowerShell — Windows command-line workflow

The retrieval scorer is lightweight and does not require PyTorch for the current implementation. Ollama is optional; core memory retrieval can be tested independently.

Project Structure

The following is a logical overview. Actual filenames may differ as the project evolves.

DVMN/
├── main4.py
├── dvmn_memory.db
├── dvmn05_model.json
├── dvmn05_train_model.json
├── dvmn_train.db
├── train_labels.csv
├── test_labels.csv
├── final_eval_queries.csv
├── final_relevance_labels.csv
├── trained_comparison_labels.csv
├── final_dvmn_trained_rankings.csv
├── final_keyword_rankings.csv
├── evaluate_final_corrected.py
└── README.md

Note: Do not commit personal conversation data, private memory databases, credentials, or API keys. Include only the files needed to reproduce the experiments, and use a small, sanitized demonstration dataset where possible.

Installation

Requirements

- Windows 10 or Windows 11
- Python compatible with the syntax and dependencies of the project
- PowerShell
- Ollama only if you want LLM-assisted responses

1. Clone the repository

git clone https://github.com/bshaunak13-star/DVMN-Deja-vu-memory-network

2. Verify Python

python --version

3. Inspect the CLI

python .\main4.py -h

Check the available commands before running database-changing operations.

Quick Start

Use the following commands from the project directory.

List stored episodes

python .\main4.py --db ".\dvmn_memory.db" --model ".\dvmn05_model.json" list

Retrieve memories with the default model

python .\main4.py --db ".\dvmn_memory.db" --model ".\dvmn05_model.json" recall "How does fly-inspired intelligence work?"

Retrieve memories with the trained model

python .\main4.py --db ".\dvmn_memory.db" --model ".\dvmn05_train_model.json" recall "How does fly-inspired intelligence work?"

Inspect available commands

python .\main4.py -h
python .\main4.py recall -h
python .\main4.py ask -h

Use "-h" to confirm the exact arguments supported by your checked-out version.

Ollama Integration

DVMN can optionally use Ollama to generate responses from retrieved memories.

1. Verify Ollama

ollama --version
ollama list

Use the exact model identifier printed by "ollama list". Model identifiers may differ between local models and cloud-connected models.

2. Check the Ollama API

Invoke-RestMethod http://localhost:11434/api/tags

If Ollama is not running, start it using the appropriate method for your installation. "ollama serve" can be used in a separate terminal when the service is not already running.

3. Inspect DVMN's LLM arguments

python .\main4.py ask -h

4. Run a memory-assisted query

The following is an example using the project's known CLI pattern. Adapt the model-name option to the output of "ask -h" if required.

python .\main4.py --db ".\dvmn_memory.db" --model ".\dvmn05_train_model.json" ask "What do I remember about fly-inspired intelligence?" --llm ollama --endpoint "http://localhost:11434"

The endpoint shown is the usual local Ollama API address. A cloud-connected model may also require the relevant model identifier and account configuration.

Training the Retrieval Model

DVMN can learn a relevance scorer from human-labelled query–episode pairs.

Workflow

1. Prepare query–episode pairs.
2. Label each pair as relevant ("1") or irrelevant ("0").
3. Split the dataset by query ID to keep evaluation queries separate from training queries.
4. Import the training labels into the training database.
5. Train the scorer and save its model JSON.
6. Evaluate it on held-out queries.
7. Compare the trained model with the keyword baseline.

Example training command:

python .\main4.py --db ".\dvmn_train.db" --model ".\dvmn05_train_model.json" train

The training database should contain the intended training labels and matching episodes. Keep the test labels out of training and model selection.

Reproducibility warning: A split by query ID is preferable to randomly splitting individual query–episode pairs. Pair-level splits can leak information about the same query into training and validation sets.

Evaluation and Results

DVMN was evaluated against a keyword baseline using 20 queries and a manually labelled comparison pool of 147 query–episode pairs. The results below are from the current corrected evaluation run.

Metric| Trained DVMN| Keyword baseline
Queries evaluated| 20| 20
True positives (TP)| 99| 91
False positives (FP)| 1| 9
False negatives (FN)| 38| 46
Precision@5| 0.990| 0.910
Recall@5| 0.723| 0.664
F1@5| 0.835| 0.768
Mean Recall@5| 0.736| 0.671
MRR| 1.000| 1.000

Interpretation

On this labelled comparison:

- Precision@5: DVMN was 8.0 percentage points higher.
- Recall@5: DVMN was 5.9 percentage points higher.
- F1@5: DVMN was 6.7 percentage points higher.
- False positives: DVMN returned fewer judged-irrelevant results in the evaluated top-five lists.

These results are encouraging for the current prototype, but they should be interpreted cautiously.

Evaluation scope: Recall is calculated over the human-labelled candidate pool, not necessarily every possible episode in the database. The experiment covers only 20 queries and a small memory collection. The candidate-generation and labelling process can influence the measured results. MRR should also be independently verified against the ranking files and evaluation implementation before being used as a headline claim.

The figures are experimental observations, not proof of general superiority over all retrieval systems.

Limitations

- The current experiment uses a small dataset and a limited number of evaluation queries.
- Human relevance labels can be subjective or inconsistent.
- Candidate-pool evaluation may not represent recall over the entire memory collection.
- Keyword baselines are sensitive to their implementation and configuration.
- Spreading activation is a simplified graph-based mechanism, not a biological neural simulation.
- The current scorer does not establish long-term autonomous learning or human-like memory.
- LLM response quality depends on the configured model and the retrieved context.
- Results require replication on larger, independently labelled datasets.

Future Work

Potential research directions include:

- Expanding the episodic memory dataset.
- Increasing the number of independently labelled evaluation queries.
- Comparing multiple keyword and semantic retrieval baselines.
- Testing ablations for each of the seven features.
- Studying graph depth, activation decay, thresholds, and top-k settings.
- Reporting confidence intervals and per-query performance.
- Improving evaluation with blinded relevance judgments.
- Measuring latency, storage requirements, and scaling behaviour.
- Exploring temporal and multi-hop associative retrieval.

Reproducibility

For meaningful comparisons, record:

- Code version or Git commit
- Dataset and database version
- Exact query and relevance-label files
- Model JSON and feature weights
- Retrieval threshold and top-k setting
- Baseline implementation and configuration
- Evaluation script version
- Python and relevant dependency versions

Keep training, validation, and test queries separated. Preserve the original model and label files before retraining so results can be reproduced.


Author's Note

DVMN is an ongoing exploration of concept-activated episodic retrieval: a small research prototype built to test whether associative links between past experiences can improve memory recall beyond keyword overlap.

The aim is not just to retrieve text, but to investigate how a system can reconnect ideas across different conversations and contexts.

Version: DVMN 0.5
Project type: Experimental research prototype
Primary focus: Associative memory retrieval and relevance evaluation
