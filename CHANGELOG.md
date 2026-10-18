# Changelog

## 0.5.1-final
- Removed the duplicate gradient update in the training loop.
- Kept core training dependency-free, avoiding the user's PyTorch DLL initialization failure.
- Added compatibility migration for earlier episode tables lacking an explicit `id` column.
- Added reindexing for legacy episodes with missing concept/relation metadata.
- Expanded JSON/JSONL import field handling and added JSONL export for backups.
- Added optional local Ollama integration (`ask`) to test retrieval with a real language model.
- Uses a new default database filename so older data is not overwritten accidentally.
- Clarified that pseudo-label training is not ground-truth evaluation and that score outputs are not calibrated probabilities.
