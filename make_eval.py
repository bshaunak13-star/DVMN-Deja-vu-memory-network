import csv

queries = [
    ("What did I investigate about fruit-fly vision?", "direct"),
    ("Explain my work on fruit-fly neural circuits.", "direct"),
    ("What biological mechanisms did I explore for visual perception?", "paraphrase"),
    ("What did I investigate about insect neural architecture?", "paraphrase"),
    ("How might a fly process visual information?", "conceptual"),
    ("What was my project about fly-inspired intelligence?", "conceptual"),
    ("What did I explore about robot perception?", "direct"),
    ("How could spiking neurons help a robot understand its surroundings?", "conceptual"),
    ("What did I investigate about robotic neural systems?", "paraphrase"),
    ("What connects biological vision and robot perception?", "cross-topic"),
    ("What is the purpose of my DVMN project?", "direct"),
    ("Explain my Déjà Vu Memory Network idea.", "paraphrase"),
    ("How does DVMN retrieve memories from previous discussions?", "conceptual"),
    ("What is concept-triggered memory retrieval?", "conceptual"),
    ("How can a new question activate an older memory?", "conceptual"),
    ("What problem is DVMN intended to solve?", "direct"),
    ("How does DVMN rank candidate memories?", "conceptual"),
    ("What is the difference between memory storage and retrieval?", "conceptual"),
    ("How could DVMN connect two previously discussed topics?", "cross-topic"),
    ("What did I want to measure in my DVMN research?", "direct"),
    ("How could fly vision research relate to spiking neural networks?", "cross-topic"),
    ("Can biological neural circuits inspire artificial memory systems?", "cross-topic"),
    ("How could robotic perception connect with concept-based recall?", "cross-topic"),
    ("What is the relationship between neural circuits and memory retrieval?", "cross-topic"),
    ("How might a biological inspiration influence an AI architecture?", "cross-topic"),
    ("What did I investigate about local language models?", "direct"),
    ("How does Gemma generate answers using retrieved memories?", "direct"),
    ("What role does Ollama play in my prototype?", "direct"),
    ("Why should retrieved memories support an AI answer?", "conceptual"),
    ("How can I evaluate memory retrieval objectively?", "direct"),
    ("What do precision and recall measure in my experiment?", "evaluation"),
    ("How does the retrieval threshold affect results?", "evaluation"),
    ("Why might a low threshold retrieve irrelevant memories?", "evaluation"),
    ("How can I compare DVMN with keyword matching?", "evaluation"),
    ("What does a false negative mean in memory retrieval?", "evaluation"),
    ("How can I test retrieval with paraphrased questions?", "evaluation"),
    ("What would count as evidence that DVMN works well?", "evaluation"),
    ("How can I prevent unsupported claims from retrieved memories?", "evaluation"),
    ("What is the difference between related topics and proven connections?", "evaluation"),
    ("How can I determine whether a retrieved episode is relevant?", "evaluation"),
]

with open("eval_queries.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "query", "query_type"])
    for i, (query, kind) in enumerate(queries, start=1):
        writer.writerow([i, query, kind])

print(f"Created eval_queries.csv with {len(queries)} queries.")
print("Next: match each query with actual episode IDs and label relevance.")