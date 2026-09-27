from app.core.embeddings import EmbeddingService

service = EmbeddingService()

chunks = [
    "Employees are entitled to twelve days of paid sick leave per calendar year.",
    "The company observes twelve public holidays per calendar year.",
    "Our office cafeteria serves lunch between noon and two in the afternoon.",
]

question = "How many sick leaves do I get?"

print("Embedding chunks...")
chunk_vectors = service.embed_documents(chunks)

print("Embedding question...")
question_vector = service.embed_query(question)

print(f"\nVector length: {len(question_vector)} numbers")
print(f"First 5 values: {question_vector[:5]}\n")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b)


print(f'Question: "{question}"\n')
print(f"{'score':<8} chunk")
print("-" * 70)

for chunk, vector in zip(chunks, chunk_vectors, strict=True):
    score = cosine(question_vector, vector)
    print(f"{score:.4f}   {chunk[:60]}")
