import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/manuals/risk.index")
chunks = open("data/manuals/risk_assessment_knowledge.txt").read().split("\n\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=4):
    q_emb = model.encode([query])
    _, I = index.search(np.array(q_emb), k)
    return "\n".join([chunks[i] for i in I[0]])
