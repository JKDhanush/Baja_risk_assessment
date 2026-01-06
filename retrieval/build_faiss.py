import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

text = open("data/manuals/risk_assessment_knowledge.txt").read()
chunks = text.split("\n\n")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/manuals/risk.index")
