import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K = 3

def get_retriver(query, index_path="faq_index.faiss", metadata_path="metadata.pkl"):
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        index = faiss.read_index(index_path)
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            
        query_embedding = model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, K)
        
        results =[]
        
        for i, idx in enumerate(indices[0]):
            chunk = metadata[idx]["content"]
            source = metadata[idx]["source"]
            results.append({"content":chunk, "source":source, "source": 1- distances[0][i]})
            
        return results
        
    except Exception as e:
        logger.error(f"Error setting up retriever: {str(e)}")
        