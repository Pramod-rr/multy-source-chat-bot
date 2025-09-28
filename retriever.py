import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K = 3
INDEX_PATH="faq_index.faiss"
METADATA_PATH="metadata.pkl"

def get_retriver(query, chunks=None):
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        index = faiss.read_index(INDEX_PATH)
        
        with open(METADATA_PATH, "rb") as f:
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