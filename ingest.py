import logging
from bs4 import BeautifulSoup
import requests
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DIR = r"C:\Users\rnmpr\Documents\Introduction to Machine Learning with Python ( PDFDrive ).pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_source():
    try:
       documents =[]
       
       pdf_file = SOURCE_DIR
       reader = PdfReader(pdf_file)
       text = ""
       for page in reader.pages:
           text += page.extract_text()
       documents.append({"source": os.path.basename(pdf_file), "content": text})
    
    #    for txt_file in [f for f in os.listdir(SOURCE_DIR) if f.endswith('.txt')]:
    #        with open(os.path.join(SOURCE_DIR, txt_file), 'r') as f:
    #            content = f.read()
    #        documents.append({"source": txt_file, "content":content})
       
    #    web_url = ""
    #    response = requests.get("")
    #    soup = BeautifulSoup(response.content, 'html.parser')
    #    web_text = soup.get_text()
    #    documents.append({"source": web_url, "content": web_text})  
       logger.info(f"Loaded {len(documents)} documents from {SOURCE_DIR}")
       return documents
   
    except Exception as e:
        logger.error(f"Error loading source: {str(e)}")
        raise 
    

def chunk_document(text, chunk_size=400, chunk_overlap=20):
    try:
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
            
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise 
    
def build_index():
    try:
        docs = load_source()
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        all_chunks = []
        all_metadata = []
        
        for doc in docs:
            chunks = chunk_document(doc["content"])
            for chunk in chunks:
                embedding = model.encode(chunk)
                all_chunks.append(embedding)
                all_metadata.append({"content":chunk, "source":doc["source"]})      
        
        dimension = len(all_chunks[0])
        index =  faiss.IndexFlatL2(dimension)
        embeddings = np.array(all_chunks).astype('float32')
        index.add(embeddings)
        
        faiss.write_index(index, "faq_index.faiss")
        with open("metadata.pkl","wb") as f:
            pickle.dump(all_metadata,f)
            
        logger.info("Initialized embeddings")    
        print(f"Indexed {len(all_chunks)} chunks from {len(docs)} sources.")
        
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        
if __name__ == "__main__":
    build_index() 