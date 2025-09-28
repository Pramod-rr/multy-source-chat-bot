import gradio as gr
from retriever import get_retriver
from generate import generate_response
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def chat(query):
    try:
        if not query.strip():
            return "Please ask a question!"
        
        chunks = get_retriver(query)
        answer = generate_response(query, chunks)
        
    
        sources = "\n".join([f"- {c['source']} (score: {c['score']:.2f})" 
                             if isinstance(c.get('score'), (int, float))
                             else f"- {c['source']} (score: {c.get('score', 'N/A')})" for c in chunks])
        full_response = f"{answer}\n\n**Sources:**\n{sources}"
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error setting up chat: {str(e)}")  
        
        
iface = gr.Interface(
            fn=chat,
            inputs=gr.Textbox(label="Ask your FAQ", placeholder="e.g., What do you want to know?"),
            outputs=gr.Textbox(label="Answer"),
            title="Multi-Source FAQ Bot",
            description="Powered by Phi-2 LLM."
        )
if __name__ == "__main__":
    iface.launch(share=True) 
