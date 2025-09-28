import gradio as gr
from retriever import get_retriver
from generate import generate_response

def chat(query):
    if not query.strip():
        return "Please ask a question!"
    
    chunks = get_retriver(query)
    answer = generate_response(query, chunks)
    
  
    sources = "\n".join([f"- {c['source']} (score: {c['score']:.2f})" for c in chunks])
    full_response = f"{answer}\n\n**Sources:**\n{sources}"
    
    return full_response


iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask your FAQ", placeholder="e.g., What is the refund policy?"),
    outputs=gr.Textbox(label="Answer"),
    title="Multi-Source FAQ Bot",
    description="Powered by Phi-2 LLM and RAG from your sources."
)

if __name__ == "__main__":
    iface.launch(share=True) 