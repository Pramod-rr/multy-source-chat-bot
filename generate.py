from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "microsoft/phi-2"  # Small LLM (~2.7B params)

# Load model (downloads on first run)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def generate_response(query, context_chunks):
    # Combine chunks into context
    context = "\n".join([chunk["content"] for chunk in context_chunks])
    
    prompt = f"""You are a helpful FAQ bot. Answer the question based only on the provided context. Keep it concise, like a FAQ entry.

Context:
{context}

Question: {query}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()  