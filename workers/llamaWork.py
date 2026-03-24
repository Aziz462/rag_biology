import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI

class LLMGenerator:
    def __init__(self, model_id, device="cpu"):
        print(f"loading Hugging Face model: {model_id}...")
        self.model_id = model_id
        
        if device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        print(f"Model loaded on {self.device}")

    def generate_text(self, messages, context):
        if context and context != "no context given":
            system_instruction = f"""You are a helpful medical assistant. 
Use the following Context to answer the Question. 
For every fact you state, you MUST cite the source using its ID in brackets, e.g., [PMC1234567].
If the answer is not in the Context, say "I cannot find the answer in the document."

Context:
{context}"""
        else:
            system_instruction = "You are a helpful medical assistant. If no context is provided, answer based on your general knowledge"

        formatted_messages = [{"role": "system", "content": system_instruction}]
        
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append(msg)

        # Use tokenizer's chat template if available
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            return outputs[0]["generated_text"][len(prompt):].strip()
        else:
            # Fallback for models without a chat template
            full_prompt = ""
            for msg in formatted_messages:
                full_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
            full_prompt += "ASSISTANT:"
            
            outputs = self.pipe(full_prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
            return outputs[0]["generated_text"][len(full_prompt):].strip()

class APILLMGenerator:
    def __init__(self, api_key, model_name, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate_text(self, messages, context):
        system_instruction = """You are a helpful medical assistant. 
Use the following Context to answer the Question. 
For every fact you state, you MUST cite the source using its ID in brackets, e.g., [PMC1234567].
If the answer is not in the Context, say "I cannot find the answer in the document.", but if Context is "no context given", then you just try to answer the questions of the user, but point out the lack of context in documents and suggest looking up relevant documents from PubMed."""

        if context and context != "no context given":
            context_msg = f"Context:\n{context}"
            system_instruction += f"\n\n{context_msg}"

        formatted_messages = [{"role": "system", "content": system_instruction}]
        for msg in messages:
            formatted_messages.append(msg)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=0.7
        )
        return response.choices[0].message.content
