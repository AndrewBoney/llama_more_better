import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_str = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype = torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_str)

prompt = [{"role" : "user", "content" : "hello world"}]
tokens = tokenizer.apply_chat_template(prompt, return_tensors = "pt")
with torch.no_grad():
    output = model(tokens)

print(output)