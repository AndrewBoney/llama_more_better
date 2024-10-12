import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

data = load_dataset("microsoft/wiki_qa")

model_str = "meta-llama/Llama-3.2-1B-Instruct"
#model = AutoModelForCausalLM.from_pretrained(model_str).eval()
tokenizer = AutoTokenizer.from_pretrained(model_str)

# Dset for loading 
class Dset(Dataset):
    def __init__(
        self, 
        split, 
        tokenizer = tokenizer
    ):
        super().__init__()
        self.data = data[split]
        
        tokenizer.pad_token = tokenizer.eos_token # create pad token as eos token

        #TODO: this is currently producing one long output i.e. a long ass conversation with all rows. Reformat to new rows
        self.tokenized_questions = tokenizer.apply_chat_template(
            [{"role" : "user", "content" : q} for q in self.data["question"]],
            return_tensors = "pt",
            padding = True,
            add_generation_prompt = True 
        )

        self.tokenized_answers = tokenizer(
            self.data["answer"],
            return_tensors = "pt",
            padding = True
        )

    def __len__(self):
        return len(self.data)

    @torch.no_grad()
    def __getitem__(self, idx):
        return {
            "question" : self.tokenized_questions[idx, :],
            "answer" : self.tokenized_answers[idx, :],
            "label" : self.data["label"]
        }