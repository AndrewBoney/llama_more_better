import torch

from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset


def ix_tokens(tokens, idx):
    return {k : v[idx, :] for k, v in tokens}

def prep_q_string(example, col_in, col_out, prefix, suffix):
    example[col_out] = prefix + example[col_in] + suffix
    return example

def get_sentence_embedding(model, tokens):
    with torch.amp.autocast(model.device):
        last = tokens["attention_mask"].sum() - 1 # gets num of attention mask i.e position of last real word 
        emb_out = model(**tokens)
        emb = emb_out.outputs["logits"][last, :] #TODO : check key... "logits" probably isn't right
    
    return emb

# Dset for loading 
class wiki_qa_TrainDset(Dataset):
    def __init__(
        self, 
        split, 
        model_str : str = "meta-llama/Llama-3.2-1B-Instruct",
        model_kwargs = {"torch_dtype" : torch.bfloat16, "device" : "auto"},
        q_prefix = "", # TODO: fill with default llama prompt
        q_suffix = "" # TODO : fill with assistant prompt
    ):
        super().__init__()
        
        # create data
        self.data = load_dataset("microsoft/wiki_qa", split = split)
        partial_prep = partial(prep_q_string, col_in = "question", col_out = "q_p", prefix = q_prefix, suffix = q_suffix)
        self.data = self.data.map(partial_prep) # TODO: test batched / num_proc

        # get model / tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_str, **model_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.tokenizer = tokenizer

        tokenizer.pad_token = tokenizer.eos_token # create pad token as eos token, for padding = True to work

        """
        self.tokenized_questions = tokenizer.apply_chat_template(
            [[{"role" : "user", "content" : q}] for q in self.data["question"]],
            return_tensors = "pt",
            padding = True,
            add_generation_prompt = True 
        )
        """
        self.tokenized_questions = tokenizer(
            self.data["q_p"],
            return_tensors = "pt",
            padding = True
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
        q_tok = ix_tokens(self.tokenized_questions, idx)
        a_tok = ix_tokens(self.tokenized_answers, idx)

        q_emb = get_sentence_embedding(self.model, q_tok)
        a_emb = get_sentence_embedding(self.model, a_tok) 

        return {
            "q_tok" : q_tok,
            "a_tok" : a_tok,
            "q_emb" : q_emb,
            "a_emb" : a_emb,
            "label" : self.data["label"][idx]
        }