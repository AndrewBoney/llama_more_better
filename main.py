from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    dataset_name: Optional[str] = field(default="microsoft/wiki_qa")
    output_dir: Optional[str] = field(default="./lora-llama")
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    learning_rate: Optional[float] = field(default=3e-4)
    num_train_epochs: Optional[int] = field(default=3)
    batch_size: Optional[int] = field(default=4)
    max_length: Optional[int] = field(default=512)

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
    )

    # Create PEFT model
    model = get_peft_model(model, lora_config)

    tokens_in = tokenizer.apply_chat_template([{"role" : "user", "content" : "hello world"}], add_generation_prompt=True, return_tensors = "pt")
    tokens_out = model.generate(tokens_in, max_length = 256)
    text_out = tokenizer.decode(tokens_out[0, :])
    print(text_out)

if __name__ == "__main__":
    main()