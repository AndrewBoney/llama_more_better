import numpy as np

import torch

from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

class RLHFDataProcessor:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", seed=42):
        """
        Initialize the RLHF data processor
        
        Args:
            model_name (str): Name of the model to use for tokenization
            seed (int): Random seed for reproducibility
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def convert_text_to_chat(text):
        """
        Convert text with 'H:' and 'Assistant:' prefixes to a list of chat messages
        with role/content format.
        
        Args:
            text (str): Input text containing the conversation
            
        Returns:
            list: List of dictionaries with 'role' and 'content' keys
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        messages = []
        current_message = None
        
        for line in lines:
            if line.startswith('Human:'):
                if current_message:
                    messages.append(current_message)
                current_message = {
                    'role': 'user',
                    'content': line[6:].strip()
                }
            elif line.startswith('Assistant:'):
                if current_message:
                    messages.append(current_message)
                current_message = {
                    'role': 'assistant',
                    'content': line[10:].strip()
                }
            else:
                if current_message:
                    current_message['content'] += ' ' + line.strip()
        
        if current_message:
            messages.append(current_message)
        
        return messages
    
    def process_dataset(self, example):
        """Process a single example from the dataset"""
        example["chosen_processed"] = self.tokenizer.apply_chat_template(
            self.convert_text_to_chat(example["chosen"]), 
            tokenize=False
        )
        example["rejected_processed"] = self.tokenizer.apply_chat_template(
            self.convert_text_to_chat(example["rejected"]), 
            tokenize=False
        )
        return example
    
    def collate_tokens(self, data):
        """Collate function for DataLoader"""
        data = torch.utils.data.default_collate(data)
        return {
            "chosen": self.tokenizer(
                data["chosen_processed"], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            ),
            "rejected": self.tokenizer(
                data["rejected_processed"], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            ),
        }

def get_anthropic_rlhf_data(
    batch_size : int = 4, 
    seed : int = 42, 
    num_workers : int = 4,
    model_name : str ="meta-llama/Llama-3.2-1B-Instruct"
):
    """
    Load and process the Anthropic RLHF dataset, splitting into train/val/test
    
    Args:
        batch_size (int): Batch size for DataLoader
        seed (int): Random seed for reproducibility
        num_workers (int): Number of workers for the DataLoader
        model_name (str): Model name for tokenizer
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize processor
    processor = RLHFDataProcessor(model_name=model_name, seed=seed)
    
    # Load dataset
    dataset = load_dataset("Anthropic/hh-rlhf")
    
    # Process the training data
    train_data = dataset["train"].map(processor.process_dataset)
    
    # Calculate splits
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    
    # Split training data into train and validation
    train_dataset, val_dataset = random_split(
        train_data, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Get test dataset
    test_dataset = dataset["test"].map(processor.process_dataset)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=processor.collate_tokens
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=processor.collate_tokens
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=processor.collate_tokens
    )
    
    return train_loader, val_loader, test_loader