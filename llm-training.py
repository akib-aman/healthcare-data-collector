import json
from datasets import Dataset, concatenate_datasets
from transformers import (
    # T5 imports
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    # GPT imports
    GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
)

#############################################
# T5 TRAINING LOGIC
#############################################
def train_t5():
    """
    Train a T5 model for extracting fields (age, name, sex, etc.).
    Saves the model & tokenizer to ./t5-trained-model
    """
    
    # Configuration for T5 dataset paths
    dataset_paths = {
        "age_training": {
            "file_path": "./t5-training-datasets/age-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the age from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "name_training": {
            "file_path": "./t5-training-datasets/names-and-age-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the name from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "sex_training": {
            "file_path": "./t5-training-datasets/sex-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the sex from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        # Add more T5 datasets here if needed
    }

    def load_and_convert_datasets(dataset_paths):
        processed_datasets = {}
        for dataset_name, config in dataset_paths.items():
            file_path = config["file_path"]
            conversion_logic = config["conversion_logic"]
            # Load the dataset from JSON
            with open(file_path, "r") as file:
                data = json.load(file)
            # Convert to Hugging Face Dataset format
            processed_datasets[dataset_name] = Dataset.from_list([
                conversion_logic(ex) for ex in data
            ])
        return processed_datasets

    processed_datasets = load_and_convert_datasets(dataset_paths)
    combined_dataset = concatenate_datasets(list(processed_datasets.values()))

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Preprocess the dataset
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=128, 
            padding="max_length", 
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=128,
            padding="max_length",
            truncation=True
        )
        # Replace pad_token_id with -100 for the labels (so they’re ignored in loss computation)
        model_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
            for label_ids in labels["input_ids"]
        ]
        return model_inputs

    tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

    # Load the model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./t5-trained-model",
        evaluation_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs-t5",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    # Data collator for Seq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train T5
    print("Starting T5 training...")
    trainer.train()
    print("T5 training complete.")

    # Save T5 model & tokenizer
    model.save_pretrained("./t5-trained-model")
    tokenizer.save_pretrained("./t5-trained-model")
    print("T5 model saved to ./t5-trained-model")


#############################################
# GPT TRAINING LOGIC
#############################################
def train_gpt():
    """
    Train a GPT model for more 'wordy' Q&A style interactions.
    Uses a new directory (gpt-training-datasets) for training data.
    Saves the model & tokenizer to ./gpt-trained-model
    """
    
    # Example: We'll assume there's a JSON file in gpt-training-datasets
    # called question-answer-training.json or something similar.
    # Adjust to your own file(s).
    file_path = "./gpt-training-datasets/question-answer-training.json"

    # Load the JSON data
    with open(file_path, "r") as file:
        gpt_data = json.load(file)

    # Convert to Dataset with GPT-friendly delimiters
    dataset = Dataset.from_list([
        {
            "text": (
                f"<|startoftext|>Input: {ex['input']}\n"
                f"Output: {ex['output']}<|endoftext|>"
            )
        }
        for ex in gpt_data
    ])

    # Load and configure the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "additional_special_tokens": ["<|startoftext|>", "<|endoftext|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token by default

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Load the GPT2 model and resize token embeddings
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        evaluation_strategy="no",
        save_steps=200,
        logging_steps=50,
        output_dir="./gpt-trained-model",
        overwrite_output_dir=True
    )

    # Data collator (no masked language modeling for GPT2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer for GPT
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train GPT
    print("Starting GPT training...")
    trainer.train()
    print("GPT training complete.")

    # Save GPT model & tokenizer
    model.save_pretrained("./gpt-trained-model")
    tokenizer.save_pretrained("./gpt-trained-model")
    print("GPT model saved to ./gpt-trained-model")


#############################################
# MAIN SCRIPT ENTRY POINT
#############################################
if __name__ == "__main__":
    # Option A: Just call both
    train_t5()
    train_gpt()

    # Option B (comment out the above two lines) and selectively call one:
    # train_t5()
    # train_gpt()

    # Or handle CLI arguments to choose which model to train, e.g.:
    #
    # import sys
    # args = sys.argv
    # if len(args) > 1:
    #     if args[1] == "t5":
    #         train_t5()
    #     elif args[1] == "gpt":
    #         train_gpt()
    #     elif args[1] == "both":
    #         train_t5()
    #         train_gpt()
    # else:
    #     print("Usage: python train_models.py [t5 | gpt | both]")
