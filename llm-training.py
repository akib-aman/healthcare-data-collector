import json
from datasets import Dataset, concatenate_datasets, load_dataset
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

    model.save_pretrained("./web/server/t5-trained-model")
    tokenizer.save_pretrained("./web/server/t5-trained-model")
    print("T5 model saved to ./t5-trained-model")


#############################################
# GPT TRAINING LOGIC
#############################################

def train_gdpr_qa():
    """
    Train GPT model specifically on the GDPR_QA_instruct_dataset.
    Returns a tokenized dataset formatted for GPT-2.
    """
    # Load GDPR dataset
    dataset = load_dataset("sims2k/GDPR_QA_instruct_dataset")

    # Format the dataset (combine instruction, input, and output)
    def format_example(example):
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = example["output"]

        # Create a formatted prompt for GPT-2
        return {
            "text": (
                f"<|startoftext|>Instruction: {instruction}\n"
                f"Input: {input_text}\n"
                f"Output: {output_text}<|endoftext|>"
            )
        }

    formatted_dataset = dataset["train"].map(format_example)

    # Load and configure the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "additional_special_tokens": ["<|startoftext|>", "<|endoftext|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, tokenizer

def train_gpt():
    """
    Train GPT-2 model using both GDPR QA dataset and the older question-answer-training.json dataset.
    Saves the model & tokenizer to ./gpt-trained-model.
    """
    # Prepare the GDPR QA dataset
    tokenized_gdpr_dataset, tokenizer = train_gdpr_qa()

    # Prepare the old question-answer dataset
    file_path = "./gpt-training-datasets/question-answer-training.json"
    with open(file_path, "r") as file:
        old_data = json.load(file)

    # Format the old dataset with GPT-friendly delimiters
    old_dataset = Dataset.from_list([
        {
            "text": (
                f"<|startoftext|>Question: {ex['question']}\n"
                f"Answer: {ex['answer']}<|endoftext|>"
            )
        }
        for ex in old_data
    ])

    # Tokenize the old dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_old_dataset = old_dataset.map(tokenize_function, batched=True)

    # Combine both datasets
    combined_tokenized_dataset = concatenate_datasets(
        [tokenized_gdpr_dataset, tokenized_old_dataset]
    )

    # Load the GPT-2 model and resize token embeddings
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
        output_dir="./gpt-combined-trained-model",
        overwrite_output_dir=True
    )

    # Data collator (no masked LM for GPT-2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer for GPT-2
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_tokenized_dataset,
        data_collator=data_collator,
    )

    # Train GPT-2
    print("Starting GPT-2 training on combined datasets...")
    trainer.train()
    print("GPT-2 training complete.")

    # Save GPT-2 model & tokenizer
    model.save_pretrained("./gpt-combined-trained-model")
    tokenizer.save_pretrained("./gpt-combined-trained-model")

    model.save_pretrained("./web/server/gpt-combined-trained-model")
    tokenizer.save_pretrained("./web/server/gpt-combined-trained-model")
    print("GPT-2 model saved to ./gpt-combined-trained-model")
    """
    Train GPT-2 model specifically for GDPR QA dataset.
    Saves the model & tokenizer to ./gpt-trained-model.
    """
    # Prepare the GDPR QA dataset
    tokenized_dataset, tokenizer = train_gdpr_qa()

    # Load the GPT-2 model and resize token embeddings
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
        output_dir="./gpt-gdpr-trained-model",
        overwrite_output_dir=True
    )

    # Data collator (no masked LM for GPT-2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer for GPT-2
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train GPT-2
    print("Starting GPT-2 training on GDPR QA dataset...")
    trainer.train()
    print("GPT-2 training complete.")

    # Save GPT-2 model & tokenizer
    model.save_pretrained("./gpt-gdpr-trained-model")
    tokenizer.save_pretrained("./gpt-gdpr-trained-model")

    model.save_pretrained("./web/server/gpt-gdpr-trained-model")
    tokenizer.save_pretrained("./web/server/gpt-gdpr-trained-model")
    print("GPT-2 model saved to ./gpt-gdpr-trained-model")


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
