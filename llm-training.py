import os
import json
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    # T5 imports
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    # GPT imports
    GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(" NO GPU DETECTED! ")
    exit()

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
        "classification_training": {
            "file_path": "./t5-training-datasets/classification-training.json",
            "conversion_logic": lambda ex: {
                "input": (
                    "Does this input require data extraction or a detailed response?\n"
                    f"Input: {ex['input']}\n"
                    "Answer with either 'Data Extraction' or 'Detailed Response'."
                ),
                "output": ex["output"]
            }
        },
        "religion_training": {
            "file_path": "./t5-training-datasets/religion-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the religion/belief from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "title_training": {
            "file_path": "./t5-training-datasets/title-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the title from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "disability_training": {
            "file_path": "./t5-training-datasets/disability-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the disability from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "ethnicity_training": {
            "file_path": "./t5-training-datasets/ethnicity-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the ethnicity from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "gender_reassignment_training": {
            "file_path": "./t5-training-datasets/gender-reassignment-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the gender reassignment from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "marriage_civilrelationship_training": {
            "file_path": "./t5-training-datasets/marriage-civilrelationship-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the marriage/civil partnership status from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "sexual_orientation_training": {
            "file_path": "./t5-training-datasets/sexual-orientation-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the sexual orientation from this text: {ex['input']}",
                "output": ex["output"]
            }
        }
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

    # Load the model and move it to the device (GPU/CPU)
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

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
def train_gpt():
    """
    Train a GPT model for Q&A style interactions, 
    saving the model & tokenizer to ./gpt-trained-model
    """
    file_path = "./gpt-training-datasets/question-answer-training.json"

    # 1. Load the JSON data (with 'question'/'answer' keys)
    with open(file_path, "r") as file:
        gpt_data = json.load(file)

    # 2. Convert to Dataset with a "Question: ... Answer: ..." format
    dataset = Dataset.from_list([
        {
            "text": (
                f"<|startoftext|>Question: {ex['question']}\n"
                f"Answer: {ex['answer']}"
                f"<|endoftext|>"
            )
        }
        for ex in gpt_data
    ])

    # 3. Load and configure tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {
        "additional_special_tokens": ["<|startoftext|>", "<|endoftext|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't natively have a pad token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

   # 4. Load GPT-2 model, move to device, and resize token embeddings
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))

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

    # 5. Data collator: no masked LM for GPT-2
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 6. Train
    print("Starting GPT training...")
    trainer.train()
    print("GPT training complete.")

    # 7. Save the model & tokenizer
    model.save_pretrained("./gpt-trained-model")
    tokenizer.save_pretrained("./gpt-trained-model")

    # Optionally save elsewhere too
    model.save_pretrained("./web/server/gpt-trained-model")
    tokenizer.save_pretrained("./web/server/gpt-trained-model")

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
