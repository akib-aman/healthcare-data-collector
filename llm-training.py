import os
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
import numpy as np
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

# ---------------------
# T5 Training Logic
# ---------------------
def train_t5():
    return
    """
    Train a T5 model for extracting fields (age, name, sex, etc.).
    Saves the model & tokenizer to ./t5-trained-model
    """
    
    # Configuration for T5 dataset paths
    dataset_paths = {
        "firstname_training": {
            "file_path": "./t5-training-datasets/firstname-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the firstname from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
        "lastname_training": {
            "file_path": "./t5-training-datasets/lastname-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the lastname from this text: {ex['input']}",
                "output": ex["output"]
            }
        },
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
        },
        "pregnancy_training": {
            "file_path": "./t5-training-datasets/pregnancy-training.json",
            "conversion_logic": lambda ex: {
                "input": f"Extract the pregnancy/maternity from this text: {ex['input']}",
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
            with open(file_path, "r", encoding="utf-8") as file:
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

def compute_similarity(generated, expected):
    """Compute cosine similarity between generated and expected answer embeddings"""
    embeddings = sbert_model.encode([generated, expected])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity_score

# ---------------------
# GPT Validation Logic
# ---------------------
def load_validation_dataset():
    validation_file_path = "./gpt-training-datasets/question-answer-validation.json"

    with open(validation_file_path, "r") as file:
        validation_data = json.load(file)

    validation_dataset = Dataset.from_list([
        {
            "text": (
                f"<|startoftext|>Question: {ex['question']}\n"
                f"Answer: {ex['answer']}"
                f"<|endoftext|>"
            )
        }
        for ex in validation_data
    ])
    
    return validation_dataset

# ---------------------
# GPT Evaluate
# ---------------------
def evaluate_gpt(model, tokenizer, validation_dataset, output_file="gpt2-medium-validation-results.json"):
    model.eval()
    results = []
    total_similarity = 0
    passing_threshold = 0.7  # Consider answers "acceptable" if similarity is above 0.7
    passing_count = 0

    for example in validation_dataset:
        input_text = f"<|startoftext|>Question: {example['text'].split('Question: ')[1].split('\\n')[0]}\nAnswer:"
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_answer = generated_text.split("Answer:")[-1].strip()
        expected_answer = example['text'].split('Answer: ')[1].split("<|endoftext|>")[0].strip()

        # Compute similarity score
        similarity_score = compute_similarity(generated_answer, expected_answer)
        total_similarity += similarity_score
        if similarity_score >= passing_threshold:
            passing_count += 1

        results.append({
            "question": example['text'].split('Question: ')[1].split('\\n')[0],
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "similarity_score": similarity_score
        })

    # Compute final statistics
    avg_similarity = total_similarity / len(results)
    passing_percentage = (passing_count / len(results)) * 100

    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump({
            "average_similarity": avg_similarity,
            "passing_percentage": passing_percentage,
            "detailed_results": results
        }, f, indent=4)

    print(f"Evaluation complete. Average Similarity: {avg_similarity:.4f}")
    print(f"Percentage of 'acceptable' answers: {passing_percentage:.2f}%")
    print(f"Results saved to {output_file}")

    return avg_similarity, passing_percentage

# ---------------------
# GPT Training Logic
# ---------------------
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

    # 6. Evaluate
    validation_dataset = load_validation_dataset()

    # 7. Train
    print("Starting GPT training...")
    trainer.train()
    print("GPT training complete.")

    # 8. Evaluate
    predictions = evaluate_gpt(model, tokenizer, validation_dataset)
    for pred in predictions[:5]:  # Print first 5 results
        print(f"Q: {pred['question']}\nGPT-2 Answer: {pred['generated_answer']}\nExpected: {pred['expected_answer']}\n")

    # 9. Save the model & tokenizer
    model.save_pretrained("./gpt-trained-model")
    tokenizer.save_pretrained("./gpt-trained-model")

    # 10. Save to Server
    model.save_pretrained("./web/server/gpt-trained-model")
    tokenizer.save_pretrained("./web/server/gpt-trained-model")

    print("GPT model saved to ./gpt-trained-model")


# ---------------------
# Main Entry
# ---------------------
if __name__ == "__main__":
    # Option A: Just call both
    # train_t5()
    train_gpt()
