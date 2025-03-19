import os
import json
import torch
import re
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # The directory of the current script

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
    """
    Train a T5 model for extracting fields (age, name, sex, etc.).
    Saves the model & tokenizer to ./t5-trained-model
    """
    
    # Helper function to return conversion logic.
    # For extraction tasks, simply pass the raw input.
    # For classification, use the full prompt.
    def get_conversion_logic(classification=False):
        if classification:
            return lambda ex: {
                "input": (
                    "Does this input require data extraction or a detailed response?\n"
                    f"Input: {ex['input']}\n"
                    "Answer with either 'Data Extraction' or 'Detailed Response'."
                ),
                "output": ex["output"]
            }
        else:
            return lambda ex: {
                "input": ex["input"],
                "output": ex["output"]
            }
    
    # Configuration for T5 dataset paths
    dataset_paths = {
        "firstname_training": {
            "file_path": "./t5-training-datasets/firstname-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "lastname_training": {
            "file_path": "./t5-training-datasets/lastname-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "age_training": {
            "file_path": "./t5-training-datasets/age-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "sex_training": {
            "file_path": "./t5-training-datasets/sex-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "classification_training": {
            "file_path": "./t5-training-datasets/classification-training.json",
            "conversion_logic": get_conversion_logic(True)
        },
        "religion_training": {
            "file_path": "./t5-training-datasets/religion-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "title_training": {
            "file_path": "./t5-training-datasets/title-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "disability_training": {
            "file_path": "./t5-training-datasets/disability-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "ethnicity_training": {
            "file_path": "./t5-training-datasets/ethnicity-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "gender_reassignment_training": {
            "file_path": "./t5-training-datasets/gender-reassignment-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "marriage_civilrelationship_training": {
            "file_path": "./t5-training-datasets/marriage-civilrelationship-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "sexual_orientation_training": {
            "file_path": "./t5-training-datasets/sexual-orientation-training.json",
            "conversion_logic": get_conversion_logic(False)
        },
        "pregnancy_training": {
            "file_path": "./t5-training-datasets/pregnancy-training.json",
            "conversion_logic": get_conversion_logic(False)
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
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

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
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

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

    # Evaluate Classification
    testing_classification_dataset = load_testing_classification_dataset() 
    testing_extraction_dataset = load_testing_extraction_dataset()
    evaluate_t5(model, tokenizer, testing_classification_dataset, "evaluations/t5/t5-base-classification-validation-results.json")
    evaluate_t5(model, tokenizer, testing_extraction_dataset, "evaluations/t5/t5-base-extraction-validation-results.json")

    # Save T5 model & tokenizer
    model.save_pretrained(os.path.join(BASE_DIR, "t5-trained-model"))
    tokenizer.save_pretrained(os.path.join(BASE_DIR, "t5-trained-model"))

    # Save to Server
    model.save_pretrained(os.path.join(BASE_DIR, "./web/server/t5-trained-model"))
    tokenizer.save_pretrained(os.path.join(BASE_DIR, "./web/server/t5-trained-model"))
    print("T5 model saved to ./t5-trained-model")


# ---------------------
# Cosine Similarity with Sbert
# ---------------------
def compute_similarity(generated, expected):
    """Compute cosine similarity between generated and expected answer embeddings"""
    embeddings = sbert_model.encode([generated, expected])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity_score

# ---------------------
# F1 Scores
# ---------------------
def compute_f1(generated, expected):
    """
    Compute a simple token-level F1 score between generated and expected answers.
    This function splits the strings into tokens, calculates precision and recall based on common tokens,
    and then computes the harmonic mean.
    """
    gen_tokens = generated.split()
    exp_tokens = expected.split()
    if len(gen_tokens) == 0 or len(exp_tokens) == 0:
        return 0.0
    common_tokens = set(gen_tokens) & set(exp_tokens)
    if len(common_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(gen_tokens)
    recall = len(common_tokens) / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)

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
# t5 Test Logic
# ---------------------
def load_testing_extraction_dataset(filename):
    """
    Load the T5 extraction testing dataset, converting it into the required format.
    This version directly uses the "input" and "output" keys from the dataset.
    """
    validation_file_path = "./t5-training-datasets/" + filename

    with open(validation_file_path, "r", encoding="utf-8") as file:
        validation_data = json.load(file)

    validation_dataset = Dataset.from_list([
        {
            "input": ex["input"],
            "output": ex["output"]
        }
        for ex in validation_data
    ])

    return validation_dataset

# ---------------------
# t5 Test Logic
# ---------------------
def load_testing_classification_dataset(filename):
    """
    Load the T5 classification testing dataset, converting it into the required format.
    """
    validation_file_path = "./t5-training-datasets/" + filename

    with open(validation_file_path, "r", encoding="utf-8") as file:
        validation_data = json.load(file)

    validation_dataset = Dataset.from_list([
        {
            "input": (
                "Does this input require data extraction or a detailed response?\n"
                f"Input: {ex['input']}\n"
                "Answer with either 'Data Extraction' or 'Detailed Response'."
            ),
            "output": ex["output"]
        }
        for ex in validation_data
    ])

    return validation_dataset

# ---------------------
# GPT Test Logic
# ---------------------
def load_testing_dataset():
    validation_file_path = "./gpt-training-datasets/question-answer-testing.json"

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
# Trim Last Sentence
# ---------------------
def trim_last_sentence(text):
    """
    Trims the last sentence from the text by removing everything 
    after the last period (".").
    """
    match = re.search(r'^(.*?\.)[^.]*$', text, re.DOTALL)
    return match.group(1).strip() if match else text  # Return trimmed text or original if no period found

# ---------------------
# T5 Test Logic
# ---------------------
def evaluate_t5(model, tokenizer, dataset, output_file):
    """
    Evaluate a T5 model using a provided dataset. The dataset is a list of examples with keys:
       "input": a prompt for the T5 model,
       "output": the expected answer.  
    Results are saved to the specified output_file.
    """
    print("Starting T5 Evaluation...")
    model.eval()
    results = []
    total_similarity = 0
    total_f1 = 0
    passing_threshold = 0.7  # Example threshold for acceptable similarity
    passing_count = 0
    
    for example in dataset:
        input_text = example['input']
        expected_answer = example['output']
        
        # Tokenize the input
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate output from T5
        output_ids = model.generate(inputs, max_length=50, num_beams=5)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Optionally: you could trim extra sentences if needed (see your trim_last_sentence() function)
        generated_answer = generated_text
        
        # Compute cosine similarity between generated and expected answers
        similarity_score = compute_similarity(generated_answer, expected_answer)
        total_similarity += similarity_score
        
        # Compute token-level F1 score
        f1_score = compute_f1(generated_answer, expected_answer)
        total_f1 += f1_score
        
        if similarity_score >= passing_threshold:
            passing_count += 1
        
        results.append({
            "input": input_text,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "similarity_score": float(similarity_score),
            "f1_score": float(f1_score)
        })
    
    avg_similarity = total_similarity / len(dataset)
    avg_f1 = total_f1 / len(dataset)
    passing_percentage = (passing_count / len(dataset)) * 100
    
    output_data = {
        "average_similarity": float(avg_similarity),
        "average_f1": float(avg_f1),
        "passing_percentage": float(passing_percentage),
        "detailed_results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Evaluation complete. Average Similarity: {avg_similarity:.4f}, Average F1: {avg_f1:.4f}")
    print(f"Percentage of 'acceptable' answers: {passing_percentage:.2f}%")
    print(f"Results saved to {output_file}")
    
    return avg_similarity, avg_f1, passing_percentage, results

# ---------------------
# GPT Evaluate
# ---------------------
def evaluate_gpt(model, tokenizer, dataset_to_use, output_file):
    print("Starting GPT Evaluation...")
    model.eval()
    results = []
    total_similarity = 0
    total_f1 = 0
    passing_threshold = 0.7  # Consider answers "acceptable" if similarity is above 0.7
    passing_count = 0

    for example in dataset_to_use:
        # Extract question text safely
        question_text = example['text'].split('Question: ')[1].split('\n')[0]
        input_text = f"<|startoftext|>Question: {question_text}\nAnswer:"
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        output = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            min_length=30,
            do_sample=True,
            top_k=50,
            top_p=0.85,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,  # discourage repetition
            no_repeat_ngram_size=2,  # block 2-gram repeats
            early_stopping=False
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract generated answer from the generated text
        generated_answer = generated_text.split("Answer:")[-1].strip()
        
        # First, trim the last sentence
        generated_answer = trim_last_sentence(generated_answer)
        # Now compute lowercase version for keyword matching
        lower_text = generated_answer.lower()

        keywords = ["what happens if i ", "learn more about", "however,", "for example,", "for more information,"]
        # Create regex pattern to match any of the keywords (case-insensitive)
        pattern = r'(?i)(' + '|'.join(re.escape(kw) for kw in keywords) + r').*'
        match = re.search(pattern, lower_text)
        if match:
            generated_answer = generated_answer[:match.start()].strip()
        
        expected_answer = example['text'].split('Answer: ')[1].split("<|endoftext|>")[0].strip()

        similarity_score = compute_similarity(generated_answer, expected_answer)
        total_similarity += similarity_score
        
        # Compute token-level F1 score
        f1_score = compute_f1(generated_answer, expected_answer)
        total_f1 += f1_score

        if similarity_score >= passing_threshold:
            passing_count += 1

        results.append({
            "question": question_text,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "similarity_score": float(similarity_score),
            "f1_score": float(f1_score)
        })

    avg_similarity = float(total_similarity / len(results))
    avg_f1 = float(total_f1 / len(results))
    passing_percentage = float((passing_count / len(results)) * 100)

    with open(output_file, "w") as f:
        json.dump({
            "average_similarity": avg_similarity,
            "average_f1": avg_f1,
            "passing_percentage": passing_percentage,
            "detailed_results": results
        }, f, indent=4)

    print(f"Evaluation complete. Average Similarity: {avg_similarity:.4f}, Average F1: {avg_f1:.4f}")
    print(f"Percentage of 'acceptable' answers: {passing_percentage:.2f}%")
    print(f"Results saved to {output_file}")

    return avg_similarity, avg_f1, passing_percentage, results

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
        eval_dataset=load_validation_dataset(),
        data_collator=data_collator
    )


    # 7. Train
    print("Starting GPT training...")
    trainer.train()
    print("GPT training complete.")

    # 8. Evaluate
    validation_dataset = load_validation_dataset() 
    evaluate_gpt(model, tokenizer, validation_dataset, "evaluations/gpt2/gpt2-medium-validation-results-iter-3.json")

    # 9. Save the model & tokenizer
    model.save_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))
    tokenizer.save_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))

    # 10. Save to Server
    model.save_pretrained(os.path.join(BASE_DIR, "./web/server/gpt-trained-model"))
    tokenizer.save_pretrained(os.path.join(BASE_DIR, "./web/server/gpt-trained-model"))

    print("GPT model saved to ./gpt-trained-model")

# ---------------------
# Main Entry
# ---------------------
if __name__ == "__main__":
    # train_t5()
    # train_gpt()

    # gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))
    # gpt_model = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))
    # testing_dataset = load_testing_dataset() 

    # evaluate_gpt(gpt_model, gpt_tokenizer, testing_dataset, "evaluations/gpt2/gpt2-medium-test-results.json")

    # T5
    t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(BASE_DIR, "t5-trained-model"))
    t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, "t5-trained-model")).to(device)
    testing_classification_dataset = load_testing_classification_dataset("t5-classification-testing.json") 
    load_extraction_dataset = load_testing_extraction_dataset("t5-extraction-testing.json") 

    evaluate_t5(t5_model, t5_tokenizer, testing_classification_dataset, "evaluations/t5/t5-base-classification-test-results.json")
    evaluate_t5(t5_model, t5_tokenizer, load_extraction_dataset, "evaluations/t5/t5-base-extraction-test-results.json")

