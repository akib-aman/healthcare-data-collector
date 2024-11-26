from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

file_path = "./training-datasets/age-training.json"

# Load and validate JSON data
with open(file_path, "r") as file:
    train_data_ages = json.load(file)

# Convert to Dataset with clear delimiters
dataset = Dataset.from_list([
    {"text": f"<|startoftext|>Input: {ex['input']}\nOutput: {ex['output']}<|endoftext|>"}
    for ex in train_data_ages
])

# Load and configure the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["<|startoftext|>", "<|endoftext|>"]}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model and resize token embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Set up training arguments
training_args = TrainingArguments(
    num_train_epochs=10,  # Start with fewer epochs
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="no",
    eval_steps=100,
    save_steps=200,
    logging_steps=50,
    output_dir="./results",
    overwrite_output_dir=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
