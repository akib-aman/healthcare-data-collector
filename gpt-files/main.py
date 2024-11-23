from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

file_path = "./training-datasets/age-training.json"

# Load the JSON data
with open(file_path, "r") as file:
    train_data_ages = json.load(file)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list([
    {"text": f"<|startoftext|>{ex['input']}<|endoftext|>{ex['output']}<|endoftext|>"}
    for ex in train_data_ages
])

# Load and configure the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add the pad token

# Tokenize the dataset with potentially better padding strategy
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="longest", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set up training arguments
training_args = TrainingArguments(
    num_train_epochs=5,  # Increase epochs
    per_device_train_batch_size=4,  # Use smaller batches if memory-constrained
    learning_rate=5e-5,  # Lower learning rate for stability
    evaluation_strategy="no",
    eval_steps=100,
    save_steps=200,  # Save checkpoints more frequently
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