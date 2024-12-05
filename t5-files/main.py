from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# Load your dataset
file_path = "./training-datasets/age-training.json"
with open(file_path, "r") as file:
    train_data_ages = json.load(file)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list([
    {"input": f"Extract the age from this text: {ex['input']}", "output": ex["output"]}
    for ex in train_data_ages
])


# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    
    # Tokenize targets and mask padding tokens for labels
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]
    
    return model_inputs



tokenized_dataset = dataset.map(preprocess_function, batched=True)
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
    logging_dir="./logs",
    logging_steps=100,
    overwrite_output_dir=True,
)

# Use a data collator for Seq2Seq models
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./t5-trained-model")
tokenizer.save_pretrained("./t5-trained-model")
