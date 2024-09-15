from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from google.colab import files
import os

# Constants
MODEL_NAME = "gpt2"
MODEL_PATH = './trained_model'
DATASET_FILE = "julius_caesar.txt"

# Function to load the dataset
def load_text_dataset(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read().split("\n")
    return Dataset.from_dict({"text": text})

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Function to train the model
def train_model():
    # Load dataset from uploaded file
    dataset = load_text_dataset(DATASET_FILE)

    # Tokenize the dataset using map for efficient processing
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Data collator for dynamic padding during training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

# Function to generate responses with improved parameters
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Generate the response
    outputs = model.generate(
        inputs['input_ids'],
        max_length=len(prompt) + 50,  # Allow longer responses dynamically
        num_return_sequences=1,
        temperature=0.7,  # Adjust for more deterministic output
        top_p=0.9,        # Nucleus sampling
        top_k=50,         # Limits the sampling to the top-k tokens
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,  # Prevents repetition of n-grams
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main execution
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Fine-tuning the model...")
        # Upload the dataset file
        uploaded = files.upload()

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # Train the model
        train_model()
        print("Model fine-tuning complete. Model saved at:", MODEL_PATH)
    else:
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Continuous querying
    print("Model is ready. Type 'exit' to quit.")
    while True:
        prompt = input("Ask a question: ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        response = generate_response(prompt, model, tokenizer)
        print("Response:", response)
