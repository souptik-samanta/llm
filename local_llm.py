import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer (GPT-2 in this case)
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Move model to GPU if available
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token to eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Load the full text of Julius Caesar
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

julius_caesar_text = load_text_file("julius_caesar.txt")

# Split the text into chunks (e.g., by scenes or every 1000 characters)
chunk_size = 1000  # Set chunk size based on your memory limits
text_chunks = [julius_caesar_text[i:i + chunk_size] for i in range(0, len(julius_caesar_text), chunk_size)]

# File to store the summaries
summary_file = "julius_caesar_summaries.json"

# Load existing summaries if the file already exists
if os.path.exists(summary_file):
    with open(summary_file, "r") as f:
        summaries = json.load(f)
else:
    summaries = {}

# Function to interact with LLM on each chunk
def interact_with_llm(prompt, model, tokenizer, max_new_tokens=150):
    # Tokenize on CPU (this can be done faster on CPU)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move tokenized inputs to GPU for inference
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate outputs on the GPU
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1
    )
    
    # Decode the output and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Process and store summaries if not already processed
for i, chunk in enumerate(text_chunks):
    chunk_id = f"chunk_{i+1}"
    
    if chunk_id not in summaries:
        print(f"Processing chunk {i+1}/{len(text_chunks)}")
        prompt = f"Here is the next part of Julius Caesar:\n\n{chunk}\n\nSummarize this text:"
        
        # Get LLM response
        response = interact_with_llm(prompt, model, tokenizer)
        
        # Save summary in memory
        summaries[chunk_id] = {
            "chunk_text": chunk,
            "summary": response
        }
        
        # Print the summary (optional)
        print(f"Chunk {i+1} Summary: {response}\n")
        
        # Save to file after processing each chunk
        with open(summary_file, "w") as f:
            json.dump(summaries, f, indent=4)

print("All chunks processed and stored.")

# Now you have the summaries stored, and you can reload them whenever needed.
import json

# Load summaries from the file
def load_summaries(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

summaries = load_summaries("julius_caesar_summaries.json")

# Example: Ask a question based on the summaries
def ask_question(question, summaries):
    print(f"Question: {question}")
    
    # You can adjust this to look for specific parts of the play.
    for chunk_id, data in summaries.items():
        # Use the summary to generate a response or search in it
        summary = data['summary']
        print(f"{chunk_id} Summary:\n{summary}\n")
        
    # For a specific response, you could analyze or process the summaries further
    # For now, this prints out all stored summaries as a simple response.

# Ask a question (example)
ask_question_with_answer("What happened in Act 1?", summaries)
