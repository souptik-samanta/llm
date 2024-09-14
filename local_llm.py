import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer (GPT-2 in this case)
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Model will run on CPU
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
    # Tokenize the inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate outputs on the CPU
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

# -------- New Function for Asking Questions -------- #
def generate_answer(question, summary, model, tokenizer):
    # Formulate the prompt with the question and relevant summary
    prompt = f"Question: {question}\nSummary: {summary}\nAnswer:"
    
    # Generate the answer using the LLM
    answer = interact_with_llm(prompt, model, tokenizer)
    
    return answer

def ask_question_with_answer(question, summaries, model, tokenizer):
    print(f"Question: {question}")
    
    # Extract keywords from the question to filter relevant summaries
    keywords = question.lower().split()
    
    # Initialize a list to hold relevant summaries
    relevant_summaries = []
    
    # Check each summary for relevance by matching keywords
    for chunk_id, data in summaries.items():
        summary = data['summary']
        
        # If any keyword is found in the summary, mark it as relevant
        if any(keyword in summary.lower() for keyword in keywords):
            print(f"\nRelevant chunk found in {chunk_id}:")
            relevant_summaries.append(summary)
    
    # If relevant summaries are found, ask the model to generate an answer
    if relevant_summaries:
        for summary in relevant_summaries:
            answer = generate_answer(question, summary, model, tokenizer)
            print(f"Answer: {answer}\n")
            return answer  # Return the first relevant answer found
    else:
        print("No relevant information found in the summaries.")


# ------- Interacting with the User ------- #
# After processing, you can now ask questions based on summaries
ask_question_with_answer("What is Cinna's profession?", summaries, model, tokenizer)
