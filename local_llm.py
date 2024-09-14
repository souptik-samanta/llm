from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add special tokens
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Function to load a text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load the text file
file_path = 'julius_caesar.txt'
julius_caesar_text = load_text_file(file_path)

# Function to interact with the language model
def interact_with_llm(prompt, model, tokenizer, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    attention_mask = inputs['attention_mask']  # Create the attention mask
    outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interact with the model
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add the prompt
    prompt = f"Julius Caesar text: {julius_caesar_text[:500]} \n\nQuestion: {user_input} Answer:"
    
    # Generate response from the model
    response = interact_with_llm(prompt, model, tokenizer)
    print(f"LLM: {response}")
