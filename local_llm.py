from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad token id for the model
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Load text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load Julius Caesar text
julius_caesar_text = """
ANTONY.
Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar...
"""

# Add a prompt for the model
def interact_with_llm(prompt, model, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    attention_mask = inputs['attention_mask']
    outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Use a more specific prompt with context
    prompt = f"Based on the following text from Julius Caesar:\n\n{julius_caesar_text}\n\nQuestion: {user_input} Answer:"

    # Get LLM response
    response = interact_with_llm(prompt, model, tokenizer)
    print(f"LLM: {response}")
