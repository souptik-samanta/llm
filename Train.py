import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import Dataset
import os
import json
from google.colab import files

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
MODEL_PATH = './trained_model'
VOCAB_PATH = './vocab'
EPOCHS = 2

# Function to load the dataset
def load_text_dataset(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read().split("\n")
    return Dataset.from_dict({"text": text})

# Function to encode and decode text
def prepare_data(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 90% train, 10% validation
    train_data = data[:n]
    val_data = data[n:]
    
    # Save vocab, stoi, and itos
    if not os.path.exists(VOCAB_PATH):
        os.makedirs(VOCAB_PATH)
        
    with open(f"{VOCAB_PATH}/vocab.json", "w") as vocab_file:
        json.dump({"chars": chars}, vocab_file)
    with open(f"{VOCAB_PATH}/encode.json", "w") as encode_file:
        json.dump(stoi, encode_file)
    with open(f"{VOCAB_PATH}/decode.json", "w") as decode_file:
        json.dump(itos, decode_file)
    
    return vocab_size, train_data, val_data, encode, decode

# Data loading function
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def _init_(self, head_size):
        super()._init_()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def _init_(self, num_heads, head_size):
        super()._init_()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def _init_(self, n_embd):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def _init_(self, n_embd, n_head):
        super()._init_()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def _init_(self, vocab_size):
        super()._init_()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training function
def train_model(model, train_data, val_data, num_epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for iter in range(max_iters):
            xb, yb = get_batch(train_data)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter % eval_interval == 0 or iter == max_iters - 1:
                print(f"Epoch {epoch + 1}, Step {iter}: loss {loss.item():.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), f'{MODEL_PATH}/gpt_epoch_{epoch + 1}.pth')

        # Generate a sample after each epoch
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(model.generate(context, max_new_tokens=100))

# Main execution
if _name_ == "_main_":
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    # Upload dataset
    uploaded = files.upload()
    
    # Load dataset and prepare data
    with open("julius_caesar.txt", "r", encoding="utf-8") as f:
        text = f.read()

    vocab_size, train_data, val_data, encode, decode = prepare_data(text)

    # Create the model
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    # Train the model
    train_model(model, train_data, val_data)

    print("Training complete! Model and vocab saved.")
