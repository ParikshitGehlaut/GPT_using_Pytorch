import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

with open('wizard_of_oz.txt','r', encoding='utf-8') as f:
  text = f.read()
print(len(text))

import tiktoken

encoder = tiktoken.get_encoding('gpt2')

# Get the vocabulary size from the BPE encoder
vocab_size = encoder.n_vocab

# Define the encode and decode functions
encode = lambda s: encoder.encode(s)
decode = lambda l: encoder.decode(l)

torch.manual_seed(1337)
#   ----------------------- Hyperparameter ----------------------------

vocab_size = 50304 # vocab_size of tiktoken tokenization is 50257, 50304 is closest multiple of 64
n_embd = 256   # dimension of embeddings
block_size = 256 # number of tokens that are processed to predict next tokens
batch_size = 32  # 32 set of 256 tokens are processed at one time
learning_rate = 3e-3 # learning rate
n_head = 16
max_iters = 5000
eval_interval = 1000
eval_iters = 200
dropout = 0.4

data = torch.tensor(encode(text), device = device)

# Train Test Splitting
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# print(train_data.shape)   torch.Size([204345])
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
# print(xb)
print('targets:')
print(yb.shape)
# print(yb)
print('----')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.Key = nn.Linear(n_embd, head_size, bias = False)
    self.Value = nn.Linear(n_embd, head_size, bias = False)
    self.Query = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size), device = device)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    q = self.Query(x) # (16, 32, 16)
    k = self.Key(x)# (16, 32, 16)

    wei = q @ k.transpose(-2, -1) * C**-0.5 # (16, 32, 16) @ (16, 16, 32) --> (16, 32, 32)
    tril = torch.tril(torch.ones((T, T), device = device))
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = torch.softmax(wei, dim = -1)
    wei = self.dropout(wei)
    v = self.Value(x) # (16, 32, 16)

    out = wei @ v # (16, 32, 32) @ (16, 32, 16) --> (16, 32, 16)

    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.GELU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class DecoderBlock(nn.Module):
  def __init__(self, n_embd, n_head ):
    super().__init__()
    head_size = n_embd // n_head
    self.sa_heads = MultiHeadAttention(n_head, head_size) # 8 heads for 512 dim attention
    self.ffwd = FeedForward(n_embd)
    self.l1 = nn.LayerNorm(n_embd)
    self.l2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_heads(self.l1(x)) #  plus x for residual connections
    x = x + self.ffwd(self.l2(x))
    return x

def precompute_theta_pos_frequencies(head_dim, seq_len, theta = 10000.0):
  assert seq_len % 2 == 0, "seq_len should be even"
  theta_numerator = torch.arange(0, head_dim, 2).float()
  theta = 1 / (theta ** (theta_numerator/head_dim)).to(device)
  m = torch.arange(seq_len, device = device)
  freqs = torch.outer(m, theta).float()
  freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_complex

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.freqs = torch.pow(10000, -torch.arange(0, d_model, 2).float() / d_model).to(device)
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(0, seq_len, device=x.device).float()
        angles = position.unsqueeze(1) * self.freqs.unsqueeze(0)
        angles = angles.view(-1, self.d_model // 2)
        angles = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        position_enc = self.pe(position.long())
        return self.dropout(position_enc + angles)

class GPT_Language_model(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = RotaryEmbedding(n_embd)
        self.blocks = nn.Sequential(
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
            DecoderBlock(n_embd, n_head ),
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_enc = self.positional_embedding(tok_emb)
        x = tok_emb + pos_enc
        x = self.blocks(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT_Language_model()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(), lr= learning_rate, weight_decay = 0.01)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'GPT_Language_model.pth')

model = BigramLanguageModel()
model.load_state_dict(torch.load('GPT_Language_model.pth'))

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

