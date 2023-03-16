# First load the dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

text = None  # This is loaded from dany dataset.
chars = sorted(list(set(text)))  # possible elements from the dataset.
vocab_size = len(chars)

# Tokenize the set(Character level tokenizer)
string_toInt = {ch: i for i, ch in enumerate(chars)}
int_toString = {i: ch for i, ch in enumerate(chars)}

# Forward and Reverse Mapping(String to Integer). There are different schemes, Google uses sentencepiece,
# OpenAI uses tiktoken
encode = lambda string: [string_toInt[c] for c in string]
decode = lambda lst: ''.join([int_toString[i] for i in lst])

data = torch.tensor(encode(text), dtype=torch.long)
N = int(len(data)*0.9)

trainData = data[N:]
valData = data[:N]
context_length = 10
batch_size = 4     # how many independent sequences will we process in parallel
eval_iters = 200
torch.manual_seed(69)
n_embd = 32


@torch.no_grad()
def estimate_loss():
    """[Average loss across multiple batches ; Much more resistant to fluctuations.]"""
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - context_length, (batch_size,))  # random offsets in the dataset.
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1: i+context_length+1] for i in ix])
    return x, y


xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(context_length):
        context = xb[b, :t+1]
        target = yb[b, t]


class BigramLanguageModel(nn.module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)   # 4 heads of 8 dimensional self attention
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """[Index and targets are both batch and time(context) tensor of integers. ]"""
        # Logits are basically the scores for the next character in the sequence.
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)   # B, T, vocab_size
        # Implement the loss functions. Pytorch needs the Channel thing first.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]   # (B,C)
            probs = F.softmax(logits, dims=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)   # from (B,T) to (B,T+1) in one iteration.
        return idx


m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
EPOCHS = 32
for steps in range(EPOCHS):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# How Transformers come into picture? First, let's see average context prediction(bag of words)
B, T, C = batch_size, context_length, 32
x = torch.randn(B, T, C)
xbow = torch.zeros((B, T, C))

# We can be a lot of efficient if we use matrix multiplication. Triangular Matrix to the rescue !!!!
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
xbow2 = weights @ x   # (B, T, T) @ (B, T, C) which gives (B, T, C) matrix !!
# let's implement self-attention !
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Kind of every token generates a Query and a Key, and then dot product is used to communicate relation between them
k = key(x)   # (B, T, head_size)
q = query(x)   # (B, T, head_size)

wei = q @ k.transpose(-2, -1)  # (B, T, head_size) x (B, head_size, T) --> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))    # This ensures that we do not interact with future token.
wei = F.softmax(wei, dim=-1)     # This ensures that the distribution of tokens is normalized.

v = value(x)
out = wei @ v    # (B, T, T) @ (B, T, C) which gives (B, T, C) matrix !!


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self._key = nn.Linear(n_embd, head_size, bias=False)
        self._query = nn.Linear(n_embd, head_size, bias=False)
        self._value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k = self._key(x)  # (B, T, head_size)
        q = self._query(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # divided by sqrt(head size) to control the variance.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # TODO: Do not interact with future token.
        wei = F.softmax(wei, dim=-1)  # TODO: This ensures that the distribution of tokens is normalized.

        v = self._value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) which gives (B, T, C) matrix !!
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple Scaled Self Attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead[head_size] for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h[x] for h in self.heads], dim=-1)


class Block(nn.Module):
    """[Transformer block: communication followed by computation]"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)     # communication
        self.ffwd = FeedForward(n_embd)                    # computation

    def forward(self, x):
        x = x+ self.sa(x)
        x = x+  self.ffwd(x)
        return x

