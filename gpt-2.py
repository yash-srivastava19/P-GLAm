# First load the dataset
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F

with open('math.txt', 'r', encoding='utf-8') as f:   ## add path to the dataset here.
  text = f.read()

chars = sorted(list(set(text)))  # possible elements from the dataset.
vocab_size = len(chars)

print(vocab_size)

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


@dataclass
class TuningParameters:
  context_length = 10  
  batch_size = 16
  eval_interval = 100
  n_embd = 128
  dropout = 0.2
  n_layer = 20
  n_head = 20
  max_iters = 500
  eval_iters = 50
  learning_rate = 3e-4
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = TuningParameters()


@torch.no_grad()
def estimate_loss():
    """[Average loss across multiple batches ; Much more resistant to fluctuations.]"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(params.eval_iters)
        for k in range(params.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    """[Batch up the dataset into x and y. Make sure to do the computations on the device]"""
    data = trainData if split == 'train' else valData
    ix = torch.randint(len(data) - params.context_length, (params.batch_size,))  # random offsets in the dataset.
    x = torch.stack([data[i:i+params.context_length] for i in ix])
    y = torch.stack([data[i+1: i+params.context_length+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device) 
    return x, y


xb, yb = get_batch('train')

for b in range(params.batch_size):
    for t in range(params.context_length):
        context = xb[b, :t+1]
        target = yb[b, t]

class SelfAttentionHead(nn.Module):
    """[Can't thank Vasvani et. al. for this. It's all I need]"""
    def __init__(self, head_size):
        super().__init__()
        self._key = nn.Linear(params.n_embd, head_size, bias=False)
        self._query = nn.Linear(params.n_embd, head_size, bias=False)
        self._value = nn.Linear(params.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(params.context_length, params.context_length)))

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
    """ [Multiple Headed Self Attention in parallel] """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, params.n_embd)
        self.dropout = nn.Dropout(params.dropout) 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ [A Linear Layer followed by a Relu and Dropout] """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(params.dropout),
        )

    def forward(self, x):
        return self.net(x)


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
              
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(params.context_length, params.n_embd)
        self.sa_heads = MultiHeadAttention(4, params.n_embd//4)   # 4 heads of 8 dimensional self attention
        self.lm_head = nn.Linear(params.n_embd, vocab_size)

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
            idx_cond = idx[:, -params.context_length]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]   # (B,C)
            probs = F.softmax(logits, dims=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)   # from (B,T) to (B,T+1) in one iteration.
        return idx

class PGLAm(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, params.n_embd)
        self.position_embedding_table = nn.Embedding(params.context_length, params.n_embd)
        self.blocks = nn.Sequential(*[Block(params.n_embd, n_head=params.n_head) for _ in range(params.n_layer)])
        self.ln_f = nn.LayerNorm(params.n_embd) # final layer norm
        self.lm_head = nn.Linear(params.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=params.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # perplexity = torch.exp(loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -params.context_length:]
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
    
    
model = PGLAm()
m = model.to(params.device)
print(f'{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters')   # currently, a 3.87M Parameter model.

optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

for iter in range(params.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % params.eval_interval == 0 or iter == params.max_iters - 1:
        losses = estimate_loss()
        perplexity = torch.exp(losses)
        print(f"Step: {iter}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}, Perplexity: {perplexity.item()}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=params.device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Uncomment the below line to generate the tokens and write them to the file 'gen_text.txt' as that of 'math.txt' - now 884647, but 69420 also works. 
# Ensure that both the texts are of same token length for easier reproducibility of results.

#open('gen_text.txt', 'w').write(decode(m.generate(context, max_new_tokens=884647)[0].tolist())) 
