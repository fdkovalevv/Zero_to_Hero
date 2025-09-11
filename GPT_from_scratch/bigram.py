import torch
import torch.nn as nn
from torch.nn import functional as F

#--------------
batch_size = 4
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iters = 10000
eval_interval = 300
#--------------

torch.manual_seed(42)

with open('GPT_from_scratch\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
i2s = {s:i for i, s in enumerate(chars)}
s2i = {i:s for s, i in i2s.items()}
encoder = lambda s: [i2s[char] for char in s]
decoder = lambda s: [s2i[char] for char in s]
vocab_size = len(chars)

data = torch.tensor(encoder(text), dtype=torch.long)
train_size = int(0.9 * data.shape[0])
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def validate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BiGramNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BiGramNN()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(max_iters):

    if i % eval_interval == 0:
        losses = validate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(''.join(decoder(
    model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0]
    .to('cpu')
    .tolist()
)))
