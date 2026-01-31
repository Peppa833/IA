def tokenize(text):
    return text.lower().split()

def build_vocab(text):
    tokens = tokenize(text)
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos
