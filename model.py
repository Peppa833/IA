import torch
import torch.nn as nn

class NeuralChat(nn.Module):
    """
    Modelo conversacional mejorado
    - Embeddings más grandes para mejor representación
    - 2 capas GRU para más capacidad
    - Dropout para evitar overfitting
    """
    def __init__(self, vocab_size, embed=128, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embed)
        self.dropout1 = nn.Dropout(dropout)
        
        # Capas recurrentes (GRU)
        self.rnn = nn.GRU(
            embed, 
            hidden, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de salida
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.dropout1(x)
        
        # RNN
        out, _ = self.rnn(x)
        
        # Salida
        out = self.dropout2(out)
        return self.fc(out)