import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model) # Converts input token IDs (integers) into dense vectors of size d_model
        self.pos_embedding = nn.Embedding(512, d_model) # Positional embeddings for sequence order (allows the transformer to understand the order of tokens)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout) #sets up encoder layer,
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size) # Maps final vector output to the vocabulary

    def forward(self, x):
        seq_len = x.size(1) # Gets length of input sequence
        batch_size = x.size(0) # Gets batch size
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) 
        # torch.arrange creates a 1D tensor of positions from 0 to seq_len
        # .unsqueeze adds a dimension to the tensor, making it 2D and matching the batch dimension
        # .expand replicates the tensor for each batch size
        
        x = self.token_embedding(x) + self.pos_embedding(positions) # Adds token embeddings and positional embeddings
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1) #forces the tensor to only use current and previous tokens
        x = self.transformer(x.transpose(0, 1), mask=mask)  # Transformer expects [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)  # Convert back to [batch_size, seq_len, d_model] (swaps positions 0 and 1)

        logits = self.fc_out(x)  # Projecs output embeddings to vocab logits
        return logits
