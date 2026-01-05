import torch
import torch.nn as nn
import math

# SPECIAL TOKENS
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
START_TOKEN_ID = 4
END_TOKEN_ID = 5
PAD_TOKEN_ID = 6

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Setter(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Conditioning: create rich embeddings for grade/angle
        self.condition_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Use TransformerDecoder with cross-attention to conditioning
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, grade, angle, labels=None):
        """
        Args:
            input_ids: (batch, seq_len) - input sequence tokens
            grade: (batch,) - normalized grade values
            angle: (batch,) - normalized angle values
            labels: (batch, label_len) - target sequence tokens for training

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = input_ids.size(0)

        # 1. Create conditioning vector as "memory" for cross-attention
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond_emb = self.condition_embed(conditions).unsqueeze(1)  # (batch, 1, d_model)

        # 2. Determine target sequence
        if labels is not None:
            # Training mode: teacher forcing
            # Shift labels right and prepend START token
            tgt_ids = torch.cat([
                torch.full((batch_size, 1), START_TOKEN_ID, device=labels.device, dtype=torch.long),
                labels[:, :-1]
            ], dim=1)
        else:
            # Inference mode: use input_ids as target
            tgt_ids = input_ids

        # 3. Embed target sequence with positional encoding
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 4. Create causal mask to prevent attending to future tokens
        tgt_len = tgt_emb.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=tgt_emb.device
        )

        # 5. Create padding mask for target sequence
        tgt_padding_mask = (tgt_ids == PAD_TOKEN_ID)

        # 6. Apply transformer decoder
        # tgt = target sequence, memory = conditioning (grade/angle)
        # Cross-attention allows model to attend to conditioning at each step
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=cond_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 7. Project to vocabulary
        logits = self.fc_out(out)

        return logits