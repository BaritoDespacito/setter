import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM

# SPECIAL TOKENS
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
START_TOKEN_ID = 4
END_TOKEN_ID = 5
PAD_TOKEN_ID = 6

class Setter(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.grade_embed = nn.Linear(1, d_model)  # Scalar → d_model
        self.angle_embed = nn.Linear(1, d_model)  # Scalar → d_model

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, grade, angle, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Hold sequences
            grade: (batch_size,) - Normalized V-grade (0-1)
            angle: (batch_size,) - Normalized wall angle (0-1)
            labels: (batch_size, seq_len) - Target holds (for training)
        """
        # Embed holds
        x = self.embedding(input_ids) * (self.d_model ** 0.5)  # (batch, seq, d_model)

        # Embed conditions and add to every position
        grade_emb = self.grade_embed(grade.unsqueeze(-1))  # (batch, d_model)
        angle_emb = self.angle_embed(angle.unsqueeze(-1))  # (batch, d_model)
        x = x + grade_emb.unsqueeze(1) + angle_emb.unsqueeze(1)  # Broadcast

        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        # Generate masks
        src_mask = self.transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        src_pad_mask = (input_ids == PAD_TOKEN_ID)

        # Forward pass
        if labels is not None:
            # Teacher forcing (shift labels for decoder input)
            decoder_input = labels[:, :-1]
            decoder_input = torch.cat([
                torch.full((labels.size(0), 1), START_TOKEN_ID, device=labels.device),
                decoder_input
            ], dim=1)

            decoder_emb = self.embedding(decoder_input) * (self.d_model ** 0.5)
            decoder_emb = decoder_emb.transpose(0, 1)

            out = self.transformer(
                src=x,
                tgt=decoder_emb,
                tgt_mask=src_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=(decoder_input == PAD_TOKEN_ID),
            )
        else:
            # Inference mode (autoregressive)
            out = self.transformer(
                src=x,
                tgt=None,
                src_key_padding_mask=src_pad_mask,
            )

        # Predict next holds
        out = self.fc_out(out.transpose(0, 1))  # (batch, seq, vocab_size)
        return out