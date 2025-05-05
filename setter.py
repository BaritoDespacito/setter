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
        # 1. Embed holds and conditions
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        grade_emb = self.grade_embed(grade.unsqueeze(-1))
        angle_emb = self.angle_embed(angle.unsqueeze(-1))
        x = x + grade_emb.unsqueeze(1) + angle_emb.unsqueeze(1)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)

        # 2. Generate masks
        src_pad_mask = (input_ids == PAD_TOKEN_ID) if input_ids is not None else None

        if labels is not None:
            # Training mode (teacher forcing)
            tgt_seq_len = labels.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(x.device)

            # Prepare decoder input
            decoder_input = torch.cat([
                torch.full((labels.size(0),), START_TOKEN_ID, device=x.device).unsqueeze(1),
                           labels[:, :-1]
            ], dim=1)

            decoder_emb = self.embedding(decoder_input) * (self.d_model ** 0.5)
            decoder_emb = decoder_emb.transpose(0, 1)

            out = self.transformer(
                src=x,
                tgt=decoder_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
        else:
            # Inference mode - key fix here
            out = self.transformer(
                src=x,
                tgt=x,  # Use src as tgt for decoder-free generation
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=src_pad_mask,
            )

        return self.fc_out(out.transpose(0, 1))