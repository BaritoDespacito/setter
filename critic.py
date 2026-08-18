"""
A separate, smaller model that judges a *finished* route: given the full hold
sequence and angle, predicts what difficulty it actually looks like. Unlike Setter
(which is causal - it can only see what's been placed so far, by necessity of
generating token-by-token), the critic sees the whole route at once and attends in
both directions, since judging a finished route doesn't have the "don't look ahead"
constraint generation does.

Used to rerank generate_route()'s candidate routes by actual predicted difficulty
match, instead of the cruder "closest to predicted hold count" proxy.
"""
import math
import torch
import torch.nn as nn

from setter import (
    VOCAB_SIZE,
    PAD_TOKEN_ID,
    PositionalEncoding,
    build_coord_table,
)


class RouteCritic(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=192, nhead=6, num_layers=4):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.register_buffer("coord_table", build_coord_table(vocab_size))
        self.coord_embed = nn.Linear(2, d_model)
        self.angle_embed = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
        )
        # enable_nested_tensor=False: TransformerEncoder's nested-tensor fast path (used
        # automatically with a padding mask) hits an operator that isn't implemented on
        # MPS (Apple Silicon GPU); this disables that optimization rather than silently
        # crashing on this hardware or needing a slower global CPU-fallback env var.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, route_ids, angle):
        """
        route_ids: (batch, seq_len) - full route token sequence (START...holds...END),
                   right-padded with PAD_TOKEN_ID.
        angle: (batch,) - normalized angle.
        Returns: (batch,) - predicted normalized difficulty.
        """
        padding_mask = route_ids == PAD_TOKEN_ID

        emb = self.embedding(route_ids) * math.sqrt(self.d_model)
        emb = self.pos_encoding(emb)
        emb = emb + self.coord_embed(self.coord_table[route_ids])
        emb = emb + self.angle_embed(angle.unsqueeze(-1)).unsqueeze(1)

        encoded = self.encoder(emb, src_key_padding_mask=padding_mask)

        # Masked average pool over real (non-pad) positions.
        real = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)

        return self.output_head(pooled).squeeze(-1)


def load_critic(path, device="cpu", **kwargs):
    """Loads a RouteCritic checkpoint, or returns None if the file doesn't exist -
    callers can treat a missing critic as "just don't use one" rather than crashing."""
    import os
    if not path or not os.path.exists(path):
        return None
    model = RouteCritic(**kwargs).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
