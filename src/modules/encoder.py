import torch

from src.modules.attention import StackedAttention, create_self_attn_mask
from torch import nn


# -----------------------
# Using custom attention layers.
# Correctness not verified.
# GeoJEPA uses the Encoder class below this one.
# -----------------------


class Encoder2(nn.Module):
    def __init__(
        self,
        token_dim=512,
        depth=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attn = StackedAttention(
            dim=token_dim,
            heads=num_heads,
            depth=depth,
            mlp_ratio=4,
            path_drop=dropout,
            attn_drop=dropout,
            drop=dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos_embs=None, padding_mask=None):
        B, T, _ = x.shape

        if pos_embs is not None:
            x = x + pos_embs
        # x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        attn_mask = create_self_attn_mask(padding_mask)
        enc = self.self_attn.forward(x, mask=attn_mask)

        return enc


class Encoder(nn.Module):
    def __init__(
        self, depth=12, token_dim=1024, num_heads=8, dropout=0.1, reg_tokens=0
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=token_dim,
                dim_feedforward=token_dim
                * 4,  # hidden dim of token-wise MLP after attention
                nhead=num_heads,
                dropout=dropout,
                bias=True,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=depth,
            norm=nn.LayerNorm(token_dim),
        )

        # register tokens for storing global information (VISION TRANSFORMERS NEED REGISTERS)+ T-JEPA REGULARIZATION TOKEN
        self.num_reg_tokens = reg_tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.randn(1, reg_tokens, token_dim))
        # self.number_nan_elements = 0

    def forward(self, x: torch.Tensor, pos_embs=None, padding_mask=None):
        if pos_embs is not None:
            x += pos_embs
        #
        if self.num_reg_tokens > 0:
            reg_tokens = self.reg_tokens.repeat(x.size(0), 1, 1)
            x = torch.cat((reg_tokens, x), dim=1)
            reg_mask = torch.zeros(
                (x.size(0), self.num_reg_tokens), device=x.device, dtype=torch.bool
            )
            padding_mask = torch.cat((reg_mask, padding_mask), dim=1)
        # padding mask for torch transformer uses True for padding tokens, so we need to invert it for x-transformers (False for padding tokens)
        # mask = ~padding_mask
        # feats = self.encoder.forward(x, mask=mask)
        # x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        feats = self.encoder.forward(x, src_key_padding_mask=padding_mask)
        # finite = feats.isfinite()
        # if not finite.all():
        #     feats[~finite] = 0
        #     failed_sample_indices = (~finite).nonzero(as_tuple=False)
        #     print(failed_sample_indices, kwargs)
        #     self.number_nan_elements += 1
        #     print(f"Got non-finite values in encoder output, total failed batch samples: {self.number_nan_elements}, (+{1})")

        if self.num_reg_tokens > 0:
            feats = feats[:, self.num_reg_tokens :]
        return feats


if __name__ == "__main__":
    _ = Encoder()
