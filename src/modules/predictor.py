import torch
from src.modules.attention import (
    StackedCrossAttention,
    create_cross_attn_mask,
)
from src.modules.encoder import Encoder
from src.modules.mlp import MLP
from torch import nn



class Predictor(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        predictor_dim=512,
        use_mlp_projector=False,
        depth=6,
        num_heads=4,
        dropout=0.1,
        num_register_tokens=0,
    ):
        super().__init__()
        if use_mlp_projector:
            self.predictor_dim_projection = MLP(
                token_dim, predictor_dim * 4, predictor_dim
            )
        else:
            self.predictor_dim_projection = nn.Linear(
                token_dim, predictor_dim, bias=True
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.encoder = Encoder(
            token_dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            reg_tokens=num_register_tokens,
        )
        self.token_dim_projection = nn.Linear(predictor_dim, token_dim, bias=True)

    def forward(
        self,
        context_tokens,
        context_pos,
        target_pos,
        context_padding_mask,
        target_padding_mask,
    ):
        # context_tokens: [batch_size, num_context_tokens, token_dim]
        # context_bbox:  [batch_size, num_context_tokens, 4]
        # target_bbox: [batch_size, num_target_tokens, 4]
        B, N_ctx, _ = context_tokens.shape
        _, N_tgt, _ = target_pos.shape

        # compress to predictor_token_dim
        context_tokens = self.predictor_dim_projection(context_tokens)

        target_tokens = self.mask_token.repeat(B, N_tgt, 1)

        predictor_tokens = torch.cat((context_tokens, target_tokens), dim=1)

        target_pos = self.predictor_dim_projection(target_pos)
        context_pos = self.predictor_dim_projection(context_pos)
        pos_embs = torch.cat((context_pos, target_pos), dim=1)
        padding_mask = torch.cat((context_padding_mask, target_padding_mask), dim=1)

        x = self.encoder(predictor_tokens, pos_embs=pos_embs, padding_mask=padding_mask)
        x = x[:, N_ctx:]
        x = self.token_dim_projection(x)
        return x


# -----------------------
# Never managed to get the CrossPredictor to work properly.
# -----------------------
class CrossPredictor(nn.Module):
    def __init__(
        self,
        token_dim=512,
        predictor_dim=256,
        depth=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.connector = nn.Sequential(
            MLP(token_dim, predictor_dim * 4, token_dim, drop=dropout, bias=True),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, predictor_dim, bias=True),
        )
        self.pos_emb = nn.Linear(token_dim, predictor_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.encoder = StackedCrossAttention(
            dim=predictor_dim,
            heads=num_heads,
            depth=depth,
            mlp_ratio=4,
            path_drop=dropout,
            attn_drop=dropout,
            drop=dropout,
        )
        self.out_proj = nn.Linear(predictor_dim, token_dim, bias=True)
        self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        torch.nn.init.zeros_(self.mask_token)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        context_tokens,
        context_pos,
        target_pos,
        context_padding_mask,
        target_padding_mask,
    ):
        B, N_ctx, _ = context_tokens.shape
        _, N_tgt, _ = target_pos.shape
        assert context_pos.shape == context_tokens.shape
        assert target_pos.shape[:2] == target_padding_mask.shape
        assert context_pos.shape[:2] == context_padding_mask.shape

        # ctx tokens
        ctx = self.connector(context_tokens)
        ctx = ctx + self.pos_emb(context_pos)

        # target tokens
        m = self.mask_token.repeat(B, N_tgt, 1)
        m = m + self.pos_emb(target_pos)

        # prepend cls token
        # m = torch.cat((self.cls_token.expand(B, -1, -1), m), dim=1)

        # decode
        self_mask, cross_mask = create_cross_attn_mask(
            q_token_pad_mask=target_padding_mask, kv_token_pad_mask=context_padding_mask
        )
        pred = self.encoder(q=m, kv=ctx, self_mask=self_mask, cross_mask=cross_mask)

        # out
        pred = self.out_proj(pred)
        return pred
