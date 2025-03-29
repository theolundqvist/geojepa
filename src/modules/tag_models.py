from dataclasses import dataclass

import torch
from einops import repeat
from torch import nn

from src.data.components.tiles import TileBatch
from src.modules.encoder import Encoder
from src.modules.masks import RandomMask, apply_mask, MaskingStrategy
from src.modules.mlp import MLP
from src.modules.tag_encoder import TagEncoder, TagIndexer
from src.modules.tokenizer import Modality


@dataclass
class TagformerIntermediates:
    cls: torch.Tensor
    ctx_mask: torch.Tensor
    tgt_mask: torch.Tensor
    features: torch.Tensor
    tag_embeddings: torch.Tensor
    all_feat_pad_mask: torch.Tensor
    position_embs: torch.Tensor


class Tagformer(nn.Module):
    def __init__(
        self,
        embedding_file: str,
        use_padding_mask=True,
        use_proj_activation=True,
        use_positional_encoding=False,
        use_semantic_encoding=True,
        return_intermediaries=False,
        depth=5,
        h_dim=128,
        out_dim: int = 1,
    ):
        super().__init__()
        self.use_padding_mask = use_padding_mask
        self.use_proj_activation = use_proj_activation
        self.use_positional_encoding = use_positional_encoding
        self.use_semantic_encoding = use_semantic_encoding
        self.return_intermediaries = return_intermediaries

        self.tag_encoder = TagEncoder(embedding_file).requires_grad_(False).eval()
        self.num_tags = self.tag_encoder.nbr_unique_tags
        assert self.num_tags >= 1024, (
            f"Expected at least 1024 tags, got {self.num_tags}"
        )

        if self.use_semantic_encoding:
            self.semantic_proj = MLP(1024, h_dim * 4, h_dim, bias=True)
        else:
            self.multi_hot_norm = nn.LayerNorm(self.num_tags)
            self.multi_hot_proj = nn.Linear(self.num_tags, h_dim, bias=True)

        self.proj_norm = nn.LayerNorm(h_dim)

        if self.use_positional_encoding:
            self.positional_encoding = MLP(8, h_dim * 4, h_dim, 0.1)

        self.model = Encoder(depth=depth, token_dim=h_dim, num_heads=8, dropout=0.1)
        self.cls = nn.Parameter(torch.zeros(1, 1, h_dim))
        if out_dim != h_dim:
            self.out_proj = nn.Linear(h_dim, out_dim)
        self.logs = {}

    def forward(
        self, batch: TileBatch, token_masking_strategy: MaskingStrategy | None = None
    ):
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"

        B, T, _ = batch.tags.shape
        if self.use_semantic_encoding:
            with torch.no_grad():
                encoded_tags = self.tag_encoder(batch.tags)
            self.logs["tag_transformer/tag_encoder_std"] = encoded_tags.std(dim=0).sum()
            x = self.semantic_proj(encoded_tags)
        else:
            multi_hot = torch.zeros((B, T, self.num_tags), device=batch.device)
            multi_hot.scatter_(2, batch.tags, 1)
            multi_hot.clamp_(0, 1)  # should only have one of each feature anyway
            x = self.multi_hot_norm(multi_hot)
            x = self.multi_hot_proj(x)

        if self.use_proj_activation:
            x = nn.functional.gelu(x)
            x = self.proj_norm(x)

        self.logs["tag_transformer/proj_std"] = x.std(dim=0).sum()
        if self.use_positional_encoding:
            pos = self.positional_encoding(batch.min_boxes.view(B, T, 8))
            x += pos

        padding_mask = None
        if self.use_padding_mask:
            feat_idx = torch.arange(T, device=x.device)
            feat_idx = repeat(feat_idx, "f -> b f", b=B)  # [B, F]
            padding_mask = feat_idx >= batch.feature_counts.unsqueeze(1)  # [B, F]
            all_feat_pad_mask = padding_mask.clone()

        # for use in masked autoencoder
        use_token_masking = token_masking_strategy is not None
        if use_token_masking:
            assert self.use_positional_encoding and self.use_padding_mask, (
                "Token masking requires positional encoding and padding mask"
            )
            token_masking_strategy.num_targets = 1
            ctx_mask, tgt_masks = token_masking_strategy(
                x, batch.min_boxes.view(B, T, 8), ~padding_mask
            )
            tgt_mask = tgt_masks[0]
            x = apply_mask(x, ctx_mask)
            padding_mask = apply_mask(padding_mask, ctx_mask)

        cls = self.cls.expand(batch.size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.use_padding_mask:
            cls_mask = repeat(
                torch.tensor([0], device=x.device), "f -> b f", b=B
            ).bool()
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.model(x, padding_mask=padding_mask)

        self.logs["tag_transformer/model_std"] = x.std(dim=0).sum()
        if self.return_intermediaries:
            # if has out_proj attribute
            if hasattr(self, "out_proj"):
                x = self.out_proj(x)
            return TagformerIntermediates(
                cls=x[:, 0],
                features=x[:, 1:],
                position_embs=pos if self.use_positional_encoding else None,
                tag_embeddings=encoded_tags,
                all_feat_pad_mask=all_feat_pad_mask if self.use_padding_mask else None,
                # only forin-line masking
                ctx_mask=ctx_mask if use_token_masking else None,
                tgt_mask=tgt_mask if use_token_masking else None,
            )
            # return x[:, 0], x[:, 1:], encoded_tags, padding_mask[:, 1:] # cls, features, frozen_tag_embeddings
        x = x[:, 0]  # cls tokens
        if self.out_proj:
            x = self.out_proj(x)
            self.logs["tag_transformer/out_proj_std"] = x.std(dim=0).sum()
        return x

    def get_metrics(self):
        return self.logs


# class TagSelector(nn.Module):
#     def forward(self, batch: TileBatch):
#         assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
#         return batch.tags


class EntityTagAvg(nn.Module):
    def __init__(self, embedding_file: str):
        super().__init__()
        self.tag_encoder = TagEncoder(embedding_file).requires_grad_(False).eval()

    def forward(self, batch: TileBatch):
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        feats = self.tag_encoder(batch.tags)
        B, T, C = feats.shape
        cls = torch.cat(
            [torch.max(feats, dim=1).values, torch.mean(feats, dim=1)], dim=1
        )

        mods = torch.ones((B, T), device=feats.device) * Modality.PAD
        for b in range(B):
            mods[b, : batch.feature_counts[b]] = Modality.OSM

        return cls, feats, mods


class TagCountEncoder(nn.Module):
    def __init__(self, embedding_file: str, h_dim=256, out_dim: int = 1):
        super().__init__()
        self.h_dim = h_dim
        self.num_tags = TagIndexer(embedding_file).nbr_unique_tags
        self.brains = nn.Sequential(
            nn.LayerNorm(self.num_tags),
            nn.Linear(self.num_tags, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
            MLP(h_dim, h_dim * 2, h_dim, drop=0.1),
            nn.Linear(h_dim, out_dim),
        )
        self.logs = {}

    def forward(self, batch: TileBatch):
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"

        B, F, T = batch.tags.shape
        multi_hot = torch.zeros((B, F, self.num_tags), device=batch.device)
        multi_hot.scatter_(2, batch.tags, 1)
        multi_hot.clamp_(0, 1)  # should only have one of each feature anyway

        tile_tag_count = multi_hot.sum(dim=1)  # [B, 12573]
        return self.brains(tile_tag_count)

    def get_metrics(self):
        return self.logs


class TagCountAE(nn.Module):
    def __init__(self, embedding_file: str, h_dim=256):
        super().__init__()
        self.h_dim = h_dim
        self.num_tags = TagIndexer(embedding_file).nbr_unique_tags
        self.tag_norm = nn.LayerNorm(self.num_tags)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_tags + 1, h_dim * 4),
            nn.GELU(),
            nn.LayerNorm(h_dim * 4),
            MLP(h_dim * 4, h_dim * 4, h_dim, drop=0.1),
        )
        self.decoder = nn.Sequential(
            MLP(h_dim, h_dim * 4, self.num_tags + 1, drop=0.1),
        )
        self.logs = {}

    def forward(self, batch: TileBatch, decode=True):
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"

        B, F, T = batch.tags.shape
        multi_hot = torch.zeros((B, F, self.num_tags), device=batch.device)
        multi_hot.scatter_(2, batch.tags, 1)
        multi_hot.clamp_(0, 1)  # should only have one of each feature anyway

        tile_tag_count = multi_hot.sum(dim=1)  # [B, 12573]
        y = self.tag_norm(tile_tag_count)
        y = torch.cat([batch.feature_counts.unsqueeze(1) / 100, y], dim=1)
        enc = self.encoder(y)
        if decode:
            dec = self.decoder(enc)
            return y, enc, dec
        else:
            return enc

    def get_metrics(self):
        return self.logs


TagAutoEncoderModule = TagCountAE


class TagformerAE(nn.Module):
    def __init__(self, embedding_file: str, h_dim=256):
        super().__init__()

        self.encoder = Tagformer(
            embedding_file,
            use_positional_encoding=True,
            use_semantic_encoding=True,
            return_intermediaries=True,
            depth=10,
            h_dim=h_dim,
            out_dim=h_dim,
        )

        self.decoder = Encoder(
            depth=10, token_dim=h_dim, num_heads=8, dropout=0.1, reg_tokens=2
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, h_dim))

        self.hdim_to_tag_projection = nn.Linear(h_dim, 1024)

    def compile_supported(self):
        self.encoder.model = torch.compile(self.encoder.model, fullgraph=True)
        self.decoder = torch.compile(self.decoder, fullgraph=True)
        self.hdim_to_tag_projection = torch.compile(
            self.hdim_to_tag_projection, fullgraph=True
        )
        self.encoder.semantic_proj = torch.compile(
            self.encoder.semantic_proj, fullgraph=True
        )
        self.encoder.positional_encoding = torch.compile(
            self.encoder.positional_encoding, fullgraph=True
        )

    def forward(self, batch: TileBatch, decode=True):
        res = self.encoder(batch)

        # teacher_cls = self.encoder(batch, token_masking_strategy=self.masking_strategy).cls

        if not decode:
            return res.cls
            # pool = torch.cat([
            #     torch.max(res.features, dim=1).values,
            #     torch.mean(res.features, dim=1)
            # ], dim=1)
            # return pool

        cls = res.cls
        feats = res.features
        tag_embs = res.tag_embeddings
        pos = res.position_embs
        padding_mask = res.all_feat_pad_mask

        tgt_tokens = pos + self.mask_token.expand_as(pos)

        tokens = torch.cat((cls.unsqueeze(1), tgt_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.tensor([False], device=padding_mask.device).expand(batch.size, 1),
                padding_mask,
            ),
            dim=1,
        )

        dec = self.decoder(tokens, padding_mask=padding_mask)
        dec_cls = dec[:, 0]
        dec_target = dec[:, 1:]
        dec_target = self.hdim_to_tag_projection(dec_target)

        return tag_embs, cls, dec_target


class TagformerLMAE(nn.Module):
    def __init__(self, embedding_file: str, h_dim=256):
        super().__init__()

        self.encoder = Tagformer(
            embedding_file,
            use_positional_encoding=True,
            use_semantic_encoding=True,
            return_intermediaries=True,
            depth=10,
            h_dim=h_dim,
            out_dim=h_dim,
        )

        self.decoder = Encoder(
            depth=10, token_dim=h_dim, num_heads=8, dropout=0.1, reg_tokens=2
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, h_dim))

        self.masking_strat = RandomMask(target_size=0.7, num_targets=1, min_context=0.1)

        self.hdim_to_tag_projection = nn.Linear(h_dim, 1024)

    def compile_supported(self):
        self.encoder.model = torch.compile(self.encoder.model, fullgraph=True)
        self.decoder = torch.compile(self.decoder, fullgraph=True)
        self.hdim_to_tag_projection = torch.compile(
            self.hdim_to_tag_projection, fullgraph=True
        )
        self.encoder.semantic_proj = torch.compile(
            self.encoder.semantic_proj, fullgraph=True
        )
        self.encoder.positional_encoding = torch.compile(
            self.encoder.positional_encoding, fullgraph=True
        )

    def forward(self, batch: TileBatch, decode=True):
        res = self.encoder(batch)

        # teacher_cls = self.encoder(batch, token_masking_strategy=self.masking_strategy).cls

        if not decode:
            return res.cls, res.features
            # pool = torch.cat([
            #     torch.max(res.features, dim=1).values,
            #     torch.mean(res.features, dim=1)
            # ], dim=1)
            # return pool

        cls = res.cls
        feats = res.features
        tag_embs = res.tag_embeddings
        pos = res.position_embs
        padding_mask = res.all_feat_pad_mask

        ctx_mask, tgt_mask = self.masking_strat(feats, pos, ~padding_mask)

        # Create context tokens
        ctx_tokens = apply_mask(feats, ctx_mask)
        ctx_pos = apply_mask(pos, ctx_mask)
        ctx_tokens = ctx_tokens + ctx_pos

        # Create target tokens (masked decoder input)
        tgt_pos = apply_mask(pos, tgt_mask)
        tgt_tokens = tgt_pos + self.mask_token.expand_as(tgt_pos)

        tokens = torch.cat((cls.unsqueeze(1), tgt_tokens, ctx_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.tensor([False], device=padding_mask.device).expand(batch.size, 1),
                apply_mask(padding_mask, tgt_mask),
                apply_mask(padding_mask, ctx_mask),
            ),
            dim=1,
        )

        truths = apply_mask(tag_embs, tgt_mask)

        dec = self.decoder(tokens, padding_mask=padding_mask)
        dec_cls = dec[:, 0]
        dec_target = dec[:, 1 : (tgt_tokens.size(1) + 1)]
        dec_target = self.hdim_to_tag_projection(dec_target)

        return truths, feats, dec_target


# -----------------------
# Does not work.
# -----------------------
class TagformerMAE(nn.Module):
    def __init__(self, embedding_file: str, h_dim=128, use_semantic_encoding=True):
        super().__init__()

        self.encoder = Tagformer(
            embedding_file,
            use_positional_encoding=True,
            use_semantic_encoding=use_semantic_encoding,
            return_intermediaries=True,
            depth=10,
            h_dim=h_dim,
            out_dim=h_dim,
        )

        self.decoder = Encoder(
            depth=6, token_dim=h_dim, num_heads=8, dropout=0.1, reg_tokens=2
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        self.masking_strategy = RandomMask(
            target_size=0.4,
            num_targets=1,
        )
        self.norm = nn.LayerNorm(h_dim)

    def compile_supported(self):
        pass
        # self.encoder = torch.compile(self.encoder)
        # self.decoder = torch.compile(self.decoder)
        # self.norm = torch.compile(self.norm)

    def forward(self, batch: TileBatch, decode=True):
        res = self.encoder(batch, token_masking_strategy=self.masking_strategy)

        # teacher_cls = self.encoder(batch, token_masking_strategy=self.masking_strategy).cls

        if not decode:
            pool = torch.cat(
                [
                    torch.max(res.features, dim=1).values,
                    torch.mean(res.features, dim=1),
                ],
                dim=1,
            )
            return pool

        cls = res.cls
        truth = res.tag_embeddings
        pos = res.position_embs

        padding_mask = res.all_feat_pad_mask
        ctx_mask = res.ctx_mask
        tgt_mask = res.tgt_mask

        # masked features
        ctx_tokens = res.features
        tgt_positions = apply_mask(pos, tgt_mask)
        tgt_tokens = tgt_positions + self.mask_token.expand_as(tgt_positions)

        tokens = torch.cat((tgt_tokens, ctx_tokens), dim=1)
        padding_mask = torch.cat(
            (apply_mask(padding_mask, tgt_mask), apply_mask(padding_mask, ctx_mask)),
            dim=1,
        )
        x = self.decoder(tokens, padding_mask=padding_mask)

        truth_target = apply_mask(truth, tgt_mask)
        decoded_target = x[:, : tgt_tokens.size(1)]

        # ctx_mask, tgt_masks = self.masking_strategy(feats, pos, ~mask)
        # tgt_mask = tgt_masks[0] # only use one target mask
        #
        # ctx_tokens = apply_mask(feats, ctx_mask)
        # target_tokens = apply_mask(pos, tgt_mask)
        #
        # ctx = torch.cat((cls, ctx_tokens, target_tokens), dim=1)
        # mods = torch.cat((res.cls_mask, ctx_mask, tgt_mask), dim=1)
        # target_mask = torch.cat((torch.zero_like(res.cls_mask), torch.zero_like(ctx_mask), tgt_mask), dim=1)
        # dec = self.decoder(ctx, padding_mask=(mods == 0))
        # truth_target = apply_mask(truth, tgt_mask)

        # return x, enc, dec
        return truth_target, ctx_tokens, decoded_target
