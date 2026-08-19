import torch
from torch import nn


class Inception3D(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super().__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
        )

        self.branch_3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                branch_channels,
                branch_channels,
                (1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.ReLU(inplace=True),
        )

        self.branch_5 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                branch_channels,
                branch_channels,
                (1, 5, 5),
                padding=(0, 2, 2),
            ),
            nn.ReLU(inplace=True),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(
                (1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
            ),
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat(
            [
                self.branch_1(x),
                self.branch_3(x),
                self.branch_5(x),
                self.branch_pool(x),
            ],
            dim=1,
        )


class SolarWindBaseline(nn.Module):
    def __init__(
        self,
        d_model=256,
        wind_dim=64,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pos_embedding_std=0.5,
    ):
        super().__init__()

        # ============================================================
        # 1. Image feature extractor
        # ============================================================
        # input:
        #   images: (B, 20, 2, 64, 64)
        #
        # Conv3D input:
        #   (B, 2, 20, 64, 64)
        #
        # Temporal dimension 20 is NOT reduced.
        # ============================================================

        self.stem = nn.Sequential(
            nn.Conv3d(
                2,
                32,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        blocks = []
        in_channels = 32

        for _ in range(3):
            blocks.extend(
                [
                    Inception3D(in_channels, 32),

                    nn.MaxPool3d(
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    ),
                ]
            )

            # 4 branches * 32 channels
            in_channels = 128

        self.image_encoder = nn.Sequential(*blocks)

        # Existing CNN output:
        #
        # (B, 128, 20, 4, 4)
        #
        # -> timestep-wise:
        #
        # (B, 20, 128 * 4 * 4)
        #
        # -> image slice of the fused token
        #
        # No LayerNorm here: normalization happens once on the fused
        # token, after the positional embedding has been added.
        self.image_projection = nn.Linear(
            128 * 4 * 4,
            d_model - wind_dim,
        )

        # ============================================================
        # 2. Wind embedding
        # ============================================================
        #   each wind value gets its own slice of the fused token,
        #   so the scalar keeps a dedicated subspace instead of
        #   competing with the image features for all d_model dims.
        #
        # (B, 20)
        #       ↓
        # (B, 20, 1)
        #       ↓
        # (B, 20, wind_dim)
        # ============================================================

        self.wind_projection = nn.Linear(1, wind_dim)

        # ============================================================
        # 3. Positional embedding + fused-token normalization
        # ============================================================
        #
        # Image and wind are concatenated per timestep, so there is a
        # single token type and no modality embedding is needed.
        #
        # std=0.5 is deliberate. The LayerNorm below forces the token
        # to per-dim std 1.0, so an embedding initialized at the usual
        # 0.02 would contribute ~2% of the token and be normalized
        # away -- the encoder would be nearly order-blind at init.
        # Pushing it to 1.0 instead starts diluting the token content,
        # so 0.5 is the balance point.
        # ============================================================

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 20, d_model) * pos_embedding_std
        )

        self.token_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 4. Multimodal Transformer Encoder
        # ============================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ============================================================
        # 5. 12 future query tokens
        # ============================================================
        #
        # query 0  -> +6h
        # query 1  -> +12h
        # ...
        # query 11 -> +72h
        #
        # These are learned parameters.
        #
        # 0.02 is fine here, unlike the positional embedding above:
        # these are not added to anything, and the decoder is
        # norm_first, so LayerNorm -- which is scale invariant --
        # is the first thing applied to them.
        # ============================================================

        self.future_queries = nn.Parameter(
            torch.randn(1, 12, d_model) * 0.02
        )

        # ============================================================
        # 6. Transformer Decoder
        # ============================================================

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ============================================================
        # 7. Prediction head
        # ============================================================
        #
        # each decoder query:
        #
        # (d_model) -> one solar wind value
        # ============================================================

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, images, wind):

        batch_size = images.size(0)

        # ============================================================
        # IMAGE TOKENS
        # ============================================================

        # dataset:
        #
        # images:
        # (B, T=20, C=2, H=64, W=64)
        #
        # Conv3D wants:
        #
        # (B, C=2, T=20, H=64, W=64)
        image_features = images.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        image_features = self.stem(image_features)
        image_features = self.image_encoder(image_features)

        # expected:
        #
        # (B, 128, 20, 4, 4)

        # move time dimension:
        #
        # (B, 20, 128, 4, 4)
        image_features = image_features.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        # flatten spatial/channel dimensions
        #
        # (B, 20, 2048)
        image_features = image_features.flatten(2)

        # (B,20,2048)
        #       ↓
        # (B,20,d_model - wind_dim)
        image_tokens = self.image_projection(
            image_features
        )

        # ============================================================
        # WIND TOKENS
        # ============================================================

        # (B,20)
        #    ↓
        # (B,20,1)
        wind = wind.unsqueeze(-1)

        # (B,20,wind_dim)
        wind_tokens = self.wind_projection(wind)

        # ============================================================
        # MULTIMODAL SEQUENCE
        # ============================================================
        #
        # Image and wind observed at the same timestamp are fused into
        # a single token, so the encoder never has to learn to pair
        # them up:
        #
        # [
        #   [image_t0 | wind_t0],
        #   [image_t1 | wind_t1],
        #   ...
        # ]
        #
        # shape:
        #
        # (B, 20, d_model)
        # ============================================================

        multimodal_tokens = torch.cat(
            [image_tokens, wind_tokens],
            dim=-1,
        )

        multimodal_tokens = self.token_norm(
            multimodal_tokens + self.pos_embedding
        )

        # ============================================================
        # ENCODER
        # ============================================================
        # Every timestep can attend to every other observed timestep.
        # memory:
        # (B,20,d_model)
        # ============================================================

        memory = self.transformer_encoder(
            multimodal_tokens
        )

        # ============================================================
        # FUTURE QUERIES
        # ============================================================
        #
        # (1,12,d_model)
        #       ↓
        # (B,12,d_model)
        # ============================================================

        queries = self.future_queries.expand(
            batch_size,
            -1,
            -1,
        )

        # ============================================================
        # DECODER
        # ============================================================
        # query +6h  -> attends encoder memory
        # query +12h -> attends encoder memory
        # ...
        # query +72h -> attends encoder memory
        #
        # All 12 are predicted simultaneously.
        # NO autoregressive loop.
        # NO teacher forcing.
        # ============================================================

        decoded = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )

        # decoded:
        #
        # (B,12,d_model)

        # ============================================================
        # OUTPUT
        # ============================================================

        prediction = self.output_head(decoded)

        # (B,12,1)
        #     ↓
        # (B,12)
        prediction = prediction.squeeze(-1)

        return prediction
