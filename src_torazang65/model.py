import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size, padding=0):
    """Conv3d -> BatchNorm3d -> ReLU.

    The CNN previously had no normalization at all, which is the piece
    it was actually missing (residual connections would do little at a
    7-layer depth). BatchNorm adds only ~1.2k parameters in total.
    conv bias is dropped because BatchNorm's shift subsumes it.
    """
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class Inception3D(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super().__init__()

        self.branch_1 = conv_block(in_channels, branch_channels, 1)

        self.branch_3 = nn.Sequential(
            conv_block(in_channels, branch_channels, 1),
            conv_block(
                branch_channels,
                branch_channels,
                (1, 3, 3),
                padding=(0, 1, 1),
            ),
        )

        self.branch_5 = nn.Sequential(
            conv_block(in_channels, branch_channels, 1),
            conv_block(
                branch_channels,
                branch_channels,
                (1, 5, 5),
                padding=(0, 2, 2),
            ),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(
                (1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
            ),
            conv_block(in_channels, branch_channels, 1),
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
        image_size=64,
        use_images=True,
        time_mask_prob=0.15,
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
            conv_block(
                2,
                32,
                (1, 5, 5),
                padding=(0, 2, 2),
            ),

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

        # The stem pool and the 3 block pools each halve the spatial
        # dims, so the CNN output is image_size // 16 on a side:
        #
        #    64px -> (B, 128, 20, 4, 4) -> 2048 per timestep
        #   128px -> (B, 128, 20, 8, 8) -> 8192 per timestep
        #
        # Derived rather than hard-coded so changing IMAGE_SIZE in
        # config.py does not silently break the projection.
        if image_size % 16 != 0:
            raise ValueError(
                f"image_size must be divisible by 16, got {image_size}"
            )

        spatial = image_size // 16

        # Diagnostic switch. With use_images=False the CNN is skipped
        # entirely and the image slice of every token is zeroed, so the
        # model sees wind history only. Parameters are still built, so
        # checkpoints stay compatible between the two modes.
        self.use_images = use_images

        # Temporal masking augmentation (SpecAugment style). During
        # training each timestep's image slice is zeroed independently
        # with this probability, so the model cannot lean on any single
        # frame. Only the image slice is masked -- wind is the stronger
        # signal and masking it costs more than it buys.
        #
        # This targets the real constraint: 9,607 samples drawn with a
        # 6h stride over 6.9 years of data means only a few hundred
        # effectively independent examples.
        #
        # No 1/(1-p) rescaling: token_norm renormalizes every token
        # after the positional embedding is added, so a masked token is
        # already on the same scale as an unmasked one.
        self.time_mask_prob = time_mask_prob

        # -> image slice of the fused token
        #
        # No LayerNorm here: normalization happens once on the fused
        # token, after the positional embedding has been added.
        self.image_projection = nn.Linear(
            128 * spatial * spatial,
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

        # Zero-init the last layer so the head starts at exactly zero.
        # Combined with the persistence residual in forward(), the
        # model starts as pure persistence and learns only the
        # correction on top of it. Without this the initial prediction
        # is wind[-1] plus a random offset of roughly +-200 km/s.
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, images, wind):

        batch_size = images.size(0)

        # Last observed wind value, kept for the persistence residual
        # at the very end of forward.
        #
        # (B,20) -> (B,1)
        last_wind = wind[:, -1:]

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
        # IMAGE TOKENS
        # ============================================================

        if self.use_images:
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

            # (B,20,1) Bernoulli keep-mask, resampled every forward
            # pass. Training only -- evaluation sees every frame.
            if self.training and self.time_mask_prob > 0:
                keep = (
                    torch.rand(
                        image_tokens.size(0),
                        image_tokens.size(1),
                        1,
                        device=image_tokens.device,
                    )
                    >= self.time_mask_prob
                )
                image_tokens = image_tokens * keep
        else:
            # wind-only diagnostic: skip the CNN and blank the image
            # slice. dtype follows wind_tokens so this stays correct
            # under autocast.
            image_tokens = wind_tokens.new_zeros(
                batch_size,
                wind_tokens.size(1),
                self.image_projection.out_features,
            )

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

        # ============================================================
        # PERSISTENCE RESIDUAL
        # ============================================================
        # The head predicts the *change* from the last observed wind
        # value rather than the absolute level. wind and target share
        # the same /1000 scaling in dataset.py, so the units match.
        #
        # (B,12) + (B,1) -> (B,12)
        # ============================================================

        return prediction + last_wind
