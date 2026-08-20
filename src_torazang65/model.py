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
        image_size=64,
        use_images=True,
        time_mask_prob=0.15,
        modality_drop_prob=0.25,
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
        # dims (64px -> 4x4 grid), and forward() then collapses that
        # grid with global average pooling, so the projection below is
        # independent of image_size. The image_size argument is kept
        # only so call sites stay unchanged.

        # Diagnostic switch. With use_images=False the CNN is skipped
        # and the image tokens are dropped entirely, so the encoder
        # sees only the 20 wind tokens. Parameters are still built, so
        # checkpoints stay compatible between the two modes.
        self.use_images = use_images

        # Temporal masking augmentation (SpecAugment style). During
        # training each timestep's image token is zeroed independently
        # with this probability, so the model cannot lean on any single
        # frame. Only image tokens are masked -- wind is the stronger
        # signal and masking it costs more than it buys.
        #
        # This targets the real constraint: 9,607 samples drawn with a
        # 6h stride over 6.9 years of data means only a few hundred
        # effectively independent examples.
        #
        # No 1/(1-p) rescaling: token_norm renormalizes every token
        # after the embeddings are added, so a masked token is already
        # on the same scale as an unmasked one.
        self.time_mask_prob = time_mask_prob

        # Modality dropout: with this probability, per sample and per
        # training forward pass, the entire image stream is zeroed.
        # The model is thereby forced to keep a self-sufficient
        # wind-only pathway and cannot lean exclusively on image
        # tokens -- the pathway that carried the observed
        # memorization. Complements the per-timestep masking above.
        self.modality_drop_prob = modality_drop_prob

        # -> one image token per timestep
        #
        # The input is 128-dim because forward() global-average-pools
        # the CNN's 4x4 spatial grid before this projection. The
        # previous flatten+Linear(2048 -> d_model) preserved
        # per-position features, which let the model fingerprint
        # individual frames (each reused by ~20 overlapping windows)
        # and memorize their targets -- that single layer held ~30%
        # of all parameters. The trade-off is real: disk position of
        # coronal features does matter physically (a central coronal
        # hole is Earth-directed, a limb one is not), so if
        # overfitting stops but the image contribution shrinks, a
        # 2x2 pooled middle ground is the next thing to try.
        #
        # No LayerNorm here: normalization happens once per token,
        # after the positional/modality embeddings have been added.
        self.image_projection = nn.Linear(128, d_model)

        # ============================================================
        # 2. Wind embedding
        # ============================================================
        # Each wind value becomes its own full-width token. Wind is
        # NOT fused into the image token of the same timestep: the
        # wind measured at Earth at time t left the Sun 2-5 days
        # earlier (300-800 km/s over 1 AU), so wind_t is causally
        # paired with images 8-20 timesteps back -- and that lag
        # itself depends on the wind speed. Separate token streams
        # let attention discover the speed-dependent alignment
        # instead of hard-wiring a wrong same-timestamp pairing.
        #
        # (B, 20)
        #       ↓
        # (B, 20, 1)
        #       ↓
        # (B, 20, d_model)
        # ============================================================

        self.wind_projection = nn.Linear(1, d_model)

        # ============================================================
        # 3. Positional + modality embeddings, token normalization
        # ============================================================
        #
        # Image and wind tokens taken at the same timestamp share one
        # time embedding; a per-modality embedding tells them apart.
        # (The projection biases could learn the modality offset on
        # their own, but keeping it explicit costs 2*d_model params
        # and spares the content weights the job.)
        #
        # std=0.5 is deliberate. The LayerNorm below forces the token
        # to per-dim std 1.0, so an embedding initialized at the usual
        # 0.02 would contribute ~2% of the token and be normalized
        # away -- the encoder would be nearly order-blind at init.
        # Pushing it to 1.0 instead starts diluting the token content,
        # so 0.5 is the balance point. The modality embeddings reuse
        # the same scale; pos + modality add in quadrature (~0.7 std),
        # still below the content scale.
        # ============================================================

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 20, d_model) * pos_embedding_std
        )

        self.image_modality = nn.Parameter(
            torch.randn(1, 1, d_model) * pos_embedding_std
        )

        self.wind_modality = nn.Parameter(
            torch.randn(1, 1, d_model) * pos_embedding_std
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

        batch_size = wind.size(0)

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

        # (B,20,d_model)
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

            # global average pool over the spatial grid, then move
            # the time dimension:
            #
            # (B, 128, 20) -> (B, 20, 128)
            image_features = image_features.mean(dim=(3, 4))
            image_features = image_features.transpose(1, 2)

            # (B,20,128)
            #       ↓
            # (B,20,d_model)
            image_tokens = self.image_projection(
                image_features
            )

            # (B,20,1) Bernoulli keep-mask, resampled every forward
            # pass. Zeroes the whole image token; the time/modality
            # embeddings added below still mark its slot.
            # Training only -- evaluation sees every frame.
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

            # Modality dropout: (B,1,1) per-sample keep-mask zeroing
            # all 20 image tokens at once. Unlike wind-only mode the
            # token slots stay in the sequence, carrying only the
            # time/modality embeddings. Training only.
            if self.training and self.modality_drop_prob > 0:
                keep = (
                    torch.rand(
                        image_tokens.size(0),
                        1,
                        1,
                        device=image_tokens.device,
                    )
                    >= self.modality_drop_prob
                )
                image_tokens = image_tokens * keep
        else:
            # wind-only diagnostic: skip the CNN and drop the image
            # stream entirely -- the encoder runs on wind tokens only.
            image_tokens = None

        # ============================================================
        # MULTIMODAL SEQUENCE
        # ============================================================
        # Two parallel token streams over the same 20 timestamps:
        #
        # [
        #   image_t0, ..., image_t19,
        #   wind_t0,  ..., wind_t19,
        # ]
        #
        # Same-index image and wind share a time embedding but stay
        # separate tokens: wind_t left the Sun days before image_t
        # was taken, so fusing them would assert a causal alignment
        # that does not exist (see section 2 in __init__). Attention
        # is free to match wind to the earlier images instead.
        #
        # shape:
        #
        # (B, 40, d_model) -- (B, 20, d_model) in wind-only mode
        # ============================================================

        wind_tokens = (
            wind_tokens + self.pos_embedding + self.wind_modality
        )

        if image_tokens is not None:
            image_tokens = (
                image_tokens
                + self.pos_embedding
                + self.image_modality
            )
            multimodal_tokens = torch.cat(
                [image_tokens, wind_tokens],
                dim=1,
            )
        else:
            multimodal_tokens = wind_tokens

        multimodal_tokens = self.token_norm(multimodal_tokens)

        # ============================================================
        # ENCODER
        # ============================================================
        # Every token can attend to every other token, across both
        # modalities and all timesteps.
        # memory:
        # (B,40,d_model)
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
