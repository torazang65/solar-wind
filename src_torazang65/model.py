import torch
from torch import nn
from torch.nn import functional as F


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
        correction_drop_prob=0.0,
        use_correction=True,
        use_surge_head=False,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pos_embedding_std=0.5,
        image_in_channels=2,
        add_diff_channels=False,
    ):
        super().__init__()

        # v4: running-difference 채널 (I_t - I_{t-1})을 forward에서 GPU로
        # 계산해 원본 채널에 concat한다. coronal dimming(음수)/플레어
        # 증광(양수) 같은 CME의 on-disk 신호가 사는 곳인데, 아래 conv는
        # 시간 커널이 전부 1이라 이 정보를 스스로 만들 수 없다. dataset
        # (CPU)이 아니라 여기서 만드는 이유: DataLoader의 in-flight 배치
        # 메모리를 2배로 만들어 원격 pod에서 OOM이 났기 때문. GPU에서는
        # 배치당 일시 텐서 하나 비용이다. True면 image_in_channels는
        # 원본 채널 수의 2배여야 한다 (2 -> 4).
        self.add_diff_channels = add_diff_channels

        # ============================================================
        # 1. Image feature extractor
        # ============================================================
        # input:
        #   images: (B, 20, C_in, 64, 64)
        #
        # C_in=4 (v4): 2 EUV band + 2 running-difference 채널. 아래의
        # 모든 conv는 시간 커널이 1이라 프레임 간 픽셀 차분을 이
        # 네트워크가 스스로 계산할 수 없다 -- CME의 on-disk 신호
        # (dimming/플레어 증광)는 dataset.py가 Δ 채널로 공급한다.
        # image_in_channels=2면 이전(원본 채널만) 구조와 동일.
        #
        # Conv3D input:
        #   (B, C_in, 20, 64, 64)
        #
        # Temporal dimension 20 is NOT reduced.
        # ============================================================

        self.stem = nn.Sequential(
            conv_block(
                image_in_channels,
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

        # v7 분석용 스위치: True면 eval에서도 modality drop과 동일하게
        # 이미지 스트림(토큰 + readout cell feature)을 통째로 0으로
        # 만든다. analysis.py의 wind-only ablation이 학습 분포 안의
        # 조건(그냥 modality drop)과 정확히 일치하게 하는 장치 --
        # 구 speed_ablation.py의 image_projection hook은 v7에서
        # projection을 우회하는 readout 경로를 못 껐다.
        self.force_image_drop = False

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

        # v5c: correction dropout. 학습 중 샘플 단위로 출력 조립의
        # correction 항을 0으로 만들어 base + 전파 branch가 자립하도록
        # 강제한다. v5a의 병목이 correction(full transformer) 경로의
        # 과적합(train 48 vs val 74, epoch-4 best -- 성숙한 전파
        # branch가 체크포인트에 못 담김)이었고, 이미지 modality_drop과
        # 같은 논리를 출력단에 적용한 것. Training only.
        self.correction_drop_prob = correction_drop_prob

        # v6a: correction 경로 스위치. v5b branch decomposition에서
        # base+prop(67.3) < full(69.2): corr own-gain은 quiet +4.0,
        # surge -3.3(유의)이고 full은 surge에서 base+prop 대비 -7~-9를
        # 잃는다. decoder attention COM은 horizon/속도군 무관 ~55h
        # 고정(정렬된 읽기를 안 함)이며, correction_drop 0.3으로도
        # epoch-6 조기 best를 못 막았다 (val hindcast는 epoch 13+까지
        # 개선 -- 전파 branch 성숙이 체크포인트에 못 담김).
        #
        # False면 transformer encoder/decoder/output head와 wind
        # 토큰화를 아예 만들지 않는다. 출력은
        #   base + alpha*(v_img - base) = (1-a)*B + a*S
        # 의 2-expert convex mixture만 남고, 과거 wind는 base와 gate
        # 요약으로만, EUV는 propagation readout으로만 미래에
        # 기여한다 (EUV -> 미래의 유일 경로가 arrival-conditioned).
        self.use_correction = use_correction

        # -> one image token per timestep
        #
        # 2x4 (lat x lon) pooled middle ground between two extremes:
        #
        #   flatten + Linear(2048 -> d_model): kept per-position
        #     features but that single layer held ~30% of all
        #     parameters and let the model fingerprint individual
        #     frames (each reused by ~20 overlapping windows).
        #   global average pool + Linear(128 -> d_model): stopped the
        #     memorization but destroyed disk position entirely --
        #     and the speed_ablation analysis showed the image
        #     pathway's whole value is detecting incoming streams,
        #     for which the source's distance from the central
        #     meridian is the decisive variable (central =
        #     Earth-directed, limb = misses us).
        #
        # The pooling is deliberately anisotropic. A symmetric 2x2
        # has no center cell: a central coronal hole smears across
        # all four quadrants and central-vs-limb contrast is weak --
        # yet that is the axis that matters. 2x4 keeps the CNN's
        # full longitude resolution (4 columns = E-limb, E-center,
        # W-center, W-limb; solar rotation carries features east to
        # west, so longitude also encodes time-to-Earth-facing) and
        # folds only latitude to 2 rows (hemisphere info survives).
        #
        # Projection is 1024 -> d_model (~131k weights): half the
        # old flatten layer, and the time_mask / modality_drop
        # regularizers guard the reopened capacity. AdaptiveAvgPool
        # keeps the projection independent of image_size.
        #
        # No LayerNorm here: normalization happens once per token,
        # after the positional/modality embeddings have been added.
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 2, 4))
        self.image_projection = nn.Linear(128 * 2 * 4, d_model)

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

        # v6a: wind 토큰은 encoder 입력에만 쓰이므로 correction 경로와
        # 함께 생긴다 (base/gate는 wind 원시 시퀀스를 직접 읽는다).
        if use_correction:
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

        # v6a: 시간/모달리티 embedding과 token_norm도 encoder 전용이다.
        # propagation readout(6.5)과 fusion 요약은 embedding이 더해지기
        # 전의 content 토큰에서 계산하므로 영향이 없다.
        if use_correction:
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

        if use_correction:
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

        if use_correction:
            self.future_queries = nn.Parameter(
                torch.randn(1, 12, d_model) * 0.02
            )

        # ============================================================
        # 6. Transformer Decoder
        # ============================================================

        if use_correction:
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
        # 6.5 Propagation readout -- v5a: learned-ESWF, v7: source-map
        # ============================================================
        # v2/v3의 dynamic temporal prior(cross-attention logit bias)를
        # 대체한다. 실패 원인 진단: (1) prior가 출력 경로 밖의
        # side-channel이라 gate zero-init에서 tau의 gradient가 gate에
        # 비례해 0이었고(순환 사멸), (2) wind persistence 지름길이
        # 있는 forecast loss만으로는 이미지 시간 정렬을 배울 유인이
        # 없었다 (attention COM이 균일분포 평균(~57h)에 고정, slow-fast
        # 그룹 차이 없음).
        #
        # 교체 구조: 토큰별 물리 스칼라 3개를 뽑아 결정론적 전파식으로
        # 과거+미래 시간축의 wind 기여를 직접 조립한다.
        #
        #   s_t   in [250, 900] km/s   source 속도 (/1000 스케일)
        #   p_t   >= 0                 source 존재/지구 방향성
        #   tau_t = D_eff/s_t + 24h*tanh(.)    통과시간
        #   a_t   = tau_t - age_t      도착 시각 (음수 = 관측된 과거)
        #
        #   v_img(u) = (sum_t p_t K(u-a_t) s_t + lam*c)
        #              / (sum_t p_t K(u-a_t) + lam)
        #
        # 설계 근거:
        # - ballistic backbone + 유계 잔차: 자유 tau는 "예측값을 곡선의
        #   아무 시점에나 배치"하는 퇴화해를 허용한다. D/s 결합은 값이
        #   시각을 결정하게 묶고(값-시각 잠금), tanh 잔차 ±24h가
        #   SIR/connectivity 편차를 표현한다.
        # - 정규화 가중평균: 출력이 s_t들의 convex hull 안에 갇혀
        #   p로 진폭을 위조할 수 없다. 소스가 없으면 climatology c로
        #   후퇴한다 (quiet에서 persistence에 지던 문제의 구조적 방어).
        # - u는 hindcast 격자 [-72..0]h + forecast 격자 [+6..+72]h를
        #   한 번에 쓴다. 같은 layer가 관측된 과거 wind를 재구성하는
        #   보조 손실(train.py)로 이미지 전용 supervision을 받고,
        #   그대로 미래로 연장된다. 커버리지는 속도 의존적(-|s| 시점은
        #   tau <= 114-|s|인 소스만 도달)이라 깊은 과거는 fast stream만
        #   커버하지만, 그게 정확히 이미지 gain이 0이던 레짐이다.
        #   커버 불가 시점은 kernel 질량이 0이라 c로 수렴하고 전파
        #   head에는 gradient가 없다. 커버리지를 학습량(p)으로
        #   마스킹하면 p를 낮춰 loss를 회피할 유인이 생기므로 일부러
        #   마스킹하지 않는다.
        # - 초기화: 모든 head가 forward 경로에 있어 gradient가 항상
        #   흐르므로(v2 gate와 달리 사멸 지점이 없다) 시작점만 현재
        #   구조와 맞춘다. speed bias -0.96 -> s=0.43=climatology,
        #   나머지 zero-init -> 순수 ballistic, 균일 p.
        #
        # v7 (source-map readout): 소스 단위를 프레임(20)에서
        # 프레임 x 디스크 cell(20 x 2lat x 4lon = 160)로 분해하고,
        # 자전 기하를 도착 시각 공식에 박는다.
        #
        #   arrival_{t,c} = -lon_{t,c}/omega + D_eff/s_{t,c}
        #                   + 24h*tanh(.) - age_t
        #
        #   -lon/omega : cell 경도(음수=동쪽)가 CM(지구 방향)까지
        #                자전해 오는 시간. 표준 ballistic backmapping
        #                (출발 시점 소스는 CM 근방)의 순방향이다.
        #   D_eff/s    : CM 출발 후 radial 통과시간 (v5a와 동일).
        #
        # 근거 (v6b 진단): 프레임당 소스 1개는 "이 프레임 어딘가"까지만
        # 표현해 시간 정렬이 속도 무관 고정 템플릿(COM 속도군 분화 7h
        # vs 이론 30h)으로 퇴화했다. 경도는 도착 시각의 지배 변수다
        # (자전 0.55도/h -> 인접 column 45도 = ~82h, horizon 전체보다
        # 크다). 같은 프레임의 east-limb CH와 CM CH가 다른 arrival을
        # 갖게 되므로 "east는 아직, CM은 곧"이 자유 학습이 아니라
        # 공식이 된다.
        #
        # head는 cell 공유 Linear (입력 = CNN cell feature 128 + 좌표
        # 2). 공유가 정규화의 핵심: CH는 어디 있든 같은 head가 같은
        # s/p를 뽑고 위치는 공식의 lon만 바꾼다. 좌표 입력은 공유
        # head가 반구/limb 감쇠 같은 위치 의존 gate를 배울 통로다
        # (per-cell 상수 입력이지만 head가 공유라 bias로 흡수되지
        # 않는다 -- flatten+Linear의 per-cell weight와 다른 지점).
        # delta-lon head(tanh * 22.5도)는 45도 column granularity를
        # 연속 경도로 보정한다 (자전 환산 +-41h -> 연속).
        # ============================================================

        self.register_buffer(
            "image_age_hours",
            torch.arange(19, -1, -1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "horizon_hours",
            torch.arange(1, 13, dtype=torch.float32) * 6.0,
        )
        # hindcast 재구성 격자. (v7: 자전 대기 항 덕에 west 소스가
        # 깊은 과거도 커버할 수 있지만 격자는 그대로 둔다 -- wind
        # 입력 창과의 대응이 단순하고 v6와 비교 가능하게.)
        self.register_buffer(
            "hindcast_hours",
            torch.arange(-12, 1, dtype=torch.float32) * 6.0,
        )

        # v7 좌표/기하 상수. 경도 column 중심(도, 음수=동쪽): 가시
        # 원반 [-90,+90]을 spatial_pool의 4 column이 균등 분할한다고
        # 근사. 위도 row는 기하식에 안 들어가고 head 좌표 입력 전용
        # (북/남 반구 구분). omega는 synodic 자전율(27.2753d) -- 고정
        # 상수, 튜닝 노브를 늘리지 않는다.
        self.register_buffer(
            "cell_lon_deg",
            torch.tensor([-67.5, -22.5, 22.5, 67.5]),
        )
        self.register_buffer(
            "cell_lat_norm",
            torch.tensor([0.5, -0.5]),
        )
        self.omega_deg_per_hour = 360.0 / (27.2753 * 24.0)

        # cell 공유 head. 입력 128은 CNN 출력 채널(d_model과 무관).
        head_in_features = 128 + 2
        self.source_speed_head = nn.Linear(head_in_features, 1)
        self.source_gate_head = nn.Linear(head_in_features, 1)
        self.transit_residual_head = nn.Linear(head_in_features, 1)
        self.lon_offset_head = nn.Linear(head_in_features, 1)

        # ballistic 상수: tau[h] = dist_eff / s[/1000 km/s 스케일].
        # 1 AU = 1.496e8 km / 3.6e6 = 41.6. 학습 가능 -- Parker
        # spiral/가속 구간의 평균 효과를 흡수한다 -- 하되 sigmoid로
        # [30, 55]에 유계: tau(430 km/s)가 [70, 128]h. raw가 O(1)
        # 스케일이 되어 물리 param group(train.py, 고lr)과 맞물린다.
        # init -0.144 -> 41.6.
        # (참고: v2/v3 주석의 "[48,120]h = 800..300 km/s"는 오류였다.
        #  실제 ballistic은 800->52h, 300->139h.)
        self.dist_eff_raw = nn.Parameter(torch.tensor(-0.144))
        # kernel 폭 sigma. 6h 격자에서 ±1스텝 0.88, ±4스텝 0.14.
        # 고정 상수 -- 실험 표면(튜닝 노브)을 늘리지 않는다.
        self.kernel_sigma_hours = 12.0
        # quiet fallback 가중치 (softplus -> init 1.0).
        self.fallback_weight_raw = nn.Parameter(torch.tensor(0.5413))
        # climatology (/1000 스케일). hindcast fallback과 base_h의
        # 복귀 목표를 겸한다. 주의: hindcast 경로의 fallback이
        # last_wind면 image-only 재구성에 wind가 새므로 반드시
        # per-sample 정보가 없는 상수여야 한다.
        self.climatology = nn.Parameter(torch.tensor(0.43))

        nn.init.zeros_(self.source_speed_head.weight)
        nn.init.constant_(self.source_speed_head.bias, -0.96)
        nn.init.zeros_(self.source_gate_head.weight)
        nn.init.zeros_(self.source_gate_head.bias)
        nn.init.zeros_(self.transit_residual_head.weight)
        nn.init.zeros_(self.transit_residual_head.bias)
        # zero-init -> delta-lon=0 = cell 중심에서 시작.
        nn.init.zeros_(self.lon_offset_head.weight)
        nn.init.zeros_(self.lon_offset_head.bias)

        # ============================================================
        # 6.6 Persistence anchor + fusion gate -- v5a
        # ============================================================
        #   base_h = last_wind + beta_h * (c - last_wind)
        #
        # "wind 중요도는 horizon이 늘수록 감소" prior를 horizon별
        # 스칼라 12개로 명시한다. init sigmoid(-4)=0.018 -> 순수
        # persistence (기존 residual과 동일 시작점). 학습된 beta_h가
        # h에 단조 증가하는지 자체가 prior의 검증 지표다.
        self.reversion_logit = nn.Parameter(torch.full((12,), -4.0))

        # alpha_h: 전파 branch를 얼마나 믿을지 (forecast 전용 융합).
        # source gate p_t(이미지 전용 -- hindcast 무결성 필수)와 달리
        # 여기는 hindcast loss 밖이므로 wind 레짐 요약을 봐도 안전하고,
        # 보는 게 맞다 (quiet 판단은 wind 상태와 결합된 정보).
        # modality drop 시 이미지 요약이 0이어도 bias 경로로 alpha를
        # 낮추는 것을 학습할 수 있다.
        # v5b: surge 확률(6.7)이 추가 입력 -- gate가 "quiet면 닫고
        # pre-surge면 연다"를 할 수 있는 판별 특징.
        self.fusion_image_proj = nn.Linear(d_model, 16)
        self.fusion_gate_head = nn.Linear(
            16 + 4 + (1 if use_surge_head else 0), 12
        )

        # zero-init + bias -2 -> alpha=0.12. 초기 v_img가 c 근방이라
        # 시작 효과는 beta 0.12 상당의 mean-reversion뿐이다. alpha의
        # gradient는 자기 값과 무관하게 (v_img - base)에 비례하고
        # v_img는 hindcast로 독립 학습되므로 v2 gate식 순환 사멸이
        # 없다.
        nn.init.zeros_(self.fusion_gate_head.weight)
        nn.init.constant_(self.fusion_gate_head.bias, -2.0)

        # ============================================================
        # 6.7 Surge head -- v5b
        # ============================================================
        # v5a branch decomposition이 남긴 결함 지도:
        #   - slow-quiet에서 alignment gain -10.6: quiet인데 v_img~430
        #     이 persistence(~360)를 끌어올리는 false-positive 소스
        #   - fusion gate가 레짐 무차별 상수(alpha 0.61~0.66): gate
        #     입력(wind 요약 + 이미지 평균)으로는 "quiet slow"와
        #     "pre-surge slow"를 구분할 특징이 없다 (slow-surge는
        #     align +22.8이라 함부로 닫혀서도 안 된다)
        #
        # 처방: 미래 target에서 파생한 binary surge label
        # (max_h W(t+h) - last_wind > 100 km/s, train.py)로 이미지
        # 전용 surge 확률을 supervise하고 그 확률을 fusion gate에
        # 연결한다. BCE는 v4가 회귀 loss로 주지 못했던 분류
        # supervision을 Δ채널(dimming/증광)에 준다.
        #
        # 이미지 전용인 이유: label이 미래 파생이라 wind를 봐도 누수는
        # 아니지만, wind를 주면 head가 wind 지름길로 새서 CNN(특히
        # Δ채널)으로 가는 gradient가 약해진다. wind와의 결합 판단은
        # 어차피 wind를 보는 fusion gate에서 일어난다.
        #
        # 입력 요약: mean = 디스크에 뭐가 있나, last5 - first5 = 최근
        # 발달 (CME 신호는 최근 프레임의 변화에 산다).
        self.use_surge_head = use_surge_head
        self.surge_head = nn.Sequential(
            nn.Linear(2 * d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 분석용: 마지막 forward의 전파 변수들 (detached).
        # v7: 소스 축이 (B,20)에서 (B,20,2,4)로 바뀌었다 (analysis.py
        # 가 소비).
        self.last_surge_prob = None
        self.last_coverage = None
        self.last_source_speed_kms = None
        self.last_arrival_hours = None
        self.last_source_gate = None
        self.last_source_lon_deg = None
        self.last_fusion_alpha = None

        # ============================================================
        # 7. Prediction head
        # ============================================================
        #
        # each decoder query:
        #
        # (d_model) -> one solar wind value
        # ============================================================

        if use_correction:
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            # Zero-init the last layer so the head starts at exactly
            # zero. Combined with the persistence residual in
            # forward(), the model starts as pure persistence and
            # learns only the correction on top of it. Without this
            # the initial prediction is wind[-1] plus a random offset
            # of roughly +-200 km/s.
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, images, wind, return_aux=False):
        # return_aux=True면 (prediction, aux dict)를 반환한다. aux는
        # train.py의 보조 손실용: hindcast 재구성(B,13), image_keep
        # (B,) modality-drop 마스크, transit_residual(B,20,2,4),
        # source_weight(B,20,2,4,25). 기본값 False라 inference.py 등
        # 기존 call site는 그대로 동작한다.

        batch_size = wind.size(0)

        # wind 시퀀스 원형은 마지막 fusion gate 요약(6.6)에도 쓴다.
        # last_wind는 persistence anchor의 기준값.
        #
        # (B,20) -> (B,1)
        wind_seq = wind
        last_wind = wind_seq[:, -1:]

        # ============================================================
        # IMAGE TOKENS
        # ============================================================
        # (v6a: wind 토큰화는 encoder 전용이라 CORRECTION PATH 블록
        #  안으로 이동했다. base/gate는 wind_seq를 직접 읽는다.)

        if self.use_images:
            # v8a: dataset이 uint8을 반환하면 여기(GPU)서 정규화한다.
            # 호스트 RAM/전송을 1/4로 줄이는 OOM 방어 (dataset.py 참고).
            # float 입력이 오면 no-op이라 기존 call site와 호환.
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            # dataset:
            #
            # images:
            # (B, T=20, C=2, H, W)  H=W=IMAGE_SIZE
            #
            # v4: running-difference 채널을 여기(GPU)서 만들어 concat.
            # 부호가 정보(음수=dimming, 양수=플레어)라 절대값을 취하지
            # 않고, [-1,1] 스케일 그대로 둔다 (클리핑/증폭은 효과 확인
            # 후의 튜닝 노브). 첫 프레임은 이전 프레임이 없어 0.
            # 채널 순서: (193, 211, Δ193, Δ211) -> C_in=4.
            if self.add_diff_channels:
                diff = torch.zeros_like(images)
                diff[:, 1:] = images[:, 1:] - images[:, :-1]
                images = torch.cat([images, diff], dim=2)

            # Conv3D wants:
            #
            # (B, C_in, T=20, H=64, W=64)
            image_features = images.permute(
                0, 2, 1, 3, 4
            ).contiguous()

            image_features = self.stem(image_features)
            image_features = self.image_encoder(image_features)

            # expected:
            #
            # (B, 128, 20, 4, 4)

            # pool the spatial grid to 2x4 (full longitude columns
            # survive, latitude folds to hemispheres):
            #
            # (B, 128, 20, 2, 4)
            image_features = self.spatial_pool(image_features)

            # v7: readout은 flatten 전의 cell feature를 직접 쓴다
            # (per-cell 소스 attribution). 토큰 경로(projection ->
            # fusion/surge 요약)는 v6와 동일하게 유지.
            #
            # (B, 128, 20, 2, 4) -> (B, 20, 2, 4, 128)
            cell_features = image_features.permute(0, 2, 3, 4, 1)

            # (B, 128, 20, 2, 4) -> (B, 20, 1024) -> (B, 20, d_model)
            image_tokens = self.image_projection(
                image_features.permute(0, 2, 1, 3, 4).flatten(2)
            )

            # (B,20,1) Bernoulli keep-mask, resampled every forward
            # pass. Zeroes the whole image token; the time/modality
            # embeddings added below still mark its slot.
            # Training only -- evaluation sees every frame.
            # v7: 같은 마스크를 readout의 cell feature에도 적용해 두
            # 경로가 항상 같은 프레임 가림을 본다.
            if self.training and self.time_mask_prob > 0:
                time_keep = (
                    torch.rand(
                        batch_size, 20, 1,
                        device=image_tokens.device,
                    )
                    >= self.time_mask_prob
                )
                image_tokens = image_tokens * time_keep
                cell_features = (
                    cell_features * time_keep.view(batch_size, 20, 1, 1, 1)
                )

            # Modality dropout: (B,1,1) per-sample keep-mask zeroing
            # all 20 image tokens at once. Unlike wind-only mode the
            # token slots stay in the sequence, carrying only the
            # time/modality embeddings. Training only.
            #
            # image_keep: drop된 샘플의 hindcast 재구성은 상수라
            # train.py가 이 마스크로 해당 샘플을 손실에서 제외한다.
            # force_image_drop은 분석용 결정론 drop (constructor 참고).
            image_keep = torch.ones(
                batch_size, device=image_tokens.device
            )
            modal_keep = None
            if self.training and self.modality_drop_prob > 0:
                modal_keep = (
                    torch.rand(
                        batch_size, 1, 1,
                        device=image_tokens.device,
                    )
                    >= self.modality_drop_prob
                )
            if self.force_image_drop:
                modal_keep = torch.zeros(
                    batch_size, 1, 1,
                    dtype=torch.bool, device=image_tokens.device,
                )
            if modal_keep is not None:
                image_tokens = image_tokens * modal_keep
                cell_features = (
                    cell_features * modal_keep.view(batch_size, 1, 1, 1, 1)
                )
                image_keep = modal_keep.float().reshape(batch_size)

            # ========================================================
            # PROPAGATION READOUT (section 6.5, v7 source-map)
            # ========================================================
            # post-mask cell feature에서 계산: modality drop된 샘플은
            # 상수 재구성이 되므로 image_keep으로 손실에서 제외되고,
            # time-mask된 개별 프레임은 head bias가 주는 "평균적
            # 소스"로 나타난다 (확률적 증강 노이즈로 수용).
            #
            # source_speed:     (B,20,2,4)  [0.25,0.90] = 250..900 km/s
            # source_gate:      (B,20,2,4)  >= 0
            # transit_residual: (B,20,2,4)  tanh = delta/24h
            # source_lon:       (B,20,2,4)  도(음수=동쪽), cell+offset
            # arrival:          (B,20,2,4)  도착 시각, 음수 = 과거
            # weight:           (B,20,2,4,25)  gate * kernel
            # v_image:          (B,25)      과거 13점 + 미래 12점
            # ========================================================

            coords = torch.cat(
                [
                    self.cell_lat_norm.view(1, 1, 2, 1, 1).expand(
                        batch_size, 20, 2, 4, 1
                    ),
                    (self.cell_lon_deg / 90.0).view(1, 1, 1, 4, 1).expand(
                        batch_size, 20, 2, 4, 1
                    ),
                ],
                dim=-1,
            )
            head_input = torch.cat([cell_features, coords], dim=-1)

            source_speed = 0.25 + 0.65 * torch.sigmoid(
                self.source_speed_head(head_input)
            ).squeeze(-1)
            source_gate = F.softplus(
                self.source_gate_head(head_input)
            ).squeeze(-1)
            transit_residual = torch.tanh(
                self.transit_residual_head(head_input)
            ).squeeze(-1)
            source_lon = self.cell_lon_deg.view(1, 1, 1, 4) + (
                22.5 * torch.tanh(
                    self.lon_offset_head(head_input)
                ).squeeze(-1)
            )

            # 자전 대기: 경도가 CM(지구 방향, lon=0)까지 오는 시간.
            # 동쪽(음수) 소스는 양수(미래 출발), 서쪽은 음수(이미 출발).
            rotation_wait = -source_lon / self.omega_deg_per_hour

            dist_eff = 30.0 + 25.0 * torch.sigmoid(self.dist_eff_raw)
            transit = dist_eff / source_speed + 24.0 * transit_residual
            arrival = (
                rotation_wait + transit
                - self.image_age_hours.view(1, 20, 1, 1)
            )

            time_grid = torch.cat(
                [self.hindcast_hours, self.horizon_hours]
            )
            kernel = torch.exp(
                -((time_grid - arrival.unsqueeze(-1)) ** 2)
                / (2.0 * self.kernel_sigma_hours**2)
            )
            weight = source_gate.unsqueeze(-1) * kernel

            fallback = F.softplus(self.fallback_weight_raw)
            weight_sum = weight.sum(dim=(1, 2, 3))
            v_image = (
                (weight * source_speed.unsqueeze(-1)).sum(dim=(1, 2, 3))
                + fallback * self.climatology
            ) / (weight_sum + fallback)

            hindcast = v_image[:, : self.hindcast_hours.numel()]
            v_image_future = v_image[:, self.hindcast_hours.numel():]

            # fusion gate(6.6)용 이미지 요약. embedding이 더해지기
            # 전의 content 토큰에서 계산한다.
            image_summary = F.gelu(
                self.fusion_image_proj(image_tokens.mean(dim=1))
            )

            # surge head (6.7): 이미지 전용, post-mask 토큰.
            # modality drop된 샘플은 상수 logit이 되므로 train.py가
            # image_keep으로 BCE에서 제외한다.
            if self.use_surge_head:
                surge_logit = self.surge_head(
                    torch.cat(
                        [
                            image_tokens.mean(dim=1),
                            image_tokens[:, -5:].mean(dim=1)
                            - image_tokens[:, :5].mean(dim=1),
                        ],
                        dim=-1,
                    )
                )
                surge_prob = torch.sigmoid(surge_logit)
                self.last_surge_prob = surge_prob.detach().squeeze(-1)
            else:
                surge_logit = None
                surge_prob = None
                self.last_surge_prob = None

            # coverage: 각 격자점(과거 13 + 미래 12) 재구성에서 이미지
            # 기여 비율 (1 - fallback 지분). 계속 낮으면 전파 branch가
            # 사실상 climatology라는 실패 신호.
            self.last_coverage = (
                weight_sum / (weight_sum + fallback)
            ).detach()
            self.last_source_speed_kms = source_speed.detach() * 1000.0
            self.last_arrival_hours = arrival.detach()
            self.last_source_gate = source_gate.detach()
            self.last_source_lon_deg = source_lon.detach()
        else:
            # wind-only diagnostic: skip the CNN and drop the image
            # stream entirely -- the encoder runs on wind tokens only.
            image_tokens = None
            hindcast = None
            image_keep = None
            transit_residual = None
            v_image_future = None
            image_summary = None
            surge_logit = None
            surge_prob = None
            weight = None
            self.last_surge_prob = None
            self.last_coverage = None
            self.last_source_speed_kms = None
            self.last_arrival_hours = None
            self.last_source_gate = None
            self.last_source_lon_deg = None

        # ============================================================
        # CORRECTION PATH: encoder + decoder + head (v6a: 통째 제거)
        # ============================================================
        # use_correction=False면 이 블록 전체가 빠지고 correction=0.
        # 남는 출력 경로는 base + alpha*(v_img - base)뿐이다 --
        # 근거와 판정 기준은 constructor의 use_correction 주석.
        # ============================================================
        if self.use_correction:
            # ========================================================
            # WIND TOKENS + MULTIMODAL SEQUENCE
            # ========================================================
            # Two parallel token streams over the same 20 timestamps:
            #
            # [
            #   image_t0, ..., image_t19,
            #   wind_t0,  ..., wind_t19,
            # ]
            #
            # Same-index image and wind share a time embedding but
            # stay separate tokens: wind_t left the Sun days before
            # image_t was taken, so fusing them would assert a causal
            # alignment that does not exist (see section 2 in
            # __init__). Attention is free to match wind to the
            # earlier images instead.
            #
            # shape:
            #
            # (B, 40, d_model) -- (B, 20, d_model) in wind-only mode
            # ========================================================

            # (B,20) -> (B,20,1) -> (B,20,d_model)
            wind_tokens = self.wind_projection(wind_seq.unsqueeze(-1))
            wind_tokens = (
                wind_tokens + self.pos_embedding + self.wind_modality
            )

            if image_tokens is not None:
                # content 토큰(image_tokens)은 건드리지 않는다 --
                # 아래 output assembly의 None 체크에 그대로 쓰인다.
                encoder_image_tokens = (
                    image_tokens
                    + self.pos_embedding
                    + self.image_modality
                )
                multimodal_tokens = torch.cat(
                    [encoder_image_tokens, wind_tokens],
                    dim=1,
                )
            else:
                multimodal_tokens = wind_tokens

            multimodal_tokens = self.token_norm(multimodal_tokens)

            # ========================================================
            # ENCODER
            # ========================================================
            # Every token can attend to every other token, across
            # both modalities and all timesteps.
            # memory:
            # (B,40,d_model)
            # ========================================================

            memory = self.transformer_encoder(multimodal_tokens)

            # ========================================================
            # FUTURE QUERIES + DECODER
            # ========================================================
            # query +6h..+72h가 encoder memory를 읽는다. 12개 동시
            # 예측 -- autoregressive loop/teacher forcing 없음.
            #
            # v5a: cross-attention에는 더 이상 temporal prior bias가
            # 없다. 시간 정렬은 section 6.5의 propagation branch가
            # 출력 경로에서 직접 담당하고, 이 decoder는 그 위의
            # 자유형 correction만 만든다.
            # ========================================================

            queries = self.future_queries.expand(batch_size, -1, -1)

            decoded = self.transformer_decoder(
                tgt=queries,
                memory=memory,
            )

            # (B,12,d_model) -> (B,12,1) -> (B,12)
            correction = self.output_head(decoded).squeeze(-1)

            # v5c correction dropout (constructor 참고). 1/(1-p)
            # 재스케일은 일부러 안 한다: 유지된 샘플은 eval과 동일한
            # 함수를 보므로 head가 정확한 스케일을 배우고, drop된
            # 샘플만 base+전파 단독 적합을 훈련한다 (기대값 보존
            # 재스케일은 per-sample drop에선 오히려 eval 스케일을
            # 어긋나게 한다).
            if self.training and self.correction_drop_prob > 0:
                correction_keep = (
                    torch.rand(
                        batch_size, 1, device=correction.device
                    )
                    >= self.correction_drop_prob
                )
                correction = correction * correction_keep
        else:
            # v6a: correction 없음. 0 텐서로 두면 output assembly와
            # 분석 스크립트(branch_decomposition의 재조립 검증)가
            # 수정 없이 성립한다 (base+corr = base, full = base+prop).
            correction = wind_seq.new_zeros(batch_size, 12)

        # ============================================================
        # OUTPUT ASSEMBLY (section 6.6)
        # ============================================================
        #   v_hat(h) = base_h + alpha_h*(v_img(h) - base_h) + corr(h)
        #   base_h   = last_wind + beta_h*(climatology - last_wind)
        #
        # correction head는 zero-init이므로 시작점은 base(+alpha
        # 0.12의 미미한 c-복귀 항) = 사실상 persistence. wind/target은
        # dataset.py의 /1000 스케일을 공유한다.
        # ============================================================

        beta = torch.sigmoid(self.reversion_logit)
        base = last_wind + beta * (self.climatology - last_wind)

        if image_tokens is not None:
            # alpha_h (B,12): 이미지 요약 + wind 레짐 요약으로 전파
            # branch의 신뢰도를 정한다 (forecast 전용 -- hindcast
            # 경로와 무관하므로 wind를 봐도 새지 않는다).
            wind_summary = torch.stack(
                [
                    wind_seq[:, -1],
                    wind_seq.mean(dim=1),
                    wind_seq.std(dim=1),
                    wind_seq[:, -1] - wind_seq[:, 0],
                ],
                dim=-1,
            )
            gate_inputs = [image_summary, wind_summary]
            if surge_prob is not None:
                # gradient를 끊지 않는다: BCE가 의미를 앵커하고,
                # forecast loss의 미세 조정은 허용.
                gate_inputs.append(surge_prob)
            alpha = torch.sigmoid(
                self.fusion_gate_head(
                    torch.cat(gate_inputs, dim=-1)
                )
            )
            prediction = (
                base + alpha * (v_image_future - base) + correction
            )
            self.last_fusion_alpha = alpha.detach()
            self.last_v_image_future = v_image_future.detach()
        else:
            prediction = base + correction
            self.last_fusion_alpha = None
            self.last_v_image_future = None

        # 분석용 (branch_decomposition.py): 출력 성분 분해.
        # pred = base + alpha*(v_img - base) + correction
        self.last_base = base.detach().expand(batch_size, 12)
        self.last_correction = correction.detach()

        if return_aux:
            return prediction, {
                "hindcast": hindcast,
                "image_keep": image_keep,
                "transit_residual": transit_residual,
                "surge_logit": surge_logit,
                # v7: backmapping alignment loss(train.py)용 미정규화
                # kernel weight (B,20,2,4,25). label에서 역산한 소스
                # 궤적 분포와 이 분포의 정규화본을 KL로 맞춘다.
                "source_weight": weight,
            }
        return prediction
