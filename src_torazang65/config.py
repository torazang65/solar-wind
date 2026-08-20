import os
import random
from pathlib import Path
import numpy as np
import torch

# ==========================================
# 1. Seed Setup
# ==========================================
# SEED=1234 python train.py 처럼 환경변수로 오버라이드. v2/v3 비교에서
# 그룹별 delta ±5가 학습 궤적 노이즈로 판명됐으므로, 아키텍처 변경의
# 판정은 seed 2개 이상 평균으로 한다. OUTPUT_DIR에 seed가 들어가
# 멀티시드 런이 서로 덮어쓰지 않는다.
SEED = int(os.environ.get("SEED", "777"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. Paths & Directories
# ==========================================
DATA_ROOT = Path("public_dataset/competition_dataset_6h")
# train.py가 시작 시 기존 best_model.pth를 지우므로, 런이 바뀔 때마다
# 이름을 올려 이전 산출물을 보존한다.
OUTPUT_DIR = Path(f"outputs/transformer_v7_srcmap_torazang65_seed{SEED}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("outputs/cache")

# ==========================================
# 3. Hyperparameters
# ==========================================
IMAGE_SIZE = 64
CHANNELS = ("193", "211")
RMSE_EPSILON = 1e-8
BATCH_SIZE = 256
# v6b: early stopping 전에 cosine tail을 실제로 지나가도록 학습 길이를
# patience와 정합한다. v6a는 100-epoch cosine을 정의해 놓고 epoch 26에
# 멈춰 lr이 peak의 88%인 채로 끝났다.
EPOCHS = 35
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 4

# ---- 학습률 스케줄 ----
# v2 런의 병목: lr=1e-4에서 val 바닥이 epoch 4에 왔고, 이후 train은
# 계속 내려가는데 val은 다시 69.8 밑으로 못 왔다. ReduceLROnPlateau의
# 첫 감축(epoch 10)은 이미 과적합 구간이라 늦었다. peak를 낮추고
# cosine으로 1에폭부터 계속 식혀서 바닥을 늦고 깊게 만드는 것이 목표.
# zero-init prior가 학습될 시간을 버는 효과도 겸한다. PEAK_LR가 주된
# 스윕 대상.
PEAK_LR = 3e-5
MIN_LR = 1e-6
WARMUP_EPOCHS = 3

# ---- v5a: propagation/hindcast 보조 손실 ----
# hindcast: 이미지 전용 전파 branch(model.py 6.5)가 관측된 최근 wind
# (s=-72..0h)를 재구성하는 손실. wind persistence 지름길이 없는 경로에
# 시간 정렬 supervision을 두는 것이 v2/v3 prior 사멸의 해법. 초반에
# 크게 시작해(정렬부터 학습) DECAY_EPOCHS에 걸쳐 END로 선형 감쇠.
# 전부 스윕하지 않는 고정 기본값 -- mechanism 지표(hindcast RMSE,
# alpha/beta 거동)가 학습을 확인한 뒤에만 튜닝 대상으로 승격한다.
# v6b: v6a에서 epoch 6의 forecast 최적점 뒤에도 hindcast 개선 압력이
# 강하게 남아 두 목표가 경쟁했다. 정렬 학습에는 쓰되, 예측 최적화가
# 주도권을 빠르게 되찾도록 초반 8 epoch에 감쇠를 끝낸다.
HINDCAST_LAMBDA_START = 0.7
HINDCAST_LAMBDA_END = 0.1
HINDCAST_LAMBDA_DECAY_EPOCHS = 8
# ballistic 잔차(tanh 출력 = delta/24h)의 L2 계수. tau를 D_eff/s
# backbone 근방에 잡아둬 "값을 아무 시점에나 배치"하는 퇴화를 막는다.
TRANSIT_RESIDUAL_L2 = 3e-3

# ---- v7: backmapping alignment 보조 손실 ----
# 진단(v6b): hindcast/forecast는 값(value)만 감독해 RMSE 최적해가
# "여러 소스를 넓게 평균"이 되고, 그 뭉갬이 속도 조건부 정렬을
# 구조적으로 억압했다 (COM 속도군 분화 7h vs 이론 30h, gate 9.1,
# coverage 0.97). 이 손실은 위치(location)를 직접 감독한다:
#
#   label (u, y_u)에서 radial 통과시간 tau* = D_eff/y_u를 역산하면
#   책임 소스의 출발 시각은 u - tau* (출발 시 CM). 자전을 되감으면
#   각 프레임(age a)에서 그 소스가 보였던 경도
#       lon*(a) = -omega * (u - tau* + a)
#   가 나온다 -- (frame, lon-cell) 격자 위의 궤적. 이 궤적 주변
#   가우시안(soft 타깃 q)과 모델의 정규화 kernel weight를 KL로 맞춘다.
#
# u는 hindcast 격자(-72..0h, 관측 wind -- 누수 없음)와 forecast 격자
# (+6..+72h, train label)를 모두 쓴다. |lon*| > 90도(가시 원반 밖)인
# (frame, u)는 타깃 질량 0, 전 프레임이 밖이면 그 u는 손실 제외.
# tau* 계산의 D_eff는 모델이 학습 중인 값을 detach해 사용 -- 물리
# 상수를 박으면 모델 내부 스케일과 어긋나 residual이 갭을 메운다.
#
# LAMBDA: KL 스케일은 O(1)(퍼펙트 매치 0, 초기 ~2-4), forecast loss는
# ~0.08이므로 0.02면 초기 기여 ~0.05 (동급). 판정 후 스윕 후보.
# SIGMA_DEG: SIR 압축/가속으로 지구 y_u != 출발 속도라 타깃은 soft해야
# 한다. 20도 = 자전 ~36h; transit residual +-24h(=13도)를 덮는 폭.
ALIGN_LAMBDA = 0.02
ALIGN_SIGMA_DEG = 20.0

# ---- v5b: surge head 보조 손실 ----
# label: max_h W(t+h) - last_wind > threshold (speed_ablation의 surge
# 정의와 동일). BCE는 작은 regularizer로 취급 -- RMSE를 직접 사는
# 항이 아니라 Δ채널 표현과 fusion gate 판별 특징을 만드는 항.
SURGE_THRESHOLD_KMS = 100.0
SURGE_LAMBDA = 0.02
# 클래스 불균형 보정 (val 기준 quiet:surge ~ 836:363 ~ 2.3).
SURGE_POS_WEIGHT = 2.3
# 물리 스칼라 param group의 lr 배율 (train.py is_physical_param).
# AdamW 스텝은 ~lr이라 base lr 총합(~0.1)로는 O(1) 스케일 스칼라
# (reversion logit, dist_eff raw)가 초기값에 얼어붙는다. fusion gate는
# v6a에서 alpha를 0.59 -> 0.93까지 포화시킨 원인 후보라 base lr로 둔다.
PHYSICAL_LR_MULT = 100.0

# ==========================================
# 3.5 Model architecture
# ==========================================
# train.py와 inference.py가 같은 구조를 쓰도록 여기서 한 번만 정의한다.
# 값이 어긋나면 체크포인트 load_state_dict가 실패한다.
#
# 유효표본이 수백 개 수준(9,607 샘플이지만 6시간 stride로 19배 중복)이라
# 이전 설정(d_model=256, enc3/dec2, 3.74M)은 과적합했다. e21에 val 65.951로
# 바닥을 치고 상승 전환.
# wind_dim은 제거됨: 이미지/wind가 타임스텝별로 한 토큰에 concat되던
# 구조에서 각자 별도 토큰 스트림(20+20=40 토큰)으로 분리되면서
# 두 모달리티 모두 full d_model을 쓴다. 기존 체크포인트와는 호환 안 됨.
#
# v2 (2x4pool_dynprior): attention/ablation 분석 결과를 반영한 두 가지
# 구조 변경. 이전 체크포인트와 호환 안 됨 (image_projection 1024-dim,
# prior 모듈 추가).
#  1) GAP -> 2x4 (lat x lon) pooling: 이미지 기여의 실체가 "도래
#     스트림 감지"인데 GAP가 코로나홀의 중앙/가장자리 위치를 지우고
#     있었다 (fast&quiet에서 이미지 gain이 유의미한 음수 = false
#     positive 정황). 대칭 2x2는 center cell이 없어 central meridian
#     거리 표현이 약하므로, longitude 4열을 그대로 보존하고 latitude만
#     반으로 접는 비등방 풀링을 쓴다.
#  2) dynamic temporal prior: 디코더 cross-attention의 시점 선택이
#     샘플 불변 고정 템플릿이었다 (slow-fast COM gap -2h vs 이론 +34h).
#     이미지에서 읽은 solar-state로 통과시간 tau를 예측해 attention
#     logit에 가우시안 bias를 더한다. gate=0이면 v1으로 퇴화.
# v4 (diffchan): 입력에 running-difference 채널 추가 (193, 211, Δ193,
# Δ211). CNN의 시간 커널이 전부 1이라 CME의 on-disk 신호(coronal
# dimming/플레어 증광 = 프레임 간 픽셀 차분)를 모델이 스스로 만들 수
# 없어 공급한다. Δ는 model.forward가 GPU에서 계산한다 -- dataset(CPU)
# 계산은 DataLoader 배치 메모리를 2배로 만들어 pod OOM을 냈다. 호스트
# RAM 프로필은 v3와 동일. 판정: 같은 스케줄인 v3 대비 surge/event
# 지표(surge gain, event RMSE, worst-15) + seed 2개 평균. 전체 RMSE는
# 노이즈(±1.5)에 묻히므로 판정 기준이 아니다.
# v5a (propagation): dynamic prior(구 section 6.5)를 propagation
# readout + persistence anchor + fusion gate로 교체 (파라미터 순감
# ~14k). 입력 구성(4ch diff)은 v4 그대로 유지 -- 전파 thesis와
# 독립이고 v5b(surge head)가 diff 정보를 쓸 예정. 이전 체크포인트와
# 호환 안 됨. 판정: 멀티시드 >=2 + mechanism 지표 우선(hindcast RMSE
# vs climatology, source speed-실측 상관, beta_h 단조성, alpha_h 레짐
# 분포) -- val RMSE는 노이즈에 묻혀도 이 지표들로 구조 작동 여부를
# 1런에 판정할 수 있다.
# v6a (prop-only): correction 경로(transformer encoder/decoder/output
# head + wind 토큰화) 완전 제거 -- 이것 하나만 바꾸는 단일 변수 실험.
# 근거 (v5b branch decomposition, seed 777):
#   - base+prop 67.27 < full 69.23. corr own-gain은 quiet +4.0(유의)
#     / surge -3.3(유의)이고, full은 surge에서 base+prop 대비 -7~-9.
#   - decoder attention COM이 horizon/속도군 무관 ~55h 고정: 정렬된
#     읽기가 아니라 train 패턴 bag-of-frames 지름길.
#   - correction_drop 0.3에도 epoch-6 조기 best. 그동안 val hindcast
#     는 epoch 13+까지 계속 개선 -- 전파 branch 성숙이 체크포인트에
#     못 담기는 병목의 주범이 correction 과적합.
# correction_drop=0.3 덕에 학습의 30%는 이미 base+prop 단독 적합을
# 훈련했으므로 67.27은 재학습 기대치의 합리적 추정이다. 판정
# (seed 3개 평균: 777/1234/2024, 그룹 delta 단독 판정 금지):
#   (1) val <= 67.5: ablation 67.27 재현. 미달이면 correction이 학습
#       중 정규화로 일했다는 뜻 -- 그 자체가 정보.
#   (2) best epoch > 10: 조기 종결 해소, hindcast 성숙이 담기는지.
#   (3) surge에서 v5b full 대비 -7~-9 회수 + slow-quiet align 비음수
#       유지 (branch_decomposition으로 확인).
#   (4) quiet 열화 폭 = correction의 +4.0 중 잔존분. 크면 다음 수는
#       correction 재도입이 아니라 base의 AR(2) 강화다 (correction의
#       quiet 기여 실체는 wind 단기 자기상관 표현으로 추정 --
#       taeukjung 브랜치에서 AR(2) 검증됨).
# v7 (srcmap): 소스 단위를 프레임 x 디스크 cell(20x2x4)로 분해, 자전
# 기하를 도착 공식에 내장(arrival = -lon/omega + D/s - age), label
# 역산 backmapping alignment KL 추가 (위 ALIGN_* 주석). 구조 근거:
# v6b에서 정렬이 속도 무관 고정 템플릿(tau~84h, COM 분화 7h vs 이론
# 30h)으로 퇴화 -- 값 감독만으로는 "넓게 평균"이 RMSE 최적해라 위치
# 감독과 경도 분해가 필요하다. head는 cell 공유(+좌표 2 입력)라
# 파라미터 순증 ~0.5k. 이전 체크포인트와 호환 안 됨.
# 판정 (seed 3개: 777/1234/2024, analysis.py):
#   (1) launch-age COM의 slow-fast 분화 7h -> >=20h (이론 ~30h)
#   (2) 궤적 추적: (sample,u) 내 lon-vs-age 기울기 ~ -0.55 deg/h
#   (3) best epoch > 10 + best 시점 train-val gap 축소 (과적합 완화의
#       "when" 몫이 실재하는지)
#   (4) val <= 67.5 (v6a 재현 이상) + surge/quiet 이득 유지
#   (5) gate 뭉갬 해소: coverage/align KL 하락 추세
MODEL_KWARGS = dict(
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=1,
    dim_feedforward=256,
    dropout=0.1,
    # 학습 중 이미지 토큰을 타임스텝 단위로 마스킹하는 확률.
    # 0으로 두면 증강이 꺼진다. 주된 스윕 대상.
    time_mask_prob=0.15,
    # 학습 중 샘플 단위로 이미지 스트림 전체를 drop하는 확률.
    # wind-only 경로가 항상 자립하도록 강제해서 이미지 경로 암기를
    # 억제한다. 0이면 꺼짐. time_mask_prob와 함께 스윕 대상.
    modality_drop_prob=0.25,
    # v6a: correction 경로 완전 제거 (섹션 3.5 주석). True로 되돌리면
    # v5b 구조 (그때는 correction_drop_prob=0.3도 함께 복원할 것 --
    # 경로가 없으면 무의미해서 뺐다).
    use_correction=False,
    # v5b: 이미지 전용 surge head + 그 확률을 fusion gate에 연결.
    # v5a 분해의 결함 지도(slow-quiet align -10.6, gate 레짐 무차별)
    # 를 표적한다. False면 v5c 구조와 동일 (구 체크포인트 분석 시
    # OUTPUT_DIR과 함께 되돌릴 것 -- gate 입력 폭이 달라진다).
    # 판정: branch_decomposition의 slow-quiet align 회복 + alpha의
    # surge/quiet 분화 + val surge AUROC.
    use_surge_head=True,
    # stem 입력 채널. add_diff_channels=True면 원본 채널 수의 2배.
    # v3 이전으로 되돌리려면 (2, False)로.
    image_in_channels=4,
    add_diff_channels=True,
)

# ==========================================

# ==========================================
# 4. Device Setup
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: CUDA Unavailable")
