# Diffusion Model 연구 입문 커리큘럼 (4주 / 약 30일)

Computer Vision 랩(특히 diffusion model, diffusion synchronization, text-to-3D 계열) 진학을 목표로, DDPM부터 최신 연구 논문(MultiDiffusion, SyncDiffusion, DreamFusion, SyncSDE 등)까지 스스로 따라갈 수 있는 기초 체력을 만들기 위한 4주짜리 자기주도 커리큘럼이다.

`../computer-vision/05-generative-models-beyond-the-course.md`에 이미 GAN/diffusion/3D diffusion의 넓은 개요(overview)가 정리되어 있다. 이 커리큘럼은 그 개요를 전제로 깔고, **수식 유도 + 직접 구현 + 논문 원문 정독** 수준까지 파고드는 것을 목표로 한다. 즉 "이런 게 있다더라"에서 "왜 그렇게 되는지 손으로 증명하고 코드로 재현할 수 있다"로 넘어가는 게 이 폴더의 목적이다.

## 이 폴더의 성격

이 챕터 파일들(`00-` ~ `07-`)은 커리큘럼 초안을 8개의 서브에이전트가 병렬로 검증·보강·확장해서 완성한 학습 노트다 (기존 `computer-vision/` 폴더를 4개 에이전트가 병렬로 채운 것과 같은 방식). 각 챕터는 아래 루브릭(A~G, 1~5점)으로 자체 채점 후 4점 미만 항목이 있으면 스스로 보강하는 과정을 거쳤고, 완료 후 이미지 링크 유효성/메타 지시문 삭제 여부/핵심 수식과 사실관계를 직접 재검수했다 - 8개 챕터 모두 전 항목 4점 이상으로 통과했다.

- A. 개념 스캐폴딩 (직관 → 수식 순서, 갑작스러운 도약 없음)
- B. 초심자 친화성 (전문용어 첫 등장 시 설명)
- C. 시각자료 (최소 2개 이상, 실질적으로 이해를 돕는가 - 대부분 실제로 실행한 코드의 결과)
- D. 정확성 (수식/사실관계 오류 없음, 확인된 사실과 추론을 구분해서 표기)
- E. 자기완결성 & 챕터 간 연결
- F. 실습 연계
- G. 저장소 톤 일관성

각 챕터의 실습에서 실제로 실행한 것들: 00장 가우시안 곱/ELBO 수치 검증, 01장 AE/VAE/GAN을 MNIST에 실제 학습(GAN은 계산 자원 제약으로 MLP로 축소, 문서에 명시), 02장 작은 U-Net을 MNIST에 3 epoch 끝까지 학습시켜 실제 생성 샘플 확보, 03장 2D toy 데이터로 score matching + Langevin dynamics 수렴 확인, 04장 GMM 오라클로 DDIM/DDPM·CFG scale 정량 비교 + 실제 `diffusers` 파이프라인 실행, 05장 MultiDiffusion 방식 토이 실험(seam 불연속 정량 비교), 06장 SyncSDE arXiv 원문/GitHub 코드 직접 대조, 07장 correlation 모델 3종 비교 실험.

## 전체 일정 (4주)

| 주차 | Day | 챕터 | 핵심 내용 |
|---|---|---|---|
| 1주차 | 1-2 | [00-math-foundations.md](00-math-foundations.md) | 가우시안 분포, KL divergence, ELBO, 확률과정 직관 |
| 1주차 | 3-6 | [01-generative-model-landscape.md](01-generative-model-landscape.md) | AE/VAE 수식 유도 + 구현, GAN 복습 |
| 1주차 | 7 | 복습 | 00~01 정리, 막힌 부분 재정리 |
| 2주차 | 8-13 | [02-ddpm-deep-dive.md](02-ddpm-deep-dive.md) | DDPM forward/reverse 수식 완전 유도 + MNIST/CIFAR-10 구현 |
| 2주차 | 14 | 복습 | DDPM 구현 디버깅 + 정리 |
| 3주차 | 15-17 | [03-score-based-sde.md](03-score-based-sde.md) | Score matching, Langevin dynamics, Score SDE, 2D toy 구현 |
| 3주차 | 18-21 | [04-sampling-and-guidance.md](04-sampling-and-guidance.md) | DDIM, Classifier/Classifier-Free Guidance, Latent Diffusion |
| 4주차 | 22-24 | [05-diffusion-synchronization-survey.md](05-diffusion-synchronization-survey.md) | MultiDiffusion, SyncDiffusion, DreamFusion(SDS), TexFusion |
| 4주차 | 25-26 | [06-syncsde-deep-dive.md](06-syncsde-deep-dive.md) | SyncSDE 논문 완전 정독 + 수식 재유도 |
| 4주차 | 27-30 | [07-research-transition.md](07-research-transition.md) | 미니 재현 프로젝트, 랩 서칭, 논문 팔로우 습관 |

하루에 낼 수 있는 시간에 따라 ±50% 정도는 자연스럽게 늘어날 수 있다. 중요한 건 순서를 건너뛰지 않는 것 — 특히 00→01→02는 반드시 순서대로 가야 한다 (뒤 챕터가 앞 챕터의 수식을 그대로 재사용함).

## 읽는 원칙

1. **논문보다 블로그/코스 먼저**: 각 챕터에 적힌 "쉬운 자료"를 먼저 보고 감을 잡은 뒤 논문 원문으로 넘어갈 것. 논문부터 읽으면 표기법(notation) 차이 때문에 불필요하게 막힌다.
2. **읽기만 하지 말고 반드시 구현**: 특히 02, 03 챕터는 코드로 직접 돌려보지 않으면 이해했다는 착각만 남는다.
3. **막히면 표시만 해두고 진도 유지**: 완벽히 이해 안 돼도 다음 챕터로 넘어가고, 나중에 다시 돌아온다. (예: SDE의 엄밀한 Ito 적분은 몰라도 DDPM 구현엔 지장 없음)

## 기존 자료와의 관계

- `../computer-vision/05-generative-models-beyond-the-course.md` — 이 커리큘럼 전체의 "목차 겸 미리보기"에 해당. 이미 아는 내용이면 각 챕터 서두의 "선수 확인" 체크리스트로 스킵 여부 판단.
- `../ML/` — ML Specialization 노트. 확률/최적화 기초가 가물가물하면 참고.

## 진행 상황 체크리스트 (챕터 작성 기준 - 학습자 본인의 학습 진도는 별도로 체크할 것)

- [x] 00. Math Foundations
- [x] 01. Generative Model Landscape (AE/VAE/GAN)
- [x] 02. DDPM Deep Dive
- [x] 03. Score-based / SDE
- [x] 04. Sampling & Guidance (DDIM/CFG/Latent Diffusion)
- [x] 05. Diffusion Synchronization Survey
- [x] 06. SyncSDE Deep Dive
- [x] 07. Research Transition
