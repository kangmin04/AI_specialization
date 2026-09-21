# Generative Models: GAN & Diffusion (강의 범위 밖 보충 노트)

앞의 `01~04` 노트는 Andrew Ng의 Deep Learning Specialization **Course 4 (Convolutional Neural Networks)** 커리큘럼을 그대로 따라간 건데, 이 강의는 2017~2018년에 만들어져서 GAN도, diffusion model도 다루지 않는다. 근데 "컴퓨터 비전"이라는 분야로 넓혀보면 오히려 최근 몇 년간 제일 임팩트가 컸던 게 이미지/3D를 **생성**하는 모델들이라, 강의엔 없지만 따로 정리해둔다.

이 노트는 강의처럼 체계적으로 검증된 커리큘럼이 아니라, 넓은 흐름을 잡기 위한 보충 자료라는 점을 감안하고 읽으면 된다.

## 1. Generative Model이란 - Discriminative와 뭐가 다른가

지금까지 `01~04`에서 다룬 CNN들은 전부 **discriminative model**이다. 이미지가 들어오면 "이게 무슨 클래스인가", "어디에 뭐가 있는가"처럼 입력에 대한 판단(label)을 출력한다.

**Generative model**은 반대로, "그럴듯한 이미지 자체를 새로 만들어내는" 걸 목표로 한다. 데이터의 분포 `p(x)` 자체를 (직접 또는 간접적으로) 학습해서, 거기서 샘플링하면 진짜 같은 새로운 이미지가 나오게 만드는 것. 이 관점에서 보면 04번 노트에서 다룬 Neural Style Transfer도 "이미지 자체를 최적화 대상으로 삼는다"는 점에서 생성 모델과 발상이 맞닿아 있다.

Generative model로 가는 대표적인 두 갈래 길이 GAN과 diffusion model이다. (참고로 VAE(Variational Autoencoder)도 있지만, 최근 실무에서는 GAN/diffusion 쪽이 훨씬 많이 쓰여서 이 노트에선 짧게만 언급한다.)

- **VAE 한 줄 요약**: 인코더로 이미지를 압축된 latent vector로 만들고, 디코더로 다시 이미지를 복원하도록 학습하되, latent space가 정규분포를 따르도록 강제해서 학습 후엔 latent space에서 랜덤 샘플링 -> 디코더만으로 새 이미지를 생성할 수 있게 만든 모델. 생성 이미지가 흐릿한(blurry) 경향이 있다는 게 단점으로 자주 지적된다.

## 2. GAN (Generative Adversarial Network)

- 아이디어(Goodfellow et al., 2014): 서로 경쟁하는 두 네트워크를 동시에 학습시킨다.
  - **Generator (G)**: 랜덤 노이즈 벡터 `z`를 입력받아 가짜 이미지를 만들어낸다.
  - **Discriminator (D)**: 진짜 이미지와 G가 만든 가짜 이미지를 구분하려고 한다.
  - G는 D를 속이는 쪽으로, D는 안 속는 쪽으로 서로 계속 발전하다 보면(minimax game), 결국 G가 진짜와 구분 안 되는 이미지를 만들어내게 된다는 게 핵심 발상이다. 위조지폐범(G)과 경찰(D)의 비유로 많이 설명된다.
- **학습이 어려운 이유**: 두 네트워크를 동시에 균형 있게 학습시켜야 해서 훈련이 불안정하다. 대표적인 문제가 **mode collapse** - Generator가 다양한 이미지를 만들지 않고 Discriminator를 속이기 쉬운 몇 가지 패턴만 반복해서 만들어버리는 현상.
- **대표적인 변형들** (이름과 핵심 아이디어만 짚어둔다):
  - **DCGAN**: convolution 기반으로 GAN을 안정적으로 학습시키는 아키텍처 가이드라인을 제시.
  - **Conditional GAN (cGAN)**: 노이즈뿐 아니라 클래스 라벨 등 조건 정보를 같이 넣어서 "원하는 조건의" 이미지를 생성하게 함.
  - **pix2pix**: 이미지를 다른 이미지로 변환(image-to-image translation, 예: 스케치 -> 사진)하는 conditional GAN.
  - **CycleGAN**: 짝지어진(paired) 학습 데이터 없이도 두 도메인(예: 말 <-> 얼룩말 사진) 간 변환을 학습.
  - **StyleGAN**: 사람 얼굴 생성 등에서 매우 높은 품질을 보여준 아키텍처로, latent space를 조작해서 스타일(머리색, 표정 등)을 세밀하게 제어할 수 있게 만듦.

## 3. Diffusion Model

2020년대 들어 이미지 생성의 주류로 자리잡은 방식. GAN보다 학습이 안정적이고 다양성(diversity)이 좋다는 평가를 받는다.

### 기본 아이디어: Noise를 더했다가 다시 제거하는 법을 배운다

- **Forward process (noising)**: 진짜 이미지에 아주 작은 가우시안 노이즈를 여러 단계(step)에 걸쳐 조금씩 계속 더해서, 마지막엔 완전히 순수한 노이즈(랜덤 가우시안)가 되게 만든다. 이 과정은 학습이 필요 없는, 정해진 수식(고정된 noise schedule)이다.
- **Reverse process (denoising)**: 이 과정을 거꾸로 되짚어서, "한 단계 더 노이즈가 낀 이미지"가 주어졌을 때 "한 단계 덜 노이즈가 낀 이미지"를 예측하는 걸 신경망(주로 U-Net)에게 학습시킨다.
- 학습이 끝나면, 완전한 랜덤 노이즈에서 시작해서 이 denoising 네트워크를 여러 step 반복 적용하면 점점 노이즈가 걷히면서 진짜 같은 이미지가 만들어진다.
- 이게 **DDPM (Denoising Diffusion Probabilistic Models, Ho et al., 2020)**의 핵심 아이디어다. 왜 잘 되냐면, GAN처럼 "한 번에 진짜 vs 가짜를 놓고 경쟁"하는 게 아니라 "아주 작은 노이즈 한 스텝만 제거하면 되는 훨씬 쉬운 문제"를 반복해서 풀기 때문에 학습이 훨씬 안정적이다.
- **왜 U-Net을 쓰는가**: `03-object-detection.md`에서 semantic segmentation용으로 다룬 U-Net의 encoder-decoder + skip connection 구조가, "입력 이미지와 같은 크기의 출력(=예측한 노이즈)을 만들어야 하면서 동시에 전역적인 맥락도 봐야 하는" denoising 문제에도 그대로 잘 맞아서, diffusion model의 기본 backbone으로 널리 쓰인다.

### Guidance - 원하는 조건대로 생성하게 만들기

- **Classifier guidance**: 별도로 학습된 분류기(classifier)의 gradient를 이용해서, denoising 과정이 원하는 클래스 쪽으로 편향되게 유도.
- **Classifier-free guidance**: 별도 classifier 없이, "조건을 준 경우"와 "조건 없이(unconditional)" 두 가지로 학습한 하나의 모델에서 두 예측의 차이를 이용해 조건을 더 강하게/약하게 반영하는 방법. 최근 text-to-image 모델(예: 텍스트 프롬프트 반영)에서 표준처럼 쓰인다.

### Latent Diffusion Model (Stable Diffusion)

- 문제: 원본 픽셀 공간(예: 512x512x3)에서 매 step마다 diffusion을 돌리면 연산량이 너무 크다.
- 해결: VAE 같은 걸로 이미지를 훨씬 작은 latent space(예: 64x64 크기)로 압축해두고, **그 latent space 안에서** diffusion(noising/denoising)을 수행한 뒤, 마지막에 다시 VAE decoder로 원래 해상도 이미지로 복원한다.
- 이 방식이 Stable Diffusion(2022)의 핵심 아이디어로, 픽셀 공간에서 직접 diffusion을 도는 것보다 훨씬 적은 연산으로 고해상도 이미지 생성이 가능해졌다.

## 4. 3D Diffusion Model - 이미지를 넘어 3D로

3D 게임 에셋, AR/VR 콘텐츠, 3D 프린팅용 모델 등 수요가 늘면서, "텍스트나 이미지 한 장으로 3D 모델을 생성"하는 연구가 활발해졌다. 다만 3D는 이미지와 달리 "표현 방식" 자체가 여러 가지라, 어떤 표현을 쓰느냐에 따라 접근법이 갈린다.

### 3D를 표현하는 방법들 (간단히)

- **Voxel**: 3D를 격자로 나눠서 각 칸이 차 있는지/색이 뭔지로 표현. 이미지의 3D 버전이라 직관적이지만 해상도를 올리면 메모리가 세제곱으로 늘어난다.
- **Point cloud**: 3D 공간의 점들(x, y, z 좌표) 집합으로 표현. 메모리 효율은 좋지만 표면(surface) 정보가 명시적이지 않다.
- **Mesh**: 정점(vertex)과 면(face)으로 표면을 표현. 그래픽스/게임 엔진에서 표준으로 쓰이는 형식.
- **Implicit function / NeRF (Neural Radiance Field)**: 3D 공간의 좌표를 입력받아 "그 지점의 밀도와 색"을 출력하는 함수 자체를 신경망으로 학습하는 방식. 명시적인 형태(mesh 등)를 저장하는 게 아니라 함수로 3D 장면을 표현한다는 점이 독특하다. 여러 각도에서 찍은 2D 사진들만으로 이 함수를 학습시켜서, 학습 후엔 임의의 새로운 각도에서 렌더링(novel view synthesis)할 수 있다.

### 2D Diffusion을 3D로 확장하는 두 갈래

1. **3D representation에 직접 diffusion 적용**: point cloud나 voxel 같은 3D 데이터에 forward/reverse diffusion 과정을 그대로 적용. 예: OpenAI의 **Point-E**, **Shap-E** - 텍스트/이미지 조건으로 point cloud(또는 implicit function의 파라미터)를 직접 생성. 3D 학습 데이터가 이미지보다 훨씬 적다는 게 이 방식의 근본적인 제약이다.
2. **2D diffusion model을 재활용해서 3D를 만드는 방법 (distillation 계열)**: 3D 데이터를 직접 학습하는 대신, 이미 방대한 이미지로 학습된 강력한 2D diffusion model(예: Stable Diffusion)의 "지식"을 3D 표현(주로 NeRF)에 주입한다.
   - **Score Distillation Sampling (SDS)**: NeRF 하나를 최적화 대상으로 두고, 이 NeRF를 여러 임의의 각도에서 렌더링한 2D 이미지를 2D diffusion model에 넣어서 "이 이미지가 노이즈를 얼마나 제거해야 하는지"에 대한 gradient를 구한 뒤, 그 gradient로 (diffusion model 자체가 아니라) **NeRF의 파라미터**를 업데이트한다. 즉 2D diffusion model을 "이 렌더링이 얼마나 그럴듯한지 채점하는 감독관"으로만 활용하는 것.
   - 이 아이디어를 처음 제시한 게 **DreamFusion**(Google, 2022)이고, 이후 **Magic3D**, **Zero-1-to-3**(단일 이미지 조건으로 새로운 각도를 생성) 등 후속 연구가 이어졌다.
- 두 갈래의 트레이드오프: 1번은 3D 데이터가 부족해서 품질/다양성에 한계가 있고, 2번은 2D 모델의 지식을 재활용해서 품질은 좋지만 NeRF 하나당 최적화에 시간이 오래 걸리고(장면 하나 만드는 데 수십 분~시간 단위), 여러 각도에서 일관성이 깨지는(예: 얼굴이 여러 개 생기는 "Janus problem") 문제가 있다.

## 5. 이 노트가 앞의 CV 노트들과 이어지는 지점

- **U-Net** (`03-object-detection.md`) → diffusion model의 denoising network로 재사용
- **Transfer learning / pretrained model 활용** (`02-...architectures.md`) → SDS 방식 자체가 "거대한 pretrained 2D diffusion model의 지식을 다른 문제(3D 생성)에 우려먹는다"는 점에서 결이 같은 발상
- **Neural Style Transfer의 최적화 방식** (`04-face-recognition-and-style-transfer.md`) → "네트워크 가중치가 아니라 대상 자체(이미지, 또는 NeRF 파라미터)를 gradient descent로 최적화한다"는 점에서 SDS와 사고방식이 닮아있다.
- **CNN 기본기** (`01-cnn-foundations.md`) → GAN의 Discriminator, diffusion model의 U-Net 내부 모두 결국 convolution 기반 구조라, CNN을 이해하고 있어야 내부 구현을 읽을 수 있다.

정리하자면, 이 노트에서 다룬 GAN/diffusion/3D diffusion은 Andrew Ng 강의 이후에 나온 내용이지만, 뜯어보면 앞에서 배운 CNN/U-Net/transfer learning 같은 개념들을 그대로 재활용하고 있다는 걸 알 수 있다. 완전히 새로운 분야라기보다는 "이미 배운 블록들을 새로운 목적(생성)에 맞게 다시 조립한 것"에 가깝다.
