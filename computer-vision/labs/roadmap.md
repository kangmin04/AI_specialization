# CV 4시간 실습 로드맵 (Diffusion까지)

ML Specialization 수강 후 CV 랩실 진학 전에 **실습 위주로 빠르게 훑기** 위한 로드맵이다. 이론 노트(`computer-vision/*.md`)는 각 단계를 끝낸 뒤 "방금 돌린 게 뭐였는지" 확인하는 용도로만 읽는다.

환경: Apple M5 / 16GB / `torch` (MPS 사용 가능) / `torchvision` / `diffusers` 설치 완료.

---

## 핵심 3가지

### ① 배울 것: PyTorch 학습 루프 하나

```
이미지 → 텐서 (B, C, H, W) → model(x) → loss → loss.backward() → opt.step()
```

classification이든 detection이든 diffusion이든 CV 연구 코드는 전부 이 뼈대다. diffusion 학습도 이 루프에서 "이미지에 noise 섞고 → 모델이 그 noise를 맞히게 MSE"로 두 줄만 바뀐다. 랩실에서 하는 일은 결국 **남의 GitHub 코드에서 이 루프를 찾아서 한 부분을 바꿔 실험하는 것**이다.

ML Specialization과의 연결:
- `loss.backward()` = 손으로 계산했던 `dj_dw`
- `opt.step()` = `w = w - α·dj_dw`
- Keras `model.fit()`이 숨기던 걸 PyTorch에선 직접 쓴다

### ② 완전히 무시해도 되는 것

- **TensorFlow/Keras**: CV 연구 코드는 사실상 전부 PyTorch다.
- **노트 02~04의 대부분**: LeNet/AlexNet/VGG/Inception 구조 암기, YOLO anchor box 계산, R-CNN 계보, triplet loss. "ResNet = skip connection", "U-Net이라는 구조가 있다"만 알면 된다.
- **conv를 numpy로 직접 구현하기**: `nn.Conv2d` 한 줄이면 된다.
- **GAN 변종 목록, DDPM ELBO 수식 유도**: 코드 읽기나 랩실 면담에서 요구되지 않는다.

### ③ 한 번만 하면 앞서게 되는 연습

**MNIST로 작은 diffusion model을 처음부터 학습시키고, 한 가지를 바꿔서 결과를 비교하기** (= Step 3).

데이터 로딩, U-Net(CNN), 학습 루프, 샘플링이 전부 들어 있고, "하나 바꾸고 비교"가 연구의 실제 모양이다. 대부분은 Stable Diffusion을 돌려만 보지, 직접 학습시켜서 바꿔본 적은 없다.

---

## 전체 흐름

| 단계 | 주제 | 시간 | 산출물 |
|---|---|---|---|
| Step 1 | PyTorch 학습 루프 | ~30분 | `lab01-pytorch-loop.py` |
| Step 2 | Pretrained 모델 가져다 쓰기 (inference + fine-tuning) | ~50분 | `lab02-pretrained-finetune.py` |
| Step 3 | **Diffusion 직접 학습 + 한 가지 바꾸기 (메인)** | ~100분 | `lab03-mnist-diffusion.py` + 비교 이미지 |
| Step 4 | Stable Diffusion을 `diffusers`로 뜯어보기 | ~60분 | `lab04-stable-diffusion.py` |

진행 규칙: **각 단계 끝의 질문에 답해야 다음 단계로 넘어간다.** 코드는 복붙하지 말고 직접 타이핑한다.

---

## Step 1: PyTorch 학습 루프 (~30분)

**목표**: 텐서 shape `(B, C, H, W)`와 학습 루프 5줄(forward → loss → zero_grad → backward → step)을 손에 익힌다.

**할 일**: MNIST 숫자 분류 CNN을 MPS(맥 GPU)에서 2 epoch 학습하고 예측 결과를 눈으로 확인한다.

<details>
<summary>코드: <code>computer-vision/labs/lab01-pytorch-loop.py</code></summary>

```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else "cpu"  # 맥 GPU
root = os.path.join(os.path.dirname(__file__), "data")

# 1) 데이터: 이미지를 텐서로. ToTensor가 [0,255] -> [0,1], (H,W) -> (C,H,W)로 바꿔줌
tf = transforms.ToTensor()
train = datasets.MNIST(root, train=True, download=True, transform=tf)
test = datasets.MNIST(root, train=False, download=True, transform=tf)
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=512)

x, y = next(iter(train_loader))
print("x:", x.shape, "y:", y.shape)  # <- 이 출력 기억해두기

# 2) 모델: conv -> pool 두 번 하고 마지막에 dense
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 10),  # softmax는 loss 안에 들어있음 (from_logits=True랑 같은 얘기)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 3) 학습 루프 - 앞으로 모든 연구 코드에서 이 5줄을 찾게 됨
for epoch in range(2):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)          # forward
        loss = loss_fn(logits, y)  # cost
        opt.zero_grad()            # 이전 gradient 지우기
        loss.backward()            # dj_dw 계산
        opt.step()                 # w = w - lr * dj_dw

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {correct / len(test):.4f}")

# 4) 예측 결과 눈으로 확인
x, y = next(iter(test_loader))
with torch.no_grad():
    pred = model(x[:8].to(device)).argmax(1).cpu()
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(x[i, 0], cmap="gray")
    ax.set_title(f"pred {pred[i].item()}")
    ax.axis("off")
plt.show()
```

</details>

```bash
python3 computer-vision/labs/lab01-pytorch-loop.py
```

**통과 질문**
1. `x.shape` 출력값은 무엇이고, 네 숫자는 각각 무슨 뜻인가?
2. 최종 test acc는?
3. **실험**: `opt.zero_grad()`를 주석 처리하고 다시 돌린다. **돌리기 전에** 결과를 예측해 적고, 실제 결과와 그 이유를 적는다.

**끝나고 읽을 노트**: `01-cnn-foundations.md`에서 padding/stride/pooling 부분만 (위 코드의 `28 -> 14 -> 7`이 왜 그런지 확인).

---

## Step 2: Pretrained 모델 가져다 쓰기 (~50분)

**목표**: 연구의 90%는 모델을 처음부터 만들지 않고 **pretrained weight를 불러와 붙이거나 바꾸는 것**이다. 그 3가지 동작을 익힌다.

**할 일**
1. **Inference**: `torchvision`의 ImageNet pretrained ResNet-18을 불러와서 **내 폰으로 찍은 사진**을 분류한다. `weights.transforms()`(resize + normalize 전처리)를 왜 꼭 같이 써야 하는지 확인한다.
2. **Fine-tuning**: 마지막 층 `model.fc`를 내 클래스 수에 맞는 `nn.Linear`로 교체하고, backbone은 freeze(`requires_grad = False`)한 채 CIFAR-10 일부(train 5,000장 / test 1,000장)로 학습한다. Step 1의 루프를 **그대로 재사용**한다.
3. **비교 실험**: pretrained vs. 랜덤 초기화(`weights=None`)로 같은 epoch만큼 학습해서 acc를 비교한다.

**준비물**: 폰으로 찍은 사진 한 장을 `computer-vision/labs/my_photo.jpg`로 저장 (없으면 Part 1은 건너뜀). CIFAR-10(170MB)과 ResNet-18 weight(45MB)는 첫 실행 때 자동 다운로드된다. epoch당 약 10초.

<details>
<summary>코드: <code>computer-vision/labs/lab02-pretrained-finetune.py</code></summary>

```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from PIL import Image
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else "cpu"
here = os.path.dirname(__file__)
root = os.path.join(here, "data")

PRETRAINED = True  # 실험: False로 바꿔서 한 번 더 돌리기

# pretrained weight + 그 weight가 학습될 때 썼던 전처리를 세트로 가져옴
weights = models.ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()        # resize 256 -> center crop 224 -> [0,1] -> ImageNet mean/std로 normalize
categories = weights.meta["categories"]  # ImageNet 1000개 클래스 이름
print(preprocess)

# ---------------- Part 1: 내 사진 분류 (inference만, 학습 없음) ----------------
img_path = os.path.join(here, "my_photo.jpg")
if os.path.exists(img_path):
    model = models.resnet18(weights=weights).to(device).eval()
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (3,224,224) -> (1,3,224,224): 배치 차원 추가
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top5 = probs.topk(5)
    for p, i in zip(top5.values, top5.indices):
        print(f"{categories[i.item()]:>25s}  {p.item():.3f}")
    plt.imshow(img)
    plt.title(categories[top5.indices[0].item()])
    plt.axis("off")
    plt.show()

# ---------------- Part 2: fine-tuning (마지막 층만 새로 학습) ----------------
train = Subset(datasets.CIFAR10(root, train=True, download=True, transform=preprocess), range(5000))
test = Subset(datasets.CIFAR10(root, train=False, download=True, transform=preprocess), range(1000))
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=256)

model = models.resnet18(weights=weights if PRETRAINED else None)
for p in model.parameters():
    p.requires_grad = False                         # backbone freeze
model.fc = nn.Linear(model.fc.in_features, 10)     # head 교체: 1000클래스 -> 10클래스 (새 층이라 학습됨)
model = model.to(device)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"학습되는 파라미터 {n_train:,} / 전체 {n_total:,}")

opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)  # optimizer에도 fc만 넘김
loss_fn = nn.CrossEntropyLoss()

# Step 1이랑 같은 루프. 단 train() 대신 eval(): BatchNorm 통계까지 고정해서 backbone을 진짜로 얼리기 위함
for epoch in range(3):
    model.eval()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {correct / len(test):.3f}")
```

</details>

```bash
python3 computer-vision/labs/lab02-pretrained-finetune.py
```

**통과 질문**
1. 내 사진의 top-5 결과는? 틀렸다면 왜 틀렸을까? (힌트: `categories`에 그 물체가 있나?)
2. "학습되는 파라미터 / 전체" 출력값은? 전체의 몇 %만 학습한 건가?
3. **실험**: `PRETRAINED = False`로 바꿔서 다시 돌린다. **돌리기 전에** acc를 예측해 적고, 실제 결과와 그 이유를 적는다.

**핵심 포인트**: `model.fc = nn.Linear(512, N)` 한 줄이 transfer learning의 전부다. 나중에 연구 코드에서 `backbone`, `head`, `freeze`라는 단어가 나오면 이 구조다.

**끝나고 읽을 노트**: `02-classic-and-modern-architectures.md`에서 ResNet(residual block)과 transfer learning 부분만.

---

## Step 3: Diffusion 직접 학습 + 한 가지 바꾸기 (~100분, 메인)

**목표**: diffusion의 학습 루프가 Step 1 루프와 사실상 같다는 걸 직접 확인하고, **하나를 바꿔 결과를 비교하는 연구 실험**을 한 번 해본다.

**할 일**
1. **학습**: `diffusers`의 `UNet2DModel`(작게 설정) + `DDPMScheduler`로 MNIST 생성 모델을 학습한다. 루프에서 바뀌는 건 이 부분뿐이다.
   ```
   noise = randn_like(x0)
   t = 랜덤 timestep
   x_t = scheduler.add_noise(x0, noise, t)   # 이미지에 noise 섞기
   loss = MSE(model(x_t, t), noise)          # 섞은 noise를 맞히기
   ```
2. **샘플링**: 순수 noise에서 시작해 `scheduler.step()`을 반복해서 숫자 이미지를 생성하고, 중간 단계(t = 1000 → 0)를 한 줄로 시각화한다.
3. **한 가지 바꾸기 (택1)**: 바꾼 것 외에는 모두 고정하고 결과 이미지를 나란히 비교한다.
   - **Sampler 교체**: DDPM 1000 step vs. DDIM 50 step (속도와 품질 비교)
   - **Noise schedule 교체**: `linear` vs. `squaredcos_cap_v2`
   - **Class conditioning + classifier-free guidance**: "7을 그려라"가 되게 만들고 guidance scale을 바꿔본다

**산출물**: 비교 이미지 1장 + "무엇을 바꿨고 → 무엇이 달라졌고 → 왜 그런 것 같은지" 3줄. 이게 연구 미팅에서 보여주는 것과 같은 형식이다.

**끝나고 읽을 노트**: `05-generative-models-beyond-the-course.md`의 diffusion 파트(forward/reverse process, U-Net backbone, classifier-free guidance). 이 시점엔 수식이 "아까 코드의 그 줄"로 읽힌다.

---

## Step 4: Stable Diffusion을 `diffusers`로 뜯어보기 (~60분)

**목표**: 실제 연구에서 쓰는 대형 diffusion model이 Step 3와 **같은 구조에 부품만 더 붙은 것**임을 확인하고, 남의 코드를 읽는 감을 잡는다.

**할 일**
1. **돌리기**: `StableDiffusionPipeline`(SD 1.5, fp16, `mps`)으로 텍스트에서 이미지를 생성한다.
2. **분해**: `pipe.vae`, `pipe.unet`, `pipe.text_encoder`, `pipe.scheduler`를 하나씩 출력해 본다. Step 3와 비교하면 새로 붙은 부품은 두 개다.
   - VAE: 512×512 이미지를 64×64 latent로 압축한다 (latent diffusion).
   - Text encoder: 프롬프트를 조건으로 넣는다 (Step 3의 class conditioning이 텍스트로 확장된 것).
3. **조작 실험**: `guidance_scale`, `num_inference_steps`, scheduler 교체, img2img의 `strength`를 바꿔 비교한다.
4. **코드 읽기**: `diffusers` 소스에서 pipeline의 `__call__`을 열어 **denoising loop를 찾는다**. Step 3에서 직접 쓴 샘플링 루프와 같은 모양인지 확인한다. 앞으로 논문 GitHub를 볼 때도 먼저 train loop와 sampling loop부터 찾으면 된다.

**끝나고 읽을 노트**: `05-generative-models-beyond-the-course.md`의 latent diffusion / Stable Diffusion 파트. 관심 있으면 3D diffusion 파트까지.

---

## 4시간 후 할 수 있게 되는 것

- 아무 CV 논문의 GitHub를 열어서 데이터 로더, 모델, 학습 루프, 샘플링 루프를 찾을 수 있다.
- pretrained 모델을 불러와 내 데이터에 맞게 바꿀 수 있다.
- diffusion model을 직접 학습시켜 봤고, 하나를 바꿔 비교한 결과를 3줄로 설명할 수 있다.
- 랩실 면담에서 "diffusion 해봤어요"를 **돌려본 수준이 아니라 학습시키고 바꿔본 수준**으로 말할 수 있다.
