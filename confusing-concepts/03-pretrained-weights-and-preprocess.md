# Pretrained weights 객체와 전처리(preprocess)

## 헷갈렸던 점

> `weights = models.ResNet18_Weights.IMAGENET1K_V1`은 미리 학습된 가중치를 가져오는 것 같은데, 그 외에 뭘 더 가져오는 건가?
> `weights.transforms()`, `weights.meta["categories"]`는 뭔가?
> 전처리의 Resize 256 → CenterCrop 224는 왜 하는 건가? 224 × 224 center crop이 뭔가?
> ToTensor는 출력에 안 보이는데 내부적으로 해 주는 건가?
> ToTensor가 0~255를 0~1로 바꾸는 거면 그게 이미 정규화 아닌가? mean, std로 normalize하는 거랑 뭐가 다른가?

## 결론

- `IMAGENET1K_V1`은 가중치 자체가 아니라 **"가중치 + 그 가중치를 만들 때 쓴 설정"을 묶은 명세서**다. 이 줄에서는 다운로드가 일어나지 않는다. 실제 다운로드는 `models.resnet18(weights=weights)`를 호출할 때 일어난다.
- `weights.transforms()`는 **학습 때 썼던 전처리**다. 모델은 그 전처리를 거친 입력만 봤기 때문에 반드시 같이 써야 한다.
- `weights.meta["categories"]`는 **출력 index → 클래스 이름** 표다. 모델은 숫자 1000개만 내놓는다.
- ToTensor는 내부에 들어 있다. `pil_to_tensor` + `convert_image_dtype` 두 줄로 나뉘어 있어서 이름이 안 보일 뿐이다.
- **÷255와 Normalize는 다른 작업이다.** ÷255는 단위만 0~1로 바꾸는 것(값은 전부 양수, 평균 약 0.5)이고, Normalize는 데이터 통계로 평균 0, 표준편차 1을 만드는 z-score 정규화다.

## 1. weights 객체에 들어 있는 것

```python
weights = models.ResNet18_Weights.IMAGENET1K_V1
```

| 속성 | 실제 값 | 뜻 |
|---|---|---|
| `weights.url` | `https://download.pytorch.org/models/resnet18-f37072fd.pth` | 실제 가중치 파일 주소 (약 45MB) |
| `weights.transforms()` | `ImageClassification(crop_size=[224], resize_size=[256], mean=..., std=...)` | 학습 때 쓴 전처리 |
| `weights.meta["categories"]` | 문자열 1000개 리스트 | 출력 index → 클래스 이름 |
| `weights.meta["_metrics"]` | `acc@1 69.758, acc@5 89.078` | ImageNet 검증 성능 |
| 그 외 `meta` | `num_params`, `min_size`, `recipe`, `_ops`, `_file_size` | 파라미터 수, 최소 입력 크기, 학습 레시피 등 |

```
ResNet18_Weights.IMAGENET1K_V1  (명세서, 이 시점엔 다운로드 없음)
 ├─ url            → models.resnet18(weights=...) 할 때 다운로드
 │                   (~/.cache/torch/hub/checkpoints/ 에 저장, 두 번째부터는 캐시 사용)
 ├─ transforms()   → 학습 때와 같은 입력 분포를 만드는 전처리
 └─ meta
     ├─ categories → 출력 index ↔ 클래스 이름
     ├─ _metrics   → 이 가중치의 성능
     └─ num_params, min_size, recipe ...
```

- `ResNet18_Weights.DEFAULT is IMAGENET1K_V1` → `True`. ResNet-18은 V1밖에 없다.
- ResNet-50처럼 V2가 있는 모델은 DEFAULT가 V2이고, **전처리도 버전마다 다르다.** 그래서 가중치와 전처리를 세트로 받는다.

### categories

```python
categories[0]    # 'tench' (물고기)
categories[1]    # 'goldfish'
categories[281]  # 'tabby' (얼룩 고양이)
```

모델 출력의 281번째 점수가 가장 크면 → "tabby"라고 읽는다. 사진 속 물체가 이 1000개 안에 없으면 모델은 절대 정답을 낼 수 없고, 목록 중 가장 비슷한 것을 고를 뿐이다.

## 2. 전처리 내부 코드

`print(preprocess)`에는 설정값만 보이지만 실제 코드는 이렇다.

```python
def forward(self, img):
    img = F.resize(img, 256)                       # ① Resize
    img = F.center_crop(img, 224)                  # ② CenterCrop
    if not isinstance(img, Tensor):
        img = F.pil_to_tensor(img)                 # ③-a PIL → uint8 텐서 (C,H,W)
    img = F.convert_image_dtype(img, torch.float)  # ③-b 0~255 → 0~1 (÷255)
    img = F.normalize(img, mean, std)              # ④ Normalize
    return img
```

③-a + ③-b = `ToTensor`.

### 내 사진(`my_photo.jpg`)이 단계별로 바뀐 결과

| 단계 | 결과 |
|---|---|
| 원본 | 4032 × 3024 (W × H), PIL 이미지 |
| ① Resize 256 | 341 × 256 |
| ② CenterCrop 224 | 224 × 224 |
| ③ ToTensor | `(3, 224, 224)`, 범위 0.0 ~ 1.0, 채널 평균 약 `[0.57, 0.53, 0.47]` |
| ④ Normalize | 범위 -2.05 ~ 2.43, 채널 평균 약 `[0.36, 0.33, 0.30]` |

## 3. ① Resize 256: 왜 크기를 맞추나

- **배치를 만들려면 크기가 같아야 한다.** `(B, C, H, W)` 하나로 쌓으려면 H, W가 전부 같아야 한다.
- **모델이 학습한 "물체 크기"에 맞춘다.** ResNet은 약 224 크기로 ImageNet을 봤다. 4032 크기 사진을 그대로 넣으면 고양이 귀 하나가 수백 픽셀이 되어 본 적 없는 스케일이 된다.
- **짧은 변 기준**으로 맞추는 이유: 4032 × 3024를 억지로 256 × 256으로 만들면 찌그러진다. 짧은 변을 256으로, 긴 변은 같은 비율로 줄이면 모양이 유지된다. 대신 341 × 256처럼 정사각형이 아니게 되는데, 이건 다음 단계에서 해결한다.

## 4. ② CenterCrop 224: 가운데만 잘라내기

```
        341
  ┌───┬────────┬───┐
  │   │        │   │
  │버림│ 224×224│버림│ 256
  │   │ (사용) │   │
  └───┴────────┴───┘
   위아래 16px, 좌우 약 58px씩 잘려 나감
```

**center crop = 이미지 정중앙 기준으로 224 × 224 정사각형만 남기고 가장자리를 버리는 것.**

바로 224로 줄이지 않고 256 → 224로 자르는 이유:
- 학습 때는 `RandomResizedCrop(224)`를 썼다. 사진의 **랜덤한 일부**를 잘라 224로 만드는 data augmentation이라, 모델은 "물체가 프레임을 꽉 채운" 이미지에 익숙하다.
- 테스트 때는 랜덤성이 있으면 안 되니까 고정된 방식으로 흉내 낸다. 조금 크게(256) 만든 뒤 가운데를 잘라 테두리 약 12%(224/256 = 0.875)를 버리면 물체가 화면을 더 채운다.
- 정사각형이 되니 배치로 쌓을 수도 있다.

부작용: 물체가 사진 가장자리에 있으면 잘려 나간다. 예측이 이상하면 crop된 이미지를 직접 `plt.imshow`로 확인해 본다.

## 5. ③ ToTensor: 형식 변환

- PIL 이미지는 `(H, W, C)` 순서의 0~255 정수. PyTorch conv는 `(C, H, W)` 순서의 float를 받는다. 그래서 축 순서와 dtype을 바꾼다.
- ÷255는 **단위를 0~1로 바꾸는 것뿐**이다. 다음 단계의 mean/std(0.485 등)가 0~1 단위로 계산된 값이라서 먼저 단위를 맞춘다.

## 6. ④ Normalize: ÷255와 뭐가 다른가

ML 코스의 두 가지 feature scaling과 같다.

| | ToTensor (÷255) | Normalize ((x - mean) / std) |
|---|---|---|
| 종류 | 고정 상수로 나누는 스케일링 | **z-score normalization** |
| 데이터 통계 사용 | X (0, 255는 그냥 픽셀 규격) | O (ImageNet 전체의 채널별 평균, 표준편차) |
| 결과 범위 | 0 ~ 1, **전부 양수** | 대략 -2 ~ +2.5, **0 중심** |
| 결과 평균 | 약 0.5 | ImageNet 전체 기준 0, 표준편차 1 |

÷255는 범위만 줄이고 분포 중심은 0.5 근처 그대로다. Normalize는 그 분포를 **0 중심, 퍼짐 1**로 옮긴다. 실측에서도 ÷255 후 min/max = 0, 1이었고 normalize 후 -2.05, 2.43이 됐다.

Normalize를 하는 이유:
1. **학습 분포와 맞추기 (가장 중요).** ResNet은 normalize된 입력으로만 학습했다. 이 단계를 빼면 모든 값이 평균적으로 약 +2만큼 밀려 들어가 예측이 망가진다. ML 코스의 "train에서 구한 mean/std를 test에도 똑같이 적용하라"와 같은 원리.
2. **학습이 잘 되게.** 입력이 전부 양수면 첫 층 가중치의 gradient 부호가 한쪽으로 쏠려 경사하강이 지그재그로 움직인다. 0 중심이면 수렴이 빠르다 (cost 등고선이 동그래지는 것과 같은 효과).
3. **채널별로 따로.** ImageNet은 R 평균 0.485, B 평균 0.406처럼 채널마다 통계가 달라서 mean/std가 채널별로 3개씩 있다.

> 내 사진 한 장의 normalize 후 평균이 0이 아니라 0.3 정도인 건 정상이다. 이 사진이 ImageNet 평균보다 조금 밝다는 뜻이고, 평균 0 / 표준편차 1은 **데이터셋 전체**에 대한 통계다.

### CIFAR-10(32 × 32)에도 이 전처리를 쓰는 이유

backbone이 ImageNet 방식 입력(224, ImageNet mean/std)으로 학습됐으니 새 데이터도 그 형태로 맞춰 넣어야 한다. 32 → 256으로 크게 확대해서 흐려지지만, 크기와 분포를 맞추는 게 더 중요하다.

## 기억할 것

- pretrained 모델은 **가중치 + 전처리 + 클래스 이름표**가 한 세트다. 가중치만 가져오고 전처리를 다르게 하면 안 된다.
- Resize(짧은 변) → CenterCrop → ToTensor → Normalize. 앞의 둘은 **크기**, 뒤의 둘은 **값**을 맞추는 단계다.
- ÷255 = 단위 변환, Normalize = z-score 정규화. 서로 다른 일이다.
- 전처리 결과가 궁금하면 단계별로 찍어 본다:

```python
import torchvision.transforms.functional as F
a = F.resize(img, 256);         print(a.size)
b = F.center_crop(a, 224);      print(b.size)
c = F.pil_to_tensor(b);         print(c.shape, c.dtype)
d = F.convert_image_dtype(c, torch.float); print(d.min(), d.max())
e = F.normalize(d, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); print(e.min(), e.max())
```
