# Object Detection

컴퓨터 비전에서 다루는 문제들을 난이도 순으로 나열해보면 대략 이렇게 된다.

1. **Image Classification** — 이미지 하나에 "이게 뭐다"라는 라벨 하나만 붙이면 되는 문제. 예를 들어 "이 사진엔 자동차가 있다" 정도.
2. **Classification with Localization** — 거기서 한 발 더 나가서, "자동차가 있다"뿐만 아니라 "그 자동차가 이미지의 어느 위치에 있는지"까지 bounding box로 짚어내는 문제. 근데 이건 보통 이미지 안에 물체가 딱 하나 있다고 가정하고 푼다.
3. **Object Detection** — 이미지 안에 물체가 몇 개가 있을지 모르는 상태에서, 여러 종류/여러 개의 물체를 각각 bounding box와 함께 찾아내는 문제. classification with localization을 일반화한 버전이라고 보면 된다.

이 노트는 이 흐름을 따라서 1 -> 2 -> 3으로 확장해가는 과정을 정리하고, 그 다음에 실제로 detection을 어떻게 구현하는지(sliding window -> YOLO -> region proposal 계열), 마지막으로 pixel 단위로 더 정밀하게 들어가는 semantic segmentation까지 다룬다.

## 1. Object Localization

일반 classification 네트워크는 이미지를 넣으면 softmax를 통과해서 class probability 벡터 하나만 뱉는다. 근데 localization까지 하려면 "어디에 있는지"에 대한 정보도 같이 뱉어야 한다. 그래서 네트워크의 출력 자체를 확장한다.

물체 위치는 보통 bounding box로 표현하는데, 이 bounding box는 4개의 숫자로 정의된다.

- `bx, by`: bounding box의 중심 좌표 (이미지 크기에 대해 0~1로 정규화)
- `bh, bw`: bounding box의 height, width (역시 정규화된 비율)

여기에 이 물체가 애초에 존재하는지 여부를 나타내는 `pc` (probability of class / "이미지 안에 물체가 있긴 한가?")와, 존재한다면 어떤 클래스인지 나타내는 class probability들을 합쳐서 출력 벡터를 구성한다. 예를 들어 클래스가 보행자/자동차/오토바이 3종류라고 하면 출력 벡터는 이렇게 생긴다.

```
y = [ pc, bx, by, bh, bw, c1, c2, c3 ]
```

- `pc = 1`이면: 물체가 있다는 뜻이고, 이어지는 `bx, by, bh, bw`가 그 물체의 bounding box, `c1, c2, c3`이 각 클래스일 확률.
- `pc = 0`이면: 배경(물체 없음)이라는 뜻이고, 이 경우엔 나머지 값들(`bx, by, bh, bw, c1, c2, c3`)이 뭐든 상관없다 — 어차피 loss 계산할 때 무시해버릴 값들이기 때문.

### Loss function

이 벡터가 여러 종류의 정보(존재 여부, 좌표, 클래스)를 섞어 담고 있다 보니 loss도 케이스를 나눠서 설계한다. 강의에서 나온 단순화된 버전은 대략 이런 식이다.

- `y1 (pc) = 1`인 경우 (물체가 실제로 있는 경우): 전체 벡터에 대해 loss를 계산한다. 보통 좌표들(`bx,by,bh,bw`)은 squared error, `pc`는 (구현에 따라) logistic loss나 squared error, class probability들은 log-likelihood(softmax loss) 이런 식으로 각 항목 성격에 맞게 다른 loss를 조합해서 쓸 수도 있다.
- `y1 (pc) = 0`인 경우 (배경인 경우): 어차피 나머지 값들이 의미 없기 때문에 `pc`에 대한 loss만 계산하고 좌표/클래스 부분은 무시한다.

핵심은 "출력 벡터가 여러 의미를 담고 있으니, loss도 각 부분에 맞게 따로 설계하고 필요 없는 부분은 masking 해준다"는 아이디어다.

## 2. Landmark Detection

bounding box처럼 사각형(4개 숫자)만 예측하는 게 아니라, 아예 이미지 위의 특정 좌표점들 자체를 여러 개 예측하게 만들 수도 있다. 이게 landmark detection이다.

예를 들면:
- 얼굴 인식에서 눈꼬리, 입꼬리, 코 끝 같은 얼굴 특징점(facial landmark) 좌표들을 예측 (스냅챗/인스타 필터가 얼굴에 딱 맞게 씌워지는 게 이 기술 덕분)
- 사람 자세 추정(pose estimation)에서 어깨, 팔꿈치, 무릎 같은 관절 위치들을 예측

구현 아이디어 자체는 object localization과 다르지 않다. 그냥 네트워크 출력에 `(l1x, l1y), (l2x, l2y), ..., (lnx, lny)` 이런 식으로 좌표 쌍들을 잔뜩 추가하고, 이 좌표들을 regression으로 학습시키면 된다. 결국 "좌표를 직접 회귀로 예측한다"는 게 핵심 아이디어이고, 이 아이디어가 bounding box (2개 좌표로 표현 가능한 사각형)에도, landmark(임의 개수의 점)에도 똑같이 적용되는 것.

단, 학습 데이터를 만들 때 각 landmark의 라벨링 순서/의미가 이미지마다 일관되어야 한다 (예: `l1`은 항상 왼쪽 눈꼬리, `l2`는 항상 오른쪽 눈꼬리 이런 식으로).

## 3. Object Detection - Sliding Windows

이제 이미지 안에 물체가 몇 개 있을지 모르는 진짜 detection 문제로 넘어간다. 가장 원초적인 아이디어가 sliding window다.

### 기본 아이디어와 문제점

1. 작은 크기의 window(정사각형 crop)를 하나 정한다.
2. 이 window를 이미지 위에서 일정 간격(stride)으로 좌상단부터 우하단까지 쭉 이동시키면서, 매번 그 window 안의 이미지 조각을 crop해서 classification 네트워크(예: "이 안에 자동차가 있나 없나")에 통과시킨다.
3 물체가 있다고 판단된 window의 위치가 곧 detection 결과가 된다.

근데 이 방식은 문제가 두 가지 있다.

- **연산 비용**: window를 촘촘하게(stride를 작게) 움직일수록 정확도는 올라가지만, crop한 이미지 조각 하나하나를 전부 독립적으로 CNN에 통과시켜야 하니 연산량이 엄청나게 늘어난다. 반대로 stride를 크게 하면 연산은 줄지만 물체를 놓치기 쉽다.
- **여러 window 크기**: 이미지 안의 물체는 크기가 다양하다. 작은 window 하나로만 훑으면 큰 물체를 놓치고, 큰 window로만 훑으면 작은 물체를 놓친다. 그래서 여러 크기의 window로 각각 다 슬라이딩을 해야 하는데, 이러면 연산량이 또 배로 늘어난다.

즉 sliding window는 아이디어는 직관적인데 컴퓨팅 비용 때문에 그대로 쓰기는 힘들다는 게 문제.

### Convolutional implementation of sliding windows

이 문제를 해결하는 핵심 트릭이 "FC layer를 1x1 convolution으로 바꾸기"다 (OverFeat 논문에서 나온 아이디어).

일반적인 classification 네트워크는 conv layer들을 거친 다음 마지막에 Fully Connected(FC) layer로 flatten해서 최종 출력을 낸다. 근데 이 FC layer를 수학적으로 동일한 역할을 하는 conv layer로 바꿀 수 있다.

- 예를 들어 conv layer의 마지막 출력이 `5x5x16`이고, 그 다음 FC layer가 400개의 유닛을 갖는다고 하면, 이건 `5x5x16` 볼륨 전체를 커버하는 `5x5x16` 크기의 필터 400개짜리 conv layer와 동일하다 (출력이 `1x1x400`이 됨).
- 마찬가지로 그 다음 FC layer들도 전부 `1x1` conv layer로 바꿀 수 있다.

이렇게 네트워크 전체를 fully convolutional하게 만들어놓으면, 원래 sliding window로 여러 번 나눠서 넣어야 했던 여러 개의 crop을 **한 번에 큰 이미지 전체를 통째로 네트워크에 넣는 것**으로 대체할 수 있다. 왜냐하면 conv 연산 자체가 원래 이미지 전체에 대해 필터를 슬라이딩하면서 계산하는 방식이기 때문에, sliding window에서 각 window마다 따로 하던 계산이 이 안에서 자연스럽게 공유되기 때문이다 (겹치는 영역의 연산을 중복해서 하지 않아도 됨).

- 원래 방식: 입력 이미지를 window 크기로 잘라서 하나씩 conv net에 통과 -> N번 forward pass
- Convolutional 방식: 큰 이미지 전체를 한 번 conv net에 통과 -> 출력이 `(가로칸수) x (세로칸수) x (클래스+정보)` 형태의 볼륨으로 나오고, 이 볼륨의 각 칸(cell)이 원래 sliding window 방식에서 해당 위치에 window를 놓았을 때의 결과와 대응됨 -> 1번 forward pass

즉 "여러 번 반복 계산" -> "한 번의 큰 계산으로 겹치는 부분을 재사용"으로 바꾼 것. 다만 이 방식만으로는 bounding box 위치가 정확히 grid 칸에 딱 맞아떨어지지 않는 문제가 남는데, 이건 다음 YOLO에서 더 정교하게 해결한다.

## 4. Bounding Box Predictions - YOLO 알고리즘

YOLO(You Only Look Once)는 "이미지를 grid로 나누고, 각 grid cell이 자기 위치를 기준으로 물체를 예측하게 하자"는 아이디어다.

### Grid 기반 예측

이미지를 예를 들어 `3x3` grid로 나눈다고 하자 (실제 논문/구현에서는 훨씬 잘게, `19x19` 같은 크기를 쓰기도 한다). 각 grid cell마다, 앞서 object localization에서 만든 것과 똑같은 형태의 출력 벡터를 하나씩 예측하게 한다.

```
y (한 cell당) = [ pc, bx, by, bh, bw, c1, c2, c3 ]
```

물체의 라벨링 규칙은: **물체의 bounding box 중심점이 어느 grid cell 안에 들어가느냐**에 따라, 그 물체는 그 cell의 담당이 된다 (물체가 여러 cell에 걸쳐 있어도 중심점 기준으로 딱 하나의 cell에만 할당). 그래서 `3x3` grid이고 클래스가 3개면 전체 target 출력은 `3 x 3 x 8` 텐서가 되고, 이걸 conv net으로 한 번에 예측한다 (여기서 각 cell의 `bx, by`는 그 cell 내부 기준 상대 좌표로 정규화, `bh, bw`는 전체 이미지 기준 비율로 표현하는 식으로 인코딩한다).

이 방식이 좋은 이유는 앞서 나온 convolutional sliding window와 똑같이, 이미지 전체를 한 번의 forward pass로 처리하면서 동시에 각 grid cell 위치에 딱 맞는 정확한 bounding box 좌표까지 회귀로 예측한다는 점이다 (sliding window처럼 window 크기에 bounding box가 갇히지 않음).

### Intersection over Union (IoU)

예측한 bounding box가 실제(ground truth) bounding box와 얼마나 겹치는지를 정량적으로 재는 지표가 IoU다.

```
IoU = (두 box의 겹치는 영역 넓이) / (두 box를 합친 영역 넓이)
```

- IoU 값은 0~1 사이.
- 두 box가 완전히 겹치면 IoU = 1, 전혀 안 겹치면 IoU = 0.
- 보통 "IoU >= 0.5면 맞은 걸로 친다"는 threshold를 관례적으로 많이 쓴다 (더 엄격하게 보려면 0.6, 0.7도 사용).

IoU는 크게 두 군데서 쓰인다.
1. **평가 지표**: 예측한 detection이 ground truth와 얼마나 정확히 맞았는지 채점할 때
2. **Non-max suppression 내부 로직**: 바로 아래에서 설명

### Non-max suppression (NMS)

Grid 방식으로 예측하면 문제가 하나 생긴다. 실제로는 물체 하나에 대해 여러 grid cell이 "나도 이 물체를 봤다"면서 겹치는 bounding box를 여러 개 예측해버리는 경우가 흔하다. 그래서 같은 물체에 대해 중복으로 잡힌 box들을 정리해서 하나만 남기는 후처리가 필요한데, 이게 non-max suppression이다.

동작 과정을 단계별로 정리하면:

1. `pc` (혹은 `pc * class probability`, 즉 "이 box에 물체가 있고 이 클래스일 확률")가 threshold보다 낮은 box들은 먼저 다 버린다.
2. 남은 box들 중에서 `pc`가 가장 높은 box를 하나 고르고, 이걸 최종 예측(output)으로 채택한다.
3. 채택된 box와 IoU가 일정 threshold(예: 0.5) 이상인 나머지 box들은 "같은 물체를 가리키는 중복 box"로 간주하고 전부 제거한다.
4. 남은 box들 중에서 다시 `pc`가 가장 높은 걸 골라 2~3번을 반복한다.
5. 더 이상 box가 안 남을 때까지 반복.

클래스가 여러 개일 때는 이 과정을 클래스별로 각각 독립적으로 수행한다 (자동차용 box들끼리 NMS, 보행자용 box들끼리 NMS, 이런 식).

### Anchor boxes

Grid 방식의 또 다른 문제: 한 grid cell의 중심에 서로 다른 모양의 물체 두 개가 동시에 겹쳐 있으면 (예: 사람과 자동차가 겹쳐서 두 물체의 중심이 같은 cell에 들어가는 경우), 원래 방식대로면 그 cell은 출력 벡터를 하나만 낼 수 있으니까 둘 중 하나밖에 표현을 못 한다.

이걸 해결하는 게 anchor box다. 미리 몇 가지 대표적인 모양(세로로 긴 박스, 가로로 긴 박스 등)을 anchor box로 정해놓고, 각 grid cell이 anchor box 개수만큼의 출력 벡터를 예측하게 만든다.

예를 들어 anchor box를 2개(세로로 긴 모양 anchor 1 = 사람 형태, 가로로 넓은 모양 anchor 2 = 자동차 형태) 쓴다면, 각 cell의 출력은 이렇게 두 배로 늘어난다.

```
y (한 cell당) = [ pc, bx, by, bh, bw, c1, c2, c3,   <- anchor box 1
                  pc, bx, by, bh, bw, c1, c2, c3 ]  <- anchor box 2
```

라벨링 규칙도 조금 바뀐다: 물체는 "(그 물체의 중심이 속한 grid cell, 그 물체의 실제 shape과 IoU가 가장 높은 anchor box)" 조합에 할당된다. 즉 (grid cell, anchor box) 쌍이 물체를 담당하는 단위가 되는 것. 위 예시라면 사람은 anchor 1 슬롯에, 자동차는 anchor 2 슬롯에 인코딩되어서 한 cell 안에서도 서로 다른 슬롯에 두 물체를 표현할 수 있게 된다.

물론 한 cell + 한 anchor에 물체가 3개 이상 겹치는 경우처럼 anchor box로도 못 푸는 극단적인 경우는 여전히 남지만, 대부분의 실질적인 경우는 이걸로 커버가 된다. Anchor box 모양들은 보통 학습 데이터셋에 있는 bounding box들의 모양을 k-means 같은 걸로 클러스터링해서 대표 모양 몇 개를 뽑아 미리 정해놓는다 (YOLOv2 이후에 이런 방식이 도입됨).

### YOLO 전체 구조 정리

지금까지 나온 조각들을 다 합치면 YOLO의 전체 파이프라인이 된다.

| 단계 | 내용 |
|---|---|
| 1. 입력 | 이미지 전체 (예: 448x448) |
| 2. Grid 분할 | 예: 19x19 grid (각 cell이 자기 영역을 담당) |
| 3. Anchor box | 예: cell당 anchor box 5개 (미리 정의된 대표 모양들) |
| 4. Forward pass | Fully convolutional network 한 번 통과 -> 출력 텐서: `grid_h x grid_w x anchor개수 x (5 + class개수)` |
| 5. Threshold 적용 | `pc`가 낮은 예측들 제거 |
| 6. Non-max suppression | 클래스별로 겹치는 box들 정리해서 최종 detection만 남김 |

핵심 요약:
- **v1**: grid cell 단위로 bounding box를 직접 회귀로 예측한다는 원조 아이디어 (anchor box 개념 없이, 물체 하나당 cell 하나가 담당)
- **v2 (YOLO9000)**: anchor box 도입 (k-means로 anchor 모양 결정), batch normalization 등 학습 기법 개선으로 정확도/속도 향상
- **v3**: 여러 스케일(다른 크기의 feature map)에서 각각 detection을 수행해서 크고 작은 물체를 더 잘 잡아내도록 개선 (multi-scale prediction), backbone도 더 깊어짐(Darknet-53)

버전이 올라갈수록 디테일(loss 구성, backbone 구조, 학습 트릭 등)이 계속 추가되지만, "이미지를 grid로 나누고 -> 각 cell(및 anchor)이 bounding box와 class를 동시에 회귀/분류로 예측하고 -> NMS로 정리한다"는 핵심 뼈대는 전 버전에서 공통이다. 이 뼈대 덕분에 YOLO는 sliding window와 다르게 이미지 전체를 **한 번만** 보고("You Only Look Once") 모든 물체를 동시에 찾아낼 수 있어서 속도가 훨씬 빠르다.

## 5. Region Proposal 계열 (R-CNN, Fast R-CNN, Faster R-CNN)

YOLO 계열이 "grid 전체를 다 훑으면서 예측"하는 방식이라면, region proposal 계열은 "물체가 있을 법한 후보 영역을 먼저 골라내고, 그 후보 영역들에 대해서만 classification을 한다"는 접근이다.

### R-CNN (Regions with CNN features)

1. Segmentation 알고리즘(예: selective search) 같은 걸 이미지에 돌려서, 물체가 있을 것 같은 후보 영역(region proposal)을 이미지당 대략 2000개 정도 뽑아낸다.
2. 뽑은 각 region을 crop해서 CNN에 하나씩 통과시켜 classification (이 region이 어떤 클래스인지)을 수행한다.
3. Bounding box도 이 단계에서 좀 더 정교하게 다듬는다(regression으로 보정).

문제는 **느리다**는 것. Sliding window처럼 2000개나 되는 region 각각을 독립적으로 CNN에 통과시켜야 하니, region 사이에 겹치는 연산을 전혀 공유하지 못하고, 한 이미지 처리하는 데 시간이 꽤 걸린다 (거의 실시간과는 거리가 멀다).

### Fast R-CNN

R-CNN의 "각 region마다 CNN을 통째로 다시 돌린다"는 비효율을 개선한 버전.

- 이미지 전체를 **한 번만** CNN에 통과시켜서 feature map을 만들어놓는다.
- Region proposal은 여전히 (R-CNN처럼) selective search 등으로 미리 뽑아두지만, 이 region들을 원본 이미지가 아니라 **이미 계산해둔 feature map 위에 투영**해서 그 부분만 잘라 쓴다.
- 이때 region마다 크기가 다 다르니까, 이걸 고정된 크기로 맞춰주는 **RoI Pooling(Region of Interest Pooling)** 레이어를 사용한다 — feature map에서 region에 해당하는 영역을 뽑아서 고정 크기 grid로 pooling해주는 것.
- 이렇게 고정 크기로 맞춘 feature를 FC layer에 통과시켜서 classification + bounding box regression을 한 번에 처리.

즉 conv 연산을 이미지당 한 번만 하고 그 결과를 region들이 공유한다는 게 핵심 개선점. 다만 여전히 region proposal 자체를 생성하는 단계(selective search)가 느려서 병목이었다.

### Faster R-CNN

Fast R-CNN의 남은 병목이었던 region proposal 생성 단계 자체를 신경망으로 대체한 버전.

- **Region Proposal Network (RPN)**: selective search 같은 외부의 고전적 알고리즘 대신, CNN feature map 위에서 "여기 물체가 있을 것 같다"는 후보 영역들을 직접 예측하는 작은 네트워크를 둔다. 이 RPN도 anchor box 개념을 사용해서 각 위치마다 여러 모양의 후보를 동시에 평가한다.
- RPN이 뽑은 proposal을 그대로 Fast R-CNN과 같은 방식(RoI pooling + classification/regression)으로 이어서 처리한다.

결과적으로 "region proposal 생성 + classification"까지 전체 파이프라인을 하나의 신경망 안에서 end-to-end로 학습할 수 있게 되어서, R-CNN -> Fast R-CNN -> Faster R-CNN으로 갈수록 계속 빨라졌다 (그래도 YOLO/SSD 같은 single-stage 계열보다는 일반적으로 느린 편 — region proposal이라는 별도 단계가 있는 two-stage 방식이라서).

정리하면:
- R-CNN: proposal 뽑고 -> 각 proposal마다 CNN 따로 돌림 (느림)
- Fast R-CNN: 이미지 전체 CNN 한 번 -> RoI pooling으로 proposal별 feature 재사용 (빨라짐, 근데 proposal 생성이 병목)
- Faster R-CNN: proposal 생성도 네트워크(RPN)로 대체 -> 전체가 하나의 네트워크로 통합

## 6. Semantic Segmentation

지금까지 다룬 detection은 물체를 사각형(bounding box) 단위로만 짚어냈다. 근데 실제로 물체는 사각형이 아니라 임의의 모양을 하고 있으니, "이 픽셀 하나하나가 정확히 어느 클래스에 속하는가"를 더 정밀하게 알고 싶을 때가 있다. 이게 semantic segmentation이다.

- Object detection의 출력: bounding box 좌표 + 클래스
- Semantic segmentation의 출력: 입력 이미지와 **같은 해상도**를 가진, 각 픽셀마다 클래스 라벨이 매겨진 label map

예를 들어 자율주행에서 "이 픽셀은 도로, 이 픽셀은 자동차, 이 픽셀은 보행자, 이 픽셀은 하늘"처럼 픽셀 단위로 분류해야 정확한 주행 가능 영역을 알 수 있는데, 이런 경우 bounding box 단위 detection으로는 부족하고 pixel 단위 segmentation이 필요하다.

### Transpose convolution (Upsampling)

일반 CNN은 conv/pooling을 거치면서 이미지 크기(height, width)가 점점 줄어들고 채널 수(depth)가 늘어나는 방향으로 진행한다 (encoder 역할). 근데 semantic segmentation은 최종 출력이 원본 이미지와 같은 해상도여야 하므로, 어느 시점부터는 다시 크기를 키워나가는(upsampling) 과정이 필요하다. 이때 쓰는 연산이 transpose convolution(=deconvolution이라고도 부르지만 실제 역연산은 아님)이다.

일반 convolution이 "큰 입력 -> 필터를 슬라이딩하며 -> 작은 출력"을 만드는 것과 반대로, transpose convolution은 "작은 입력 -> 필터를 이용해 -> 큰 출력"을 만든다. 동작 방식은 대략:
- 입력의 각 픽셀 값에 필터를 곱해서, 그 결과를 출력의 해당 위치에(stride만큼 간격을 두고) 겹쳐서(overlap되는 부분은 더해서) 배치한다.
- 이 과정을 반복하면 작은 feature map에서 훨씬 큰 feature map으로 크기를 키울 수 있고, 이 필터 값들도 일반 conv filter처럼 학습 가능한 파라미터다.

즉, "이미지를 줄이는 필터"를 학습하듯이 "이미지를 키우는 필터"도 학습으로 얻을 수 있다는 게 핵심.

### U-Net 구조

Semantic segmentation에서 가장 널리 쓰이는 대표적인 구조가 U-Net이다 (이름 그대로 네트워크 모양을 그리면 U자 모양이 나온다).

구조는 크게 두 부분으로 나뉜다.

- **Encoder (수축 경로, contracting path)**: 일반 CNN처럼 conv + pooling을 반복하면서 이미지 크기는 줄고 채널(추상적 feature)은 늘어난다. 여기서 "이 이미지에 뭐가 있는지"에 대한 high-level 정보를 뽑아낸다.
- **Decoder (확장 경로, expanding path)**: transpose convolution을 반복하면서 다시 크기를 키워나가고, 최종적으로 입력과 같은 해상도의 pixel-wise classification map을 출력한다.

근데 encoder에서 계속 줄여나가다 보면, "이 물체가 정확히 어디서 시작하고 어디서 끝나는지" 같은 정밀한 위치/경계 정보(low-level detail)가 pooling 과정에서 많이 손실된다. 아무리 decoder에서 upsampling을 잘해도, 이미 없어진 세부 정보를 완전히 복원하기는 어렵다.

그래서 U-Net은 **skip connection**을 사용한다. Encoder의 각 단계에서 만들어진 feature map을, 크기를 줄이지 않은 원본 상태 그대로 decoder의 대응되는(같은 해상도의) 단계로 직접 연결해서 concatenate 시켜준다. 이렇게 하면:

- Decoder는 "지금까지 upsampling해서 얻은 high-level 정보(이 근처에 뭐가 있는지)" + "encoder에서 넘어온 low-level 정보(경계선이 정확히 어디인지, 세부적인 텍스처 등)"를 동시에 활용해서 최종 pixel 분류를 할 수 있다.
- 결과적으로 물체의 대략적인 위치뿐 아니라 경계(edge)까지 훨씬 정교하게 살아있는 segmentation map을 만들어낼 수 있다.

이런 encoder-decoder + skip connection 구조 덕분에 U-Net은 (원래 의료 영상용으로 제안됐지만) 지금은 일반적인 semantic segmentation 문제에도 널리 쓰이는 표준적인 baseline 구조가 되었다.

---

**전체 흐름 한 줄 요약**: classification(뭐가 있나) -> localization(그게 어디 있나, 하나만) -> detection(여러 개를 각각 어디 있는지) 순서로 문제가 확장되고, detection을 실제로 풀 때는 sliding window의 비효율을 conv 연산 공유로 없앤 게 YOLO 계열(single-stage, grid+anchor+NMS)과 R-CNN 계열(two-stage, region proposal+classification)이라는 두 갈래 접근으로 이어진다. 여기서 한 단계 더 정밀하게 들어가면, bounding box 대신 픽셀 단위로 분류하는 semantic segmentation(encoder-decoder + skip connection, 대표적으로 U-Net)이 된다.
