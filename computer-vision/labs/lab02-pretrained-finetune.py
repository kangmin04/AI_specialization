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

PRETRAINED = True; 

weights = models.ResNet18_Weights.IMAGENET1K_V1 # 이때 가중치를 다운로드하는게 아님!! , 가중치 + 그 가중치 만들 때 쓴 설정 전체"를 담은 명세서(enum)임! 

preprocess = weights.transforms() # resize 256 -> 중앙을 224 * 224 크기로 자르기 -> rescale(픽셀값 범위를 [0,1]로) -> ImageNet mean/std로 normalize
print("process : ", preprocess)
categories = weights.meta["categories"] # imagenet 1000개 클래스 이름
# process :  ImageClassification(
#     crop_size=[224] : 가운데를 2224 * 224로 
#     resize_size=[256] : 짧은변은 256으로. 비율은 유지
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225] : Normalizae 계산
#     interpolation=InterpolationMode.BILINEAR
# ) : 여기서 toTenso도 일어남. (0,255정수를 0,1 float로 )

# --------------- PART1--------------- 
# Photo Classification (inference만. 학습아직안함. )
img_path = os.path.join(here, "my_photo2.jpg")
if os.path.exists(img_path):
    # models.renset18 : 층을 어떤 순서로 쌓을지 정한 구조를 만듦. 
    # 결국 구조 자체는 f(x) = wx + b로 동일함. 가중치는 학습으로 찾은 값이고, model(x)는 그 w,b로 fx를 계산. 
    model = models.resnet18(weights = weights).to(device).eval() # weights 에 학습된 가중치를 다운로드해서 넣어줌.  (이미 받아둔게 있다면 ~./cache에서 읽음) to : 모델의 모든 파라미터를 MPS(맥 GPU)메모리로 옮김 eval: 평가모드로 -> 이 3가지 동작이 메서드체이닝으로 연결됨


#   ResNet엔 BatchNorm 층 존재. BatchNorm은 모드에 따라 다르게 동작
#       - train() 모드: 지금 들어온 배치의 평균과 분산으로 정규화. 배치가 사진 1장이면 그 1장의 통계로 정규화하게 되니 값이 엉망이 됨
#       - eval() 모드: 학습 때 ImageNet 전체로 미리 쌓아둔 고정 통계를 사용 => 추론할땐 항상 evak. 
    img = Image.open(img_path).convert("RGB") # image.open은 PIL(이미지 라이브러리)로 파일 엶. 이때 리턴은 텐서가 아닌 이미지객체. convert(RGB)로 채널을 3개로. 
    x = preprocess(img).unsqueeze(0).to(device)  # (3,224,224) -> (1,3,224,224): 배치 차원 추가
    #pytorch는 항상 배치단위(B, C, H, W)로 입력받음. 일반 색상 이미지는 (3, 244, 244)로 축 3개뿐이라 그대로 넣으면 에러남. unsqueeze(0)으로 0번 위치에 크기 1짜리 축을 끼워넣어서 사진 1장으로 된 배치로 만드는 것 !! 

    #pytorch는 모든 연산을 기록해둠. 나중에 loss.backword()로 gradient 계산하려면 그 기록이 필요하기때문!!  추론만 할땐 학습을 안하니 그 기록이 필요없음. no_grad 내에선 기록 끄기에 메모리 덜 쓰고 더 빠름. 
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top5 = probs.topk(5)
    for p, i in zip(top5.values, top5.indices):
        print(f"{categories[i.item()]:>25s}  {p.item():.3f}")
    plt.imshow(img)
    plt.title(categories[top5.indices[0].item()])
    plt.axis("off")
    plt.show()

# --------------- PART2--------------- 
#fine-tuning
train = Subset(datasets.CIFAR10(root, train=True, download=True, transform=preprocess), range(5000))
# transform : 사진을 꺼낼떄마다 preprocess 전처리 적용. 미리 50000장 다 적용하는게 아닌, train[i]로 꺼내올 때 변환. 
# range(5000): 전체 데이터(50000장)으로 돌리면 한 epoch가 오래 걸림. 
test = Subset(datasets.CIFAR10(root, train=False, download=True, transform=preprocess), range(1000))
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader =  DataLoader(test, batch_size=256)

###1. 학습된 모델 통째로 가져오기!! 
model = models.resnet18(weights=weights if PRETRAINED else None)
# PRETRAINED = True면 ImageNet 가중치를 채움, False면 weights=None이라서 구조만 있고 가중치는 랜덤. 
###2. 전부 동결 
for p in model.parameters(): # model 안의 모든 학습 가능한 가중치 텐서 반환
    p.requires_grad = False # 가중치 동결 

#      requires_grad: 모든 파라미터 텐서가 가진 플래그. "이 텐서에 대해 gradient를 계산할까?"라는 뜻
#   - True: loss.backward()가 이 텐서의 .grad(= dj_dw)를 계산 -> opt.step()으로 업데이트
#   - False: gradient를 계산하지 않음. -> grad가 비어 있으니 업데이트도 안됨 => 고정(freeze)

###3. fc = 마지막 층! -> 이 마지막 층만 새것으로 교체 
model.fc = nn.Linear(model.fc.in_features, 10) # nn.Linear : 입력데이터에 선형변환을 수행하는 완전연결층. (in_features : 입력데이터의 마지막 차원크기(입력특성의 수), out_features : 출력데이터의 마지막 차원크기(출력 특성수)  
                                               # fc는 마지막 linear 층. 즉, 위의 가중치 동결 통해서 그전의 데이터는 그대로 쓰고, 마지막 계산만 다르게함! !
model = model.to(device)
# print("model : ", model)  # 여러 layer이 출력됨. 
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) # 텐서에 포함된 전체 element 개수를 int로 반환
n_total = sum(p.numel() for p in model.parameters())
print(f"학습되는 파라미터 {n_train:,} / 전체 {n_total:,}")


opt = torch.optim.Adam(model.fc.parameters(), lr = 1e-3) # optimizer 에도 fc만 넘긴다! 
loss_fn = nn.CrossEntropyLoss()


for epoch in range(3):
    model.eval() 
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) #GPU로 이동 
        loss = loss_fn(model(x), y)
        # zero_grad: backward의 재료는 이전 .grad가 아니라 이번 forward 때 저장한 중간값(계산 그래프)임.
        #   .grad는 backward 결과를 담는 그릇이고, PyTorch는 여기에 덮어쓰지 않고 += 로 더함.
        #   -> 안 지우면 이전 배치 기울기 위에 쌓임 (실험: -16 -> -32). 지우고 backward하면 = 새 기울기 (코스의 dj_dw = ... 와 같은 효과)
        #   누적 방식인 이유: gradient accumulation(작은 배치 여러 번 backward 후 step 한 번), loss 여러 개 합치기 등에 쓰려고
        opt.zero_grad()
        loss.backward()
        opt.step()

    correct = 0
    with torch.no_grad():
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {correct / len(test):.3f}")
