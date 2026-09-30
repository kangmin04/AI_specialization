import os; 
import torch;
import torch.nn as nn; 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt;

device = "mps" if torch.backends.mps.is_available() else "cpu"; 
print("device : ", device); 
root = os.path.join(os.path.dirname(__file__), "data"); 

#1. 데이터 
# transforms : 이미지 데이터의 전처리 및 데이터 증강을 위해 제공하는 모듈(resize, crop, normalization, 등등)
tf = transforms.ToTensor(); # tf(image)형태로 사용. 
train = datasets.MNIST(root, train=True, download=True, transform=tf); 
test = datasets.MNIST(root, train=False, download=True, transform=tf)
# DataLoader: Dataset과 DataLoader 객체를 순회 가능한(iterable) 형태로 감싸서 미니배치(mini-batch) 학습, 데이터 셔플(shuffle), 병렬 처리 등을 쉽게 수행할 수 있도록 돕는 데이터 로딩 유틸리티
# ( 미니배치 생성: 전체 데이터를 지정한 크기(batch_size)만큼 묶어서 반환) / 데이터 섞기(Shuffle): 에폭(epoch)마다 데이터를 섞어 모델이 과적합(overfitting)되는 거 방지 
train_loader = DataLoader(train, batch_size=128, shuffle=True);  # train에만 shuffle. 순서가 고정되면 매 epoch마다 같은 batch 구성으로 gradient가 편향됨. 
test_loader = DataLoader(test, batch_size=512) 
#batch 크면 계산 속도는 빨라지지만, 학습이 덜 잘 되는건 아님. (batch작으면 한 epoch당 업데이트 횟수가 큼.)

# iter 통해서 이터러블 하나 꺼냄. next로 그 iterator에서 다음것을 꺼냄. --> 한 batch가 나옴. 
x, y = next(iter(train_loader)); 
print("x : ", x.shape, "y:", y.shape)


# 2. 모델 conv -> pool
# MaxPool2D : 이 근처에 무늬가 있었는가만 남기고 정확한 위치는 버림. 크기가 절반이 되어 계산이 줆. 
# conv를 두 번 하는 이유: 첫 번째는 선, 모서리 같은 단순한 무늬를 찾고, 두 번째는 그 지도들을 조합해 곡선, 고리 같은 더 복잡한 무늬를 찾음. 
# linear : tersorflow에서의 dense layer임. 
model = nn.Sequential(
    nn.Conv2d(1,32,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 표 1장, 필터 32개, 필터크기 3*3  -> 필터 32개라 출력도 32개. (무늬지도 32장. 28 * 28) #relu로 음수부분은 0으로 -> 아예 검은색.  maxPooling2D(2) : 2 * 2구역을 하나로 -> 기존엔 28 * 28 이었다면 14 * 14가 됨. 
    nn.Conv2d(32, 64, 3, padding=1 ), nn.ReLU(), nn.MaxPool2d(2),  # 32장 입력받고 필터 64개로 다시 훑음 -> 무늬 지도 64장 (14 * 14) 그리고 maxPool2D()로  7 * 7로 
    nn.Flatten(), # 64장, 7 * 7 이미지를 1차원으로 flateen -> 3136개 (64 * 7 * 7 ) 숫자 한줄로 
    nn.Linear(64 * 7 * 7, 10 ), # 3136개를 숫자 10개로 (숫자 0 - 9 정수로)
).to(device) # 모델 안의 가중치를 맥 GPU로 옮김. 

opt = torch.optim.Adam(model.parameters(), lr=1e-3); # learningRate : 0.001
loss_fn = nn.CrossEntropyLoss(); 



#3. 학습루프 
for epoch in range(2): # 예제라 epoch 2번만 반복
    # batch 하나마다 예측 → 틀린 정도 측정 → 청소 → 방향 계산 → 수정

    model.train(); # 모델을 학습모드로 바꿈. 이 줄이 학습을 시작시키는건 아님 !!! (학습과 평가때 다르게 동작하는 층을 위한 스위치. 여기선 없음. (관례로 작성))
    for x, y in train_loader:  # train_loader의 batch 크기가 128 -> 128장씩 이미지를 꺼냄. x = (128, 1, 28, 28) y = (128, )
        x, y = x.to(device), y.to(device) # 데이터를 모델이 있는 GPU로 옮깁니다. 모델과 데이터가 다른 곳에 있으면 에러가 남. 
        logits = model(x) # 이미지 128장이 모델 통과해서 (128,10)나옴
        loss = loss_fn(logits, y) #cost function J
        opt.zero_grad() # 이전 gradient 지우기. - 잘못된 학습 방지: 이전 배치의 기울기가 남아 있으면 다음 배치 계산에 영향을 주어 모델이 엉뚱한 방향으로 업데이트됨. 실제로 기존 0.97-98에서 accuracy가 0.70까지 떨어짐.
        # PyTorch는 backward()를 부를 때마다 gradient를 덮어쓰지 않고 기존 값에 더함. 그래서 매번 지워야 이번 batch의 gradient만 남음
        loss.backward() # dj/dw 계산 (cost function 기울기 )
        opt.step() # w = w - lr * dj_dw 경사하강법

    model.eval() # 평가모드로. (model.train의 반대 스위치)
    correct = 0
    with torch.no_grad():
        for x, y in test_loader: 
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item() # argmax(1) : 1은 1번축(클래스 10개) 중에서 최댓값을 찾는 것 
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {correct / len(test):.4f}")
