import os; 
import torch;
import torch.nn as nn; 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt;

device = "mps" if torch.backends.mps.is_available() else "cpu"; 
print("device : ", device); 
root = os.path.join(os.path.dirname(__file__), "data"); 

# transforms : 이미지 데이터의 전처리 및 데이터 증강을 위해 제공하는 모듈(resize, crop, normalization, 등등)
tf = transforms.ToTensor(); # tf(image)형태로 사용. 
train = datasets.MNIST(root, train=True, download=True, transform=tf); 
test = datasets.MNIST(root, train=False, download=True, transform=tf)
# DataLoader: Dataset과 DataLoader 객체를 순회 가능한(iterable) 형태로 감싸서 미니배치(mini-batch) 학습, 데이터 셔플(shuffle), 병렬 처리 등을 쉽게 수행할 수 있도록 돕는 데이터 로딩 유틸리티
# ( 미니배치 생성: 전체 데이터를 지정한 크기(batch_size)만큼 묶어서 반환) / 데이터 섞기(Shuffle): 에폭(epoch)마다 데이터를 섞어 모델이 과적합(overfitting)되는 거 방지 
train_loader = DataLoader(train, batch_size=128, shuffle=True);  # train에만 shuffle. 순서가 고정되면 매 epoch마다 같은 batch 구성으로 gradient가 편향됨. 
test_loader = DataLoader(test, batch_size=512) 
#batch 크면 계산 속도는 빨라지지만, 학습이 덜 잘 되는건 아님. (batch작으면 한 epoch당 업데이트 횟수가 큼.)
x, y = next(iter(train_loader)); 