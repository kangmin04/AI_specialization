# Computer Vision (Deep Learning Specialization - Course 4)

Andrew Ng의 Coursera **Deep Learning Specialization** 중 4번째 코스인 "Convolutional Neural Networks"(컴퓨터 비전 파트)를 공부하면서 정리한 노트다. 기존 `ML/` 폴더가 "Machine Learning Specialization" 랩 코드를 다룬다면, 이 `computer-vision/` 폴더는 별도 스페셜라이제이션인 Deep Learning Specialization 중 컴퓨터 비전 관련 강의 내용을 정리한 마크다운 노트 모음이다. 코드 랩이 아니라 개념 정리 위주라서 `.py` 파일 대신 `.md` 파일로 구성했다.

## 읽는 순서

강의 진행 순서를 그대로 따라가면 된다. 뒤로 갈수록 앞의 개념(convolution, pooling 등)을 전제로 하니 순서대로 읽는 걸 추천한다.

0. **[00-prerequisites.md](00-prerequisites.md)** — 선수 개념 (Deep Learning Specialization Course 1~3)
   CV(Course 4) 노트를 읽기 전에 필요한 배경 지식 중 Machine Learning Specialization에는 없고 Deep Learning Specialization에서만 나오는 것들 - deep network 벡터화 표기법, weight initialization과 vanishing/exploding gradient, dropout, mini-batch/Adam 등 optimizer, batch normalization, bias/variance와 ML strategy - 만 추려서 정리.

1. **[01-cnn-foundations.md](01-cnn-foundations.md)** — CNN 기초
   FC network의 한계부터 시작해서 convolution 연산 자체(edge detection 직관, padding, stride), 다채널 convolution, pooling, 그리고 CNN이 왜 잘 동작하는지(parameter sharing, sparsity of connections)까지.

2. **[02-classic-and-modern-architectures.md](02-classic-and-modern-architectures.md)** — 대표적인 CNN 아키텍처들
   LeNet-5 -> AlexNet -> VGG-16 순으로 초기 아키텍처를 훑고, ResNet(residual block), 1x1 conv, Inception(GoogLeNet)까지의 구조적 발전을 정리. 이어서 transfer learning, data augmentation, MobileNet, EfficientNet 등 실무에서 CNN을 다룰 때의 전략까지 다룬다.

3. **[03-object-detection.md](03-object-detection.md)** — Object Detection
   "Classification -> Localization -> Detection"으로 문제가 확장되는 흐름을 따라가며, sliding window의 convolutional implementation, YOLO(grid, IoU, non-max suppression, anchor box), R-CNN 계열(R-CNN -> Fast -> Faster), 그리고 semantic segmentation(U-Net)까지 정리.

4. **[04-face-recognition-and-style-transfer.md](04-face-recognition-and-style-transfer.md)** — Face Recognition & Neural Style Transfer
   One-shot learning 문제에서 출발해 siamese network와 triplet loss로 얼굴 인식을 푸는 방법(Part 1), 그리고 content/style cost function과 gram matrix를 이용한 neural style transfer(Part 2)를 정리. 마지막에 1D/3D convolution으로의 확장도 짧게 언급.

5. **[05-generative-models-beyond-the-course.md](05-generative-models-beyond-the-course.md)** — Generative Models (강의 범위 밖 보충)
   Andrew Ng 강의(2017~2018)엔 없지만 컴퓨터 비전의 최근 흐름을 위해 따로 정리한 보충 노트. GAN(minimax game, mode collapse, DCGAN/cGAN/pix2pix/CycleGAN/StyleGAN)과 diffusion model(DDPM의 forward/reverse process, U-Net backbone, classifier-free guidance, latent diffusion/Stable Diffusion), 그리고 3D diffusion(voxel/point cloud/mesh/NeRF 표현, Point-E/Shap-E 같은 직접 생성 방식과 DreamFusion류의 Score Distillation Sampling)까지 다룬다.

## 메모

- 이 폴더의 노트들은 4개의 서브에이전트가 각 주제를 병렬로 조사해서 초안을 작성했고, 이후 내용/톤을 검토해서 정리했다.
- 저장소 전체의 톤(비공식적인 한국어 학습노트, 기술 용어는 영어 원문 유지)을 그대로 따랐다.
- 원 강의 순서/용어를 최대한 따르되, 논문 원제와 발표 연도 등 참고할 만한 포인트도 함께 남겨뒀다.
