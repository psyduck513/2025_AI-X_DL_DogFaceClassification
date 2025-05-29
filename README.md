# 2025_AI-X_DL_DogFaceClassification

### Members : 
이형민 국제학부 2020082942 | hmlee2019@gmail.com
주예찬 경영학부 2019 | teotwg513@gmail.com

## 1.Proposal (Option A)
### Motivation : Why are you doing this?
애완견을 키우다보면 내 강아지와 더욱 더 소통하고 싶고, 어떤 생각을 하는지 알고 싶어지는 것이 자연스럽습니다. 애완견의 건강상태가 나빠지기라도 하면, 아 그때 그 표정이 몸이 안 좋다는 신호였나라고 되돌아보기도 한다.
해당 프로젝트의 목적은 말하지 못하는 반려견의 감정 상태를 파악함으로써, 보호자와의 소통을 더 깊이 있게 만들어주는 것입니다.

_in loving memory of typhoon_

### What do you want to see at the end?
어떠한 강아지의 사진을 입력시키면 해당 강아지의 기분상태를 판단해주는 ㅁㅎ댈 생성

## 2.Datasets
### About
Roboflow에 공개되어 있는 'Dog' Dataset을 사용하였습니다.
해당 데이터셋은 총 2358개의 강아지 사진으로 이루어져있고, 각 사진은 얼굴 부분이 네모 박스로 라벨링 되어 있습니다.
<img width="585" alt="image" src="https://github.com/user-attachments/assets/2c301208-d47f-400e-b693-0b2806cffa18" />

train set과 test set이 별도로 분리되어 있지 않아, 모델 개발 과정에서 학습/테스트 셋을 별도로 분리하였습니다.

## 3.Methodology
### 사용한 알고리즘
해당 프로젝트를 진행함에 있어 크게 두 가지 알고리즘을 사용하였습니다.
1) YOLO v8
YOLO v8은 Ultraylytics가 2023년에 공개한 실시간 객체 탐지 모델로 이미지를 처리하여 객체를 탐지하고 분류하는 모델이다.
   
2) CNN - 

### 개발 과정 중 핵심
## 4.Evaluation & Analysis

## 5.Related Work

## 6.Conclusions
