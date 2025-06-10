# 2025_AI-X_DL_DogFaceClassification

### Members : 
이형민 국제학부 2020082942 | hmlee2019@gmail.com

주예찬 경영학부 2019028977 | teotwg513@gmail.com

## 1.Proposal (Option A)
### Motivation : Why are you doing this?
이미지 인식 및 분류 알고리즘은 기계가 시각 데이터를 인식하고 자체적으로 분석하여 분류하는 작업을 수행하는 알고리즘입니다.
최근 자율주행 자동차나 IoT 기술과 같은 분야에서 이 알고리즘의 활용도가 크게 증가하고 있으며, 이에 따라 해당 기술 자체의 중요성과 관심도도 함께 높아지는 추세입니다.

이러한 흐름에 따라, 기계학습 분야에서 가장 활발히 연구되는 분야 중 하나인 이미지 인식 및 분류 알고리즘을 직접 개발하고 실행해보고자 했습니다.
YOLO나 MobileNet과 같이 접근성이 좋은 대표적인 모델들을 활용하였고, 단순하지만 직관적인 인사이트를 추출하는 과정을 설계하기 위해 ‘강아지 사진’과 ‘강아지 감정 분류’ 데이터셋을 사용하였습니다.


### What do you want to see at the end?
어떠한 강아지의 사진을 입력시키면 해당 강아지의 기분상태를 판단해주는 모델 생성

## 2.Datasets
### About
**1) YOLO 모델 학습용 데이터**

YOLO 모델에 강아지 얼굴을 자동으로 인식하고 crop하도록 학습시키는데에는 Roboflow에 공개되어 있는 ‘Dog’ Dataset을 사용하였습니다.
해당 데이터셋은 총 2358개의 강아지 사진으로 이루어져있고, 각 사진은 얼굴 부분이 네모 박스로 라벨링 되어 있습니다.

train set과 test set이 별도로 분리되어 있지 않아, 모델 개발 과정에서 학습/테스트 셋을 별도로 분리하였습니다.

<img width="585" alt="image" src="https://github.com/user-attachments/assets/2c301208-d47f-400e-b693-0b2806cffa18" />


**2) MobileNetV2 학습용 데이터**

MobileNetV2 모델을 활용하여 강아지 얼굴 사진을 ‘angry’, ‘happy’, ‘relaxed’, ‘sad’의 네 가지 감정으로 분류하기 위해, Roboflow에 공개된 ‘dog_emotions’ 데이터셋을 사용하였습니다.
이 데이터셋은 총 3,996장의 강아지 얼굴 이미지로 구성되어 있으며, 각 사진은 앞서 언급한 네 가지 감정 중 하나로 라벨링되어 있습니다.

![image](https://github.com/user-attachments/assets/af88cebc-d406-4d79-adb6-29d5538d94ff)


## 3.Methodology
### 사용한 알고리즘
해당 프로젝트를 진행함에 있어 크게 두 가지 알고리즘을 사용하였습니다.
### 1) YOLO v8
YOLO v8은 Ultraylytics가 2023년에 공개한 실시간 객체 탐지 모델로 이미지를 처리하여 객체를 탐지하고 분류하는 모델입니다.
기존 버전에 비해 성능과 속도 모두 향상되었고, 특히 작은 물체 탐지 성능이 우수하여 해당 프로젝트에 적합하였습니다.
2025년 6월 기준, 버전 11까지 출시는 되었지만 YOLOv9~11는 대부분 연구자용으로 비전문가나 일반 프로젝트에는 부담이 큽니다. 또한 YOLOv8은 충분한 실사용 검증이 이루어져 가장 많이 사용되는 가장 안정적인 모델입니다.

![image](https://github.com/user-attachments/assets/7d0eb5f6-2109-4a6b-b4f4-a241d39f0869)

### Anchor-Free Detection
해당 버전의 큰 특징으로는 Anchor Free Detection이 있다.
기존에는 여거개의 예측될법한 Box들의 초기값을 설정(anchor) 후 이 값들을 통해 실제 검출되는 객체의 크기가 결정되는 방식으로 구동되었다. 하지만 이제 미리 입력된 Anchor Box를 사용하지 않고, 객체의 center를 직접 예측하는 방법을 사용한다.
![image](https://github.com/user-attachments/assets/13b0fd3c-01c5-43d1-946c-dee58d9ce7ad)

객체 탐지에서 중요한 두가지 요소는 정확성과 속도입니다. 
두 요소는 서로 반비례에 있으며, 상황에 따라 두 요소 모두 중요할 수 있습니다.
단 몇 초의 차이로 사고가 나고 안 나고를 결정되는 자율주행의 경우가 속도가 정확성보다 중요한 상황의 예시입니다. 

이와 더불어, 노트북으로 작업을 진행하기에 리소스제약이 있을 수 있기에, 
이번 프로젝트에서는 모델을 정확성 보다 속도에 더욱 안정적으로 모델을 시험하는 것에 중점을 두어 진행하였습니다.

따라서 YOLO v8의 nano, small, medium, large, extra large 모델 중 nano 모델을 설정하여 모델 설계를 진행하였습니다.
<img width="601" alt="image" src="https://github.com/user-attachments/assets/efa957ee-9d99-4e92-bbf5-897d188e99a4" />

3) CNN (Convolutional Neural Network)
합성곱 신경망(CNN)은 이미지나 동영상처럼 격자 형태의 데이터를 처리하는 데 특화된 딥러닝 모델입니다.
기존의 신경망(NN)이 입력을 평면 벡터로 변환해 처리하는 반면, CNN은 입력 데이터의 공간적 구조를 보존하면서 다층적인 특징(Feature)의 계층 구조를 자동으로 학습하도록 설계되어 있습니다.
이러한 구조 덕분에 CNN은 컴퓨터 비전 분야에서 뛰어난 성능을 보이며, 의료 영상 분석, 자율주행, 보안 감시, 얼굴 인식 등 다양한 실세계 응용에서 AI 기반 혁신을 이끌고 있습니다.

### MobileNetV2
앞서 가볍고 속도를 우선 고려하였든, 다양한 CNN 모델들 중에서도 가볍고 수행 속도가 빠른 것으로 유명한 MobileNetV2를 사용하였습니다.
MobileNetV2는 2018년 구글이 발표한 경량화 딥러닝 모델입니다. 설계 특징으로는 linear transformation역할을 하는 linear bottleneck layer를 통해, 차원은 줄이되 중요한 정보(manifold of interest)를 그대로 유지하여 네트워크 크기는 줄어들지만 정확도는 유지하는 모델입니다.

![image](https://github.com/user-attachments/assets/7b6ff113-57fa-4a62-8fd9-f272a135e42e)
MobileNetV2는 고차원 이미지를 저차원에서의 다양한 특징으로 mapping되는데, 이 과정을 manifolds of interest를 구성한다고 합니다. 이 후, 이 manifolds of interest를 통해 저장된 정보들이 다시 layers를 거쳐 저차원 영역으로 전달 되고, linear transformation을 통해 정보를 보존하게 됩니다. 

![image](https://github.com/user-attachments/assets/f46f1584-b6a8-4c61-b354-e9d491866e7c)

보존된 유용한 특징(feature)을 압축·확장하면서 효율적으로 표현하는 블록이 inverted residual block이고, 이를 숫자 벡터들로 정리, fully connected layer와 Softmax를 통해 특징들을 조합 후 확률로 추출하게 됩니다.



### 모델링 프로세스



- #### 0) 제반 사항


우선 개발환경의 경우 구글 Colab을 사용했습니다. 팀원 2명 모두 기존에 사용해본 적이 있는 개발환경이었기에 각자의 활용도에 이점이 있었고, 구글 드라이브 상에서 생성 및 수정할 수 있어 협업에 유리했기 때문에 선정했습니다.  

팀프로젝트 협업 툴은 구글 드라이브를 채택하였고, 이에 따라 이미지 데이터셋의 업로드, 저장, 편집 또한 구글 드라이브 상에서 진행했습니다. 코드를 통한 데이터 핸들링이 필요할 때에는 구글 Colab에 공유된 구글 드라이브를 마운트하여 진행했습니다.  



- #### 1) 강아지 얼굴 인식 모델 - YOLO v8 nano


YOLO 모델을 사용하기 위해, 가장 먼저 ultralytics 라이브러리를 install 하였습니다.  
ultralytics의 경우 빈번하게 활용되는 여타 라이브러리와 달리 구글 Colab에 설치되어 있지 않기 때문에, 하기 코드를 입력해 직접 install을 진행했습니다.  

```python
!pip install ultralytics
```


YOLO 모델의 아키텍처를 설정하는 yalm 파일의 경우, 데이터셋 zip 파일에 내포되어 있던 yalm 파일을 그대로 사용했습니다.  
따라서 하기 코드를 작성해 yaml 파일의 path를 정의하였습니다. extract_path의 경우 압축 해제한 데이터 폴더에 해당하는 구글 드라이브 주소로 정의해두었습니다.  

```python
# 기존 YAML 파일 경로 지정
yaml_path = f"{extract_path}/data.yaml"
```


이후 하기 코드와 같이 YOLO를 import한 뒤, 신속한 예측이 가능한 yolov8 nano 모델을 선정해 모델 객체를 생성하고 학습을 진행했습니다.  
불러온 yolov8 nano 모델의 경우 COCO 데이터셋을 통해 사전 학습된 모델입니다. 코드를 통한 학습 과정에서는 출력층에 대한 학습만 진항하게 됩니다.  
epochs의 경우 성능과 학습 소요 시간을 고려해 20으로 설정했습니다. 차후 기술하겠지만, epoch 20에서도 준수한 성능이 도출되어 20이면 성능 상 충분하다 판단했습니다.  
이미지 사이즈는 640으로 설정했습니다.  

```python
# 모델 학습

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data=yaml_path,
    epochs=20,
    imgsz=640
)
```


하단 이미지와 같이 epoch 20에 대한 학습을 완료했습니다.  

![image](https://github.com/user-attachments/assets/70680a4e-66bc-4b09-90ee-893eb0ab27cf)



- #### 2) 강아지 감정 분류 모델 - MobileNetV2


모델 학습에 앞서, MobilNetV2 학습에 사용할 강아지 표정 이미지 데이터를 전처리하였습니다.  
각 픽셀 값을 신경망에 입력할 수 있도록 0과 1사이의 값으로 Min-Max 정규화를 진행했습니다.  
또한 MobilNetV2에 입력할 수 있는 이미지 사이즈와 현재 보유한 강아지 표정 이미지 사이즈가 상이하기 때문에, 모델에 입력 가능하도록 (224, 224) 사이즈로 변환했습니다.  

하기 코드와 같이 keras의 ImageDataGenerator을 사용하였고, 정규화를 진행하여 객체(train_datagen, val_datagen)를 생성한 뒤  
이미지 사이즈를 설정해 새로운 변수(train_generator, val_generator)에 지정했습니다.  

```python
# 데이터 리사이징

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 데이터 경로
train_dir = os.path.join(extract_path_1, 'train')
val_dir = os.path.join(extract_path_1, 'valid')

# 이미지 전처리 (리사이즈 + 정규화)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 제너레이터
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```


MobileNetV2 모델의 경우에도 사전 학습이 진행되어 있습니다.  
이에 하기 코드에서 베이스 모델 객체를 생성할 때 weights 매개변수를 imagenet으로 설정해 imagenet 데이터셋을 통해 학습한 가중치를 불러오고,  
include_top 매개변수를 False로 설정해 출력층은 제외합니다.  
input_shape 매개변수는 앞서 리사이징한 이미지 크기에 맞추어 (224, 224, 3)으로 설정해줍니다.  
base_model의 trainable 속성은 False로 설정해 사전 학습된 가중치를 신규 학습하지 않고 그대로 사용하게 합니다.  

```python
# 사전 학습된 MobileNetV2 불러오기
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 사전 학습된 모델로 이미지 특징 추출 -> 출력층 학습만 직접 진행
```


MobileNetV2를 불러오며 제외한 출력층에 대해 하기 코드와 같이 설정해줍니다.  

x 변수에 대해 base_model 객체의 마지막 출력값 형태(텐서)를 지정해주고,  
정보 손실을 줄이는 방향으로 최대값 풀링 대신 평균 풀링을 사용하도록 해줍니다.  
마지막으로 Dense() 함수를 통해 x 변수 설정을 활용한 output 출력층을 설정합니다.  
앞서 설정한 train_generator의 출력 클래스 개수를 매개변수로 넣어주고, 다중 분류 문제이기에 소프트맥스 함수를 사용합니다.  

```python
# 출력층 설정

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)
```


하기 코드로 앞서 입력층, 출력층을 묶어 모델을 생성해줍니다.  
입력층의 경우 불러온 베이스 모델을 사용하고, 출력층의 경우 이전 코드에서 설정한 output 변수로 설정해줍니다.  

```python
# 입력층, 출력층 설정값을 통해 모델 생성

model = Model(inputs=base_model.input, outputs=output)
```


하기와 같이 모델 컴파일을 진행해 옵티마이저와 손실 함수, 성능 지표를 설정해줍니다.  
옵티마이저의 경우 Adam으로 설정했습니다. 최근 경로 변화량에 적응적으로 학습하며 진행하던 최적화 속도에 관성을 부여하는 옵티마이저로, 우수하다 평가받으며 가장 대중적으로 사용되기에 선정했습니다.  
손실 함수는 다중 분류 문제이기에 크로스엔트로피 함수로 설정했습니다.  
또한 성능 지표의 경우 다중 분류 문제에 대해 직관적으로 이해하기 쉬운 정분류율(accuracy)를 사용했습니다.  

```python
# 모델 컴파일

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 손실 함수는 다중 크로스엔트로피 함수
# 성능 지표는 정분류율
```


마지막으로 하기 코드를 통해 학습을 진행합니다.  
앞서 설정한 train_generator, val_generator를 학습 데이터와 검증 데이터로 사용했습니다.  
epochs의 경우 학습 소요 시간을 고려해 20으로 설정했습니다.  

```python
# 모델 학습

history = model.fit(train_generator, validation_data=val_generator, epochs=20)
# 성능 확인을 위해 history 변수에 지정
```


후술하겠지만, MobileNetV2의 성능 향상을 위해 일부 하이퍼파리미터 값을 조정하였으나 상기에 기재한 프로세스가 가장 높은 검증 데이터 성능을 보였습니다.  
사전 학습된 레이어 중 일부를 초기화하고 신규로 학습해 프로젝트 데이터셋에 보다 적합한 학습을 시도해보았고,  
출력층에 Dropout을 추가하여 과대적합을 방지해보고자 했습니다. 해당 과정은 4. Evaluation & Anlysis에서 간단히 설명드리겠습니다.  


## 4.Evaluation & Analysis


- #### 1) 강아지 얼굴 인식 모델 - YOLO v8 nano

![image](https://github.com/user-attachments/assets/a34d0f3e-1d4b-4d69-8863-14cf96665564)

![image](https://github.com/user-attachments/assets/33e84ef4-6442-4079-85e1-82f3ff27466e)


## 5.Related Work
YOLOv8 and Dog Face Detection
https://www.kaggle.com/code/martinpelaezdiaz/dog-face-detection-with-yolov8-and-ultraytics#5.-Model-Selection-and-Configuration

Emotion classification with minimal epochs
https://www.nature.com/articles/s41598-022-11173-0

## 6.Conclusions

## 7.Works Cited
https://docs.ultralytics.com/ko/models/yolov8/#yolov8-usage-examples
https://www.ultralytics.com/ko/glossary/convolutional-neural-network-cnn?utm_source=chatgpt.com
https://blog.naver.com/skfnsid123/223199760485
https://velog.io/@woojinn8/LightWeight-Deep-Learning-7.-MobileNet-v2
