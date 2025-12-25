---
title: AI 이노베이션 스퀘어 수업(기본반) - deeplearning
date: 2019-04-29
draft: false
description: AI 이노베이선 스퀘어에서 배운 AI 공부 정리
categories:
- AI
- Deep Learning
tags:
- AI
- Lecture
- Deep Learning
slug: AILecture_deeplearning
---


<a id = '30th'></a>
# 2019년 6월 20일 목요일 30th

## Deep Learning

**2개 이상의** <span  style="color: red; font-size:30px;">Perceptron</span> **을 연결하여** <br>
**많은** <span  style="color: red; font-size:30px;">Layer</span> **로 구성된**  <span  style="color: red; font-size:30px;">기계학습</span> **을** <br> <span  style="color: red; font-size:30px;">Deep Learning</span>**이라고 부른다** <br>

## 인공지능 발전에 기여한 인물들

![ai_people_c](https://user-images.githubusercontent.com/33630505/59897521-f6659a80-9427-11e9-8ca8-af5bcb252ea1.jpg)
<br>
사진 출처: [brunch](https://brunch.co.kr/@hvnpoet/66#comment)<br>
<br>

### Frank Rosenblatt's Perceptron
```
1957년 Frank Rosenblatt에 의해 고안된 인공신경망의 한 종류로써 Perceptron이 발표되었습니다
이 Perceptron은 뉴런의 행동 방식을 모방하여 만들어졌습니다.
Single Perceptron은 입력값의 연산 결과가 1이면 activate되고 0이되면 deactivate하는 방식의 선형 모델입니다.
이렇게 하면 OR ,AND 연산이 가능하지만 XOR 연산을 할 수 없는 문제가 발생합니다
XOR 연산을 하기 위해서는 Perceptron을 하나 더 연결하여 다층 퍼셉트론을 통해 해결 할 수 있습니다
```

![선형 모델](https://user-images.githubusercontent.com/33630505/59901536-87437280-9436-11e9-994f-3bbfb1015d34.JPG)

![xor](https://user-images.githubusercontent.com/33630505/59915021-a5b96600-9456-11e9-8752-c941d955ab8d.JPG)

<br>

### Perceptron 학습 방법

> 입력값의 결과값을 1과 0으로 분류하고 실제값과 예측값의 활성함수 리턴값이 다를 경우 가중치를 조정하며 최적의 가중치를 찾아간다

```
AND Gate를 만든다고 가정하자
Data => (0,0) / Result => 0
        (0,1) / Result => 0
	(1,0) / Result => 0
	(1,1) / Result => 1
위와 같은 dataset이 있을때 Perceptron 학습방법은 다음과 같다


활성함수: y = 1 (x1*w1 + x2*w2 + w0 >= 0) or 0 (otherwise)
# x1, x2는 입력값 w1,w2는 가중치 w0는 bias(절편)

(0,0)일때
w0 < 0 이 되어야 결과값이 0이 나오기 때문에
w0 값을 임의로 -1를 찾았음

(0,1)일때
w2 -1 < 0을 만족하는 가중치를 임의로 0.5라고 한다

(1,0)일때
w1 -1 < 0을 만족하는 가중치를 임의로 0.5라고 한다

(1,1)일때
0.5 + 0.5 - 1 = 0 결과값이 0이기 때문에 y = 1을 만족하므로

학습된 Perceptron의 coefficient또는 weight값은 0.5, 0.5, intercept 또는 bias는 -1
```


### Bernard Widrow's Adaline

> Adaline(Adptive Linear Neuron)은 신경세포의 초기 모델.

### Adaline

> 입력값의 결과값과 실제 결과값을 비교하여 오차가 최소가 되도록 가중치를 찾아간다

### 예시
```
구구단에서 2단을 학습시키기 위한 딥러닝 이라고 가정 했을때

data: 2 => 2  X  0.5(가중치)  => result: 1
result = 1
real_result = 2 X 2 = 4
오차 : real_result - result = 3
가중치를 높여야 겠군!

data: 3 => 3  X  3(가중치)    => result: 9
result = 9
real_result = 3 X 2 = 6
오차: real_result - result = -3
가중치를 낮춰야 겠군!

data: 4 => 4  X  2(가중치)    => result: 8
result = 8
real_result = 4 X 2 = 8
오차: real_result - result = 0
오차가 없군! 학습을 멈춰야겠어! 정확도가 100%네 ?!
(실제로는 정확도 100%나오기 힘듦)
```

### Perceptron vs Adaline

요약하면 Perceptron은 임계값을 1로 잡고 입력값의 결과가 1이 넘어가면 활성함수에 의해 예측값이 나오고 실제값과 예측값이 다를 경우 가중치를 조정한다.<br>
Adaline은 입력값의 결과가 예측값이 되고 활성함수(실제값-예측값)의 리턴값을 최소화 하는 방향으로 가중치를 찾아간다는 점에서 차이가 있다.

### Neuron's communication
```
뉴런이 휴지상태일때 막전류가 -70mv 극성을 띄는데 뉴런에 자극이 가해지면 이온통로가 열리고
이온이 세포 안으로 들어오면 막전위의 변화를 알립니다.
그러면서 막전류가 -55mv에 도달하게 되면 수 천개의 나트륨 통로가 열리면서 뉴런 내부에
엄청난 양의 나트륨 이온이 세포 내부로 들어와 급격하게 양전하가 되거나 혹은 극성이 없어집니다.
엄청난 나트륨 이온의 유입으로 뉴런의 내부가 +30mv가 될때, 뉴런은 항상성 유지를 위해 나트륨 통로는 닫히고
칼륨 이온 통로가 열리면서 칼륨을 세포 밖으로 내보냅니다.
이러한 방식으로 뉴런 가지안에서 연쇄 반응을 통해 탈분극과 재분극을 반복하여 활동 전위가 전도 됩니다.
이때 활동 전위는 한 방향만으로 전도됩니다.
그리고 끝에 시냅스라는 부분에서 신경 전달물질을 세포 밖으로 내보내 다른 세포를 자극하기 위해 이동합니다  
이렇게 휴지상태 => 활동전위 => 신경전달 물질 => 다른 세포 자극 사이클을 반복하여 뉴런과 소통하게 됩니다.
```

### Adaline과 Neuron


Adaline | Neuron
Input Data | 타 뉴런들의 자극들
Weight | 수상돌기
Node | 세포체
Activation Function | 축삭돌기(휴지 상태=>활동전위)
Output Data | 축삭돌기 말단, 신경전달 물질

<br>


### Paul Werbos's MLP

```
1974년 하버드대 폴 워보스는 다층 퍼셉트론환경에서 학습을 가능하게 해주는
back-propagation 알고리즘을 고안해냈습니다.
그러나 신경망에 대해 냉랭했던 분위기 때문에
매장 당할까바 발표하지 못하고 8년 후 1982년에 저널에 발표하게 됩니다.
```

### Yann LeCun & David Rumelhart & Geoffrey Everest Hinton

```
저널에 발표되고 2년뒤 1984년에 신경망 연구로 박사논문을 준비하던 얀 레쿤이
논문을 발견하여 다시 세상에 나왔고, 1986년 럼멜하트와 힌튼교수에 의해 다시 부활하게 되었습니다

힌튼 교수는 홉필드 네트워크에 신경망 알고리즘을 결합시켜 볼츠만 머신을 만들어냅니다.
그리고 마침내 1998년 힌튼 교수 밑에서 박사과정을 밟고있던 얀쿤과 요수아 벤지오가
볼츠만 머신에 back-propagation을 결합하여 CNN(Convolutional Neural Networks)알고리즘 만들어 냅니다.
```


## Scikit Perceptron

```python
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
pct = Perceptron()
iris = sns.load_dataset('iris')

iris.iloc[:,:-1] = mms.fit_transform(iris.iloc[:,:-1])
iris.species = iris.species.map({"setosa":0,"versicolor":1,"virginica":2})

pct.fit(iris.iloc[:,:-1],iris.iloc[:,-1])  

# 옵션 early_stopping => overfitting 막기위한 방법 / fit_intercept 가중치를 사용할지 말지

vars(pct)
:
{'loss': 'perceptron',
 'penalty': None,
 'learning_rate': 'constant',
 'epsilon': 0.1,
 'alpha': 0.0001,
 'C': 1.0,
 'l1_ratio': 0,
 'fit_intercept': True,
 'shuffle': True,
 'random_state': 0,
 'verbose': 0,
 'eta0': 1.0,
 'power_t': 0.5,
 'early_stopping': False,
 'validation_fraction': 0.1,
 'n_iter_no_change': 5,
 'warm_start': False,
 'average': False,
 'n_iter': None,
 'max_iter': None,
 'tol': None,
 'class_weight': None,
 'n_jobs': None,
 '_tol': None,
 '_max_iter': 5,
 'coef_': array([[-0.80555556,  1.54166667, -1.57627119, -1.75      ],
        [ 0.11111111, -5.75      ,  1.45762712, -2.70833333],
        [ 0.52777778, -2.41666667,  4.13559322,  5.875     ]]),
 'intercept_': array([ 1.,  1., -6.]),
 't_': 751.0,
 'classes_': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
 '_expanded_class_weight': array([1., 1., 1.]),
 'loss_function_': <sklearn.linear_model.sgd_fast.Hinge at 0x242aef05770>,
 'n_iter_': 5}
```

## loss, cost, objective function

loss, cost : 실제값 - 예측값 <br>
즉, 값이 작아질 수록 정확해진다는 의미 <br>

objective : 유사도 <br>
즉, 비슷하면 비슷할 수록 값이 커지고 값이 크면 정확해진다는 의미

> 오차를 구하는 방법도 여러가지이다.

## Feed-forward vs Back-Propagation

Feed-forward: 벡터연산(행렬), 앞으로 나아가는 연산 <br>
Back-Propagation: 미분(편미분) <br>

## Vanishing gradient problem

> 기울기 값이 사라져 학습이 안되는 문제

relu 알고리즘으로 이러한 문제 해결!

## Neural network 표현방식

```
1. 전통적인 머신러닝 방식으로 표현
2. Graphical neural network
```

### Graphical neural network

![graphical](https://user-images.githubusercontent.com/33630505/59913849-fbd8da00-9453-11e9-9d49-9dd0975e88b1.JPG)

## Tensorflow

### 절차
```
0. import
1. Data 불러오기
2. train-test-split
3. 학습 가능하도록 차원 변환 (n차원 => 1차원)
4. Sequence 만들기 (Layer)
5. 학습 (어떻게 학습시킬까 : compile)
```

### 예시

```python
!pip install tensorflow==2.0.0b1 # 설치
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

knn = KNeighborsClassifier()
mnist = tf.keras.datasets.mnist # 가장많이 쓰이는 손글씨 데이터
data=mnist.load_data()  # 데이터 불러오기 / numpy 처럼 보이지만 tuple임

data[0][0].shape  # x
: (60000, 28, 28)
data[0][1].shape # y
: (60000,)

plt.imshow(data[0][0][0], cmap='gray')
data[0][1][0]
: 5

(X_train, y_train),(X_test, y_test) = mnist.load_data()

# 2차원 데이터 1차원으로 바꿔주기
# temp, result 방식 둘 중 원하는 방식으로
temp = [x.flatten() for x in X_train]
result = np.array(list(map(lambda x:x.flatten(), X_train)))
```

## Keras쓰는 4가지 방식


<hr>


<span style="color: red">아직 완벽하게 복습이 되지 않아 부정확한 내용이 있을 수 있음을 알려드립니다...</span><br>

<hr>

Perceptron : [tistory](https://sacko.tistory.com/10)<br>
Adaline, gradient descent : [m_blog](https://m.blog.naver.com/samsjang/220959562205) <br>
Vanishing gradient problem : [tistory](https://ydseo.tistory.com/41)<br>
ReLu: [tistory](https://mongxmongx2.tistory.com/25)<br>
인공지능 그림으로 배우기: [brunch](https://brunch.co.kr/magazine/yamanin)<br>
CNN: [tistory](https://hamait.tistory.com/535)<br>


**복습시간**   19시 10분 ~ 22시,  / 총 2시간 50분


<br>


<a id = '34th'></a>
# 2019년 6월 27일 목요일 34th

## Pandas 시계열 분석



## Deep Learning에서 Overfitting 막는 방법

```
1. 앙상블
2. Regularization
3. Drop out
4. Early stopping
5. 데이터 양을 많이
```

## Deep Learning에서 중요한것은?

<span  style="color: skyblue; font-size:30px;">Data</span> &nbsp; **vs**  &nbsp; <span  style="color: skyblue; font-size:30px;">Algorithm</span>  <br>

![data](https://user-images.githubusercontent.com/33630505/60263814-98035500-991c-11e9-92d9-f6bfafbb0485.JPG)

출처: Peter Norvig's The Unreasonable Effectiveness of Data 논문 <br>

> Peter Norvig의 논문에서도 알 수 있듯이 복잡한 문제에서 딥러닝의 성능이 좋으려면 <br>
> 좋은 알고리즘 보다는 더 많은 데이터가 중요하다는 것을 알 수 있다. <br>
> 하지만 데이터 확보가 어려운 경우도 많고 중간 규모의 데이터는 매우 흔한 일 이기 때문에 <br>
> 알고리즘의 중요성 또한 무시할 수 없다.

<br>

## 당신의 데이터는 일반적입니까?

> Machine Learning, Deep Learning에서 학습시키는 Label data는 항상 정답이라는 가정을 했었습니다. <br>
> 그런데 과연 그 정답 데이터가 진짜 현실세계에서 정답 데이터 인지 확신할 수 있을까요? <br>
> 그리고 데이터를 수집한 사람이 혹은 시스템이 편향된 데이터를 수집하지는 않았을까요? <br>
> 좋은 데이터를 갖고 있다고 가정하더라도 학습을 시키는 사람에 따라서 편향된 모델이 나올 수 있다는 것을 명심하자 <br>

<br>

### 데이터는 공정한 데이터 여야 한다.

> 공정한 데이터... 기준이 불명확하고 사람마다 공정성의 기준이 다를 수 있으니 <br>
> 사람이 저지르는 편향 유형을 살펴보고 편향된 데이터인지 판단하는 눈을 기르도록 하자.

<br>

### 편향의 유형

```
1. 보고편향
2. 자동화 편향
3. 표본 선택 편향
4. 그룹 귀인 편향
5. 내재적 편향
```

#### 1. 보고 편향

```
보고 편향은 수집된 데이터의 속성 및 결과의 빈도가 실제 빈도를 정확하게 반영하지 않을때 나타납니다.
예를 들어 쇼핑몰 리뷰가 이러한 특징을 갖고 있다.
사람들은 무언가 물건을 사고 정말 마음에 들때 혹은 정말 마음에 들지 않을때 리뷰를 남기는
특징이 있다. 보통 정말 좋을때와 정말 나쁠떄 중간 지점의 리뷰는 수집되기 힘든걸 볼 수 있다.
```

#### 2. 자동화 편향

```
자동화 편향은 두 시스템의 오류율과 관계없이 자동화 시스템이 생성한 결과를 비 자동화 시스템이
생성한 결과보다 선호하는 경향을 말합니다.
예를 들어 병아리 성별을 감별하는 자동화 시스템과 사람이 직접 감별하는 것 두 가지가 있다는 사실을 알려 줬을 때 어느 것이 정확도가 높을까? 라고 질문을 한다면 보통 사람들은
자동화 시스템을 더 신뢰할 것이다. 하지만 실제로는 병아리 성별 감별하는 전문가 즉 감별사는
자동화 시스템의 부정확성 때문에 고액의 연봉을 받는 직종이라고 한다.
```

#### 3. 표본 선택 편향

```
1. 포함 편향
- 선택된 데이터가 대표셩을 갖지 않는 경우
- ex) A 고등학교의 영어 성적 데이터를 갖고 전국의 고등학생 영어 성적의 분포를 알려고 하는 경우
2. 무응답 편향
- 데이터 수집시 참여도의 격차로 인해 데이터가 대표성을 갖지 못하는 경우
- ex) 대선운동 할때 유선전화로 여론조사를 시행한 결과로 대선 후보의 당선 예측을 하는 경우
3. 표본 추출 편향
- 데이터 수집 과정에서 적절한 무작위 선택이 적용되지 않았을 경우
- ex)
```

#### 4. 그룹 귀인 편향

```
1. 내집단 편향
- 자신이 소속된 그룹 또는 본인이 공유하는 특성을 가진 그룹의 구성원을 선호하는 경향을 나타내는 경우
- ex) 유유상종, XX대를 졸업한 A라는 사람이 동문인 B를 만났을 때 A가 B를 더 챙겨주고 싶어하고 친해지려하는 심리
2. 외부 집단 동질화 편향
- 자신이 속하지 않은 그룹의 개별 구성원에 관해 고정 관념을 갖거나 그들이 모두 동일한 특징을 가진다고 판단하는 경향을 나타내는 경우
- ex) 어떤 A라는 사람이 컴퓨터를 전공했고 어떤 B라는 사람은 영문과를 전공했다고 했을 때 B라는 사람이 A라는 사람을 보고 "컴퓨터 수리 잘하겠네?" 라는 생각을 갖는 경향
```


#### 5. 내재적 편향

```
1. 확증 편향
- 자신의 신념과 일치하는 정보는 받아들이고, 일치하지 않는 정보는 무시하는 경향
- ex) 듣고싶은 것만 듣고 보고싶은 것만 보는 심리, 담배를 피는것은 몸에 해롭다는 것을
- 알지만 일부 담배를 피고 오래 사는 사람을 보고 '담배 아무리 적게 펴도 병걸릴 사람들은
- 다 걸리고 안걸릴 사람은 아무리 많이펴도 안걸린다,
- 100살 넘게 담배펴도 건강한 할아버지 봐라!'라는 식의 사고방식
2. 실험자 편향
- 실험자가 바라는 방향대로 되기를 바라는 마음에서 발생되는 편향
```



머신러닝 단기집중과정 (편향 출처) : [google](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias)
<br>

<hr>

## Tensor board

```python
%load_ext version_information
%load_ext tensorboard

version_information tensorflow
:
Software	Version
Python	3.7.3 64bit [MSC v.1915 64 bit (AMD64)]
IPython	7.4.0
OS	Windows 10 10.0.17134 SP0
tensorflow	2.0.0-beta1
Thu Jun 27 17:01:13 2019 ¢¥eCN©öI¡¾©ö C¡ÍA¨ª¨öA

import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 현재시간을 문자열로 바꿔서 저장

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback]
         )

%lsmagic  # %load_ext tensorboard를 하면 %tensorboard가 생긴다
: Available line magics:
%alias  %alias_magic  %autoawait  %autocall ....%tensorboard.....

%tensorboard --logdir log/
```

![tensorboard](https://user-images.githubusercontent.com/33630505/60267850-187a8380-9926-11e9-84a8-654798556cd4.JPG)



<a id = '35th'></a>
# 2019년 6월 28일 금요일 35th (마지막 수업)
