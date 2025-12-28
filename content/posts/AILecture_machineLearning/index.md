---
title: AI 이노베이션 스퀘어 수업(기본반) - machine learning
date: 2019-04-29
draft: false
description: AI 이노베이선 스퀘어에서 배운 AI 공부 정리
categories:
- AI
- Machine Learning
tags:
- AI
- Lecture
- Machine Learning
slug: AILecture_machineLearning
image: machinelearning.png
---

<a id = '21th'></a>
# 2019년 6월 4일 화요일 21th

## 기계학습 분류

![learning model](https://user-images.githubusercontent.com/33630505/59347252-1eca0680-8d4f-11e9-9104-a788a22a72e3.JPG)

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">지도 학습</span> <br>

```
정답을 알려주며 학습시킨다.
예를 들어 '바퀴 4개, 문짝 4개, 도로위를 시속 0~200km(input data) 달릴 수 있는 것은 자동차(label data or target data)야'
라고 학습 시키고 학습을 바탕으로 모델이 예측할 수 있도록 하는 방법이다.

지도학습은 크게 Classification, Regression으로 나눈다.
Classification은 또 이진분류, 다중분류로 볼 수 있다.
이진분류 같은 경우 생존자 or 비생존자와 같이 둘 중 하나로 분류 가능한 것을 말한다.
LogisticRegression 알고리즘이 대표적인 이진 분류 알고리즘이다.
다중 분류는 어떤 데이터에 대해 여러 값 중 하나로 분류 가능한 것을 말한다.
예를 들어 축구공, 야구공, 농구공 등 Label data가 여러개로 나뉠 수 있는 경우를 말한다.
이때는 KNN알고리즘으로 분류 가능하다.
KNN알고리즘은 데이터가 많아지거나 Label data가 많아지면 성능이 떨어질 가능성이 높다.

Regression는 어떤 데이터들의 특징을 토대로 값을 예측하는 것을 말한다.
예를 들어 키가 170cm인 사람의 몸무게는 65kg이다와 같이 Label data가 실수 값을 갖거나
연속적, 범위가 정해지지 않은 경우 무한대인 경우이다.

분류인지 회귀인지는 label data가 유한개인지 무한개인지 생각해보면 된다.
```
<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">비지도 학습</span> <br>	    

```
정답을 알려주지 않고 비슷한 데이터들 끼리 군집화하여 학습한다.
예를 들어 '남자, 여자 사진을 무작위로 입력값으로 줬을 때 사진을 보고 공통적으로 보이는
특성들을 찾아 비슷한 특성끼리 묶어 남자, 여자를 학습 시킨 데이터를 기반으로 분류하는 것을 말한다.

비지도학습은 크게 Clustering, Visualization & Dimensionality  Reduction, Association으로 나뉜다.
Clustering은 비슷한 것끼리 묶는 방법이다.
Clustering 방법중 대표적인 알고리즘인 k-means는 예를 들어 3가지로 묶는다고 했을 때 데이터에서 무작위로 임의의 값을 3개 찍고
랜덤한 데이터 값에서 가까운 값을 찾아 평균을 낸다. 그러면 평균낸 값에서 가까운 값을 또 찾고 그 값에서 평균을 낸다.
이와 같은 작업을 반복하여 평균값이 변하지 않는 때를 찾아 그 평균 값을 기준으로 군집화 하면 그것이 클러스터링 방법이다.

Visualization & Dimensionality  Reduction은 데이터간의 상관성을 분석하여 포함시키지 않아도 예측하는데 큰 지장 없는
데이터 열을 줄임으로써 차원을 축소하는 방법이다.
대표적으로 pca 방법이 있다. pca알고리즘은 데이터 분포에서 variance가 큰 방향의 벡터에 데이터를 정사영하여
차원을 축소시킨다. 이렇게 했을 때 데이터의 구조는 크게 바뀌지 않으면서 차원은 감소시킬수 있기 때문이다.

Association은 유사한 요소를 찾아 묶는 것이다. 이때 유사성을 파악할때 데이터간의 차이를 측정하는 방법인
유클리드 거리 측정 방법과 비-유클리드 거리 측정법으로 나눌 수 있다.
예를 들어 '근처에 사는 사람은 비슷한 성격을 갖고 있을 것이다' 처럼 묶거나
'피자를 사는 사람은 꼭 콜라를 산다' 처럼 묶을 수 있다.
```

지도학습, 비지도학습 : [tistory](https://marobiana.tistory.com/155) <br>
차원 축소 (pca): [tistory](https://excelsior-cjh.tistory.com/167), &nbsp; [wikidocs](https://wikidocs.net/7646) <br>

## 기계학습 목적

<span  style="color: red; font-size:30px;">Data</span>**로 부터** <br>
<span  style="color: red; font-size:30px;">Specific</span>**문제** <span  style="color: red; font-size:30px">해결</span>**을 위한** <br>
<span  style="color: red; font-size:30px;">최적의 모델</span> **만들기**

## Data수집부터 예측까지 과정

```
0. Data 불러들이기
- 적합한 데이터 format으로 변환
1. Tidy data인지 확인하기
2. info
- missing datat 체크 (mino.matrix)
- object, category type은 숫자 타입으로 변환
- 차원의 저주 (필요없는 열 삭제)
- 데이터 갯수 확인 (데이터 갯수가 충분한가)
- 메모리 크기 확인 (내가 불러들일 수 있는 사이즈인가)
- label(target,class) data 포함 여부 확인
3. describe
- 지도학습을 하는 경우 pairplot으로 분류 가능한지 확인
- label data가 유한개인지 무한개인지 확인
- label data 유한 --> classifications
- label data 무한 --> regression
- 상관성 확인해야 하는 경우 heatmap
- boxplot
- 비지도학습을 하는 경우 label data가 없이 즉, 기준이되는 답이 없이 학습해야함.
- 비지도학습의 경우 클러스터링, 시각화와 차원축소, 연관 규칙 학습등의 알고리즘을 사용
4. 왜도, 첨도
- skew
- kurtosis
5. 5총사중 나머지 3개 (head, tail, sample)
6. 목적에 맞게 평가 척도에 따라 최적의 모델 생성
7. 성능 테스트
```


## label이 유한일때, 무한일때

### 유한일때
```python
import seaborn as sns

iris = sns.load_dataset('iris')
iris
```

![iris](https://user-images.githubusercontent.com/33630505/58871405-e138fe00-86fc-11e9-87a6-f7f31a8a8ca0.JPG)

### 무한일때

> mpg(연비)를 예측한다고 가정했을 때 연비는 정해져 있는 label이 아니기 때문에 무한 label임으로 regression 즉, 연속된 값을 예측해야 한다.

```python
import seaborn as sns

mpg = sns.load_dataset('mpg')
mpg
```
![mpg](https://user-images.githubusercontent.com/33630505/58871406-e138fe00-86fc-11e9-94e0-c1ec9499cbd8.JPG)

## masking 기법으로 missing data 보기

```python
import seaborn as sns

mpg = sns.load_dataset('mpg')
mpg.horsepower[mpg.horsepower.isnull()] # or mpg.horsepower[mpg.horsepower.isna()]

:
32    NaN
126   NaN
330   NaN
336   NaN
354   NaN
374   NaN
```

## missing data 그래프로 확인하기

```python
# pip install missingno

import missingno as mino
import seaborn as sns

mpg = sns.load_dataset('mpg')
mino.matrix(mpg)
```

![mino](https://user-images.githubusercontent.com/33630505/58872893-fb281000-86ff-11e9-8a18-258b12ba14d1.JPG)

> data의 양이 충분하지 않을때 missing data가 있으면 적당한 값으로 채워 넣어 성능을 높여주고,
적당한 값을 채우기 애매할 때는 missing data가 있는 row를 지워야 한다.

## 데이터를 쪼개 성능 비교하기

```python
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = sns.load_dataset('iris')
iris.species = iris.species.map({'setosa': 0, 'versicolor':1,'virginica':2})

knn = KNeighborsClassifier()
iris_data = iris[iris.columns[:-1]]
iris['species']

knn.fit(iris_data, iris['species'])

# 관례상 행렬은 대문자, 벡터는 소문자로 표기
X_train, X_test, y_train , y_test = train_test_split(iris[iris.columns[:-1]], iris.species)
len(X_train.index)
len(X_test.index)
: 112
  38    
# 75 : 25 비율로 쪼갬

knn.fit(X_train, y_train)
knn.predict(X_test)
y_test.values

: array([2, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 0, 2, 2, 1,
       1, 0, 0, 2, 2, 0, 0, 2, 1, 2, 2, 2, 0, 0, 0, 1], dtype=int64)
  array([2, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 0, 2, 2, 1,
       1, 0, 0, 2, 2, 0, 0, 2, 1, 1, 2, 2, 0, 0, 0, 1], dtype=int64)     

knn.predict(X_test) == y_test.values       
:
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True,  True,  True,  True,
        True,  True])

confusion_matrix(y_test, knn.predict(X_test))
:
array([[10,  0,  0],
       [ 0, 10,  0],
       [ 0,  1, 17]], dtype=int64)
# virginica를 예측한 test에서는 한번은 versicolor이라고 잘못 예측 했기 때문에 0 , 1 , 17
```

**Model** 학습이 끝난 알고리즘 + 데이터를 Model 이라고 한다



**복습시간** 18시 50분 ~ 19시 45분 / 총 55분  



<a id = '22th'></a>
# 2019년 6월 5일 수요일 22th

## One hot encoding & Label encoding

> 기계학습으로 예측분석을 하기 위해서는 문자를 숫자로 변환 해야하기 때문에 Encoding을 해야한다
그런데 문자를 숫자로 encoding할때 성능에 영향을 미치기 때문에 상황에 따라 encoding 방식을 달리 해야 한다

### One hot encoding

> 하나의 값만 True이고 나머지는 모두 False인 인코딩 방식

#### Scikit
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
t = ohe.fit(data[['species']])
t.array()

: array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       .....


# ohe.fit_transform(data[['species']]).toarray() 한번에 가능

ohe.inverse_transform([[1., 0., 0.]])
: array([['setosa']], dtype=object)

# 숫자로 인코딩 되기 전 문자
```

> Scikit's onehotencoder의 장점은 인코딩 되기 전 문자를 알 수 있다는 것.

<span style='background-color:red'>밑의 경우에는 어떻게 해야 할까..?</span><br>

```python
pd.DataFrame(ohe.fit_transform(data[['species']]), columns=['target'])

:
target
0	(0, 0)\t1.0
1	(0, 0)\t1.0
2	(0, 0)\t1.0
3	(0, 0)\t1.0
4	(0, 0)\t1.0
5	(0, 0)\t1.0
```

#### Pandas

```python
import seaborn as sns
import pandas as pd

data = sns.load_dataset('iris')
pd.get_dummies(data.species)

:
      setosa	   versicolor	 virginica
0	1	       0	    0
1	1	       0	    0
2	1	       0	    0
3	1	       0	    0
4	1	       0	    0
5	1	       0	    0
```


### LabelEncoder

#### Scikit
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit_transform(data.species)

:

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

#### pandas map

```python
import seaborn as sns

iris = sns.load_dataset('iris')
iris.species = iris.species.map('setosa': 0, 'versicolor':1,'virginica':2})
```

**Label encoding시 주의** 거리기반 알고리즘을 사용할 때 라벨 인코딩된 값으로 학습을 하게되면 숫자간의 격차로 인해 오차가 생길 위험이 있다. 예를 들어 0, 1, 2로 라벨 인코딩 되었다고 했을 때 0과 1사이 1과 2사이는 둘다 1간격만 있어 상관 없지만 0과 2사이에는 2간격이 생겨 학습시 주의해야 한다. 따라서 label encoding 해야할 때와 하지 말아야 할때를 잘 구분해야 한다.


## Bias , Variance

![optimal](https://user-images.githubusercontent.com/33630505/58957665-fd5f9c80-87db-11e9-8094-c5ba63e4375e.JPG)

> Bias가 높으면 값이 편향되어 있어서 값이 모여있고 Variance가 높으면 값이 퍼져있게된다.
> 현실에 적용할 수 있는 모델을 만들기 위해서는 Bias와 Variance가 만나는 지점을 목표로 삼고 모델을 만들어야 한다.

## Trade off  

![tradeoff](https://user-images.githubusercontent.com/33630505/58957666-fd5f9c80-87db-11e9-9dea-1802253b3aad.JPG)

> 다양한 데이터를 학습시키지 않게 되면 bias가 높아져 정확도가 떨어지는 대신 학습하지 않은 데이터중 일부는 어쩌다 맞추는 경우는 Underfit이다.
> 다양한 데이터를 학습시키긴 했지만 데이터 양이 많지 않아 bias는 낮지만 variance가 높아 학습한 데이터에 대해서만 정확도가 높고 전혀 보지 못한 데이터에 대해서는 정확도가 현저히 낮게 되는 경우는 Overfit이다.

<span style="background-color: skyblue">Underfit의 경우 training시 정확성은 떨어지지만 test에서 오차범위가 크지 않게 예측을 할 수 있지만, Overfit의 경우 training시에 정확성은 높지만 test에서 오차범위가 크게 예측을 할 수 가 있다.</span>
<br>
<span style="background-color: skyblue">예를 들어 Underfit인 경우 사과를 맞추는 로봇이 있다고 가정했을 때 '사과는 동그랗고 빨갛다' 라고만 학습시키고 테스트를 했을 때 석류나 자두같이 동그랗고 빨간 과일을 보게되어도 사과라고 예측할 것이다. Overfit의 경우는 '지름이 10cm이며 동그랗고 빨간색이다' 라고 학습 시킨 경우에는 자두같이 작지만 빨간 과일에 대해서는 사과라고 예측하지는 않겠지만 10cm가 넘는 사과이거나 초록색 사과인 경우를 사과라고 판단하지 못하는 오류를 범할 수 있다</span>

## Model 성능 평가하는 2가지 방법

### Hold out

> Train-test-split

**Data leakage** training data에는 있지만 test data에는 없어 overfitting된경우 발생하는 문제


### Cross Validation (교차 검증)

> n등분 나누어 test, train을 n번 수행하여 평균을 내어 성능을 테스트한다. <br>
> 보통 10등분으로 함. <br>
> 모든 데이터가 최소 한번은 테스트 데이터로 쓰이도록 한다. <br>
> 데이터가 적을때 대충의 성능평가를 할때 cross_val_score를 사용한다 <br>


data leakage현상을 방지할 수 있다.<br>
데이터의 양이 많으면 매우 느리다는 단점이 있다.

## Model의 성능이 좌우되는 요소 2가지

```
1. 알고리즘
2. 하이퍼 파라미터
```

**복습시간**  21시 10분 ~ 1시 / 2시간 50분



<a id = '23th'></a>
# 2019년 6월 10일 월요일 23th


## map vs apply

```
1. map은 dictionary, 함수 방식 둘다 지원
2. apply는 함수방식만 지원
- apply방식은 args=() 옵션으로 재활용 가능한 함수 방식을 사용할 수 있다
```

<br>

## count vs size

```
count는 미싱데이터를 포함하지 않고
size는 포함한다
```

<br>

### count

```python
a = [1,1,1,2,2,3,4]
b = (1,1,1,2,2,3,4)

a.count(1)
b.count(2)

: 3
  2
```

### size

```python
import numpy as np
c = np.arange(10)
c.size

: 10
```

## cut & qcut

### cut

> 최저값과 최대값의 간격을 n등분하여 나눔

```python
import pandas as pd
import numpy as np

a = np.array([[0,0,2],[0,0,10],[0,0,20],[0,0,49],[0,0,30],[10,11,100]])
x=pd.DataFrame(a)
x.rename({0:'x',1:'y',2:'z'}, axis=1, inplace=True)
pd.cut(x.z,2)

:
0    (1.902, 51.0]
1    (1.902, 51.0]
2    (1.902, 51.0]
3    (1.902, 51.0]
4    (1.902, 51.0]
5    (51.0, 100.0]
```

### qcut

> 전체 데이터 갯수에서 n%로 나눔

```python
import pandas as pd
import numpy as np

a = np.array([[0,0,2],[0,0,10],[0,0,20],[0,0,49],[0,0,30],[10,11,100]])
x=pd.DataFrame(a)
x.rename({0:'x',1:'y',2:'z'}, axis=1, inplace=True)
pd.qcut(x.z,2)

:
0    (1.999, 25.0]
1    (1.999, 25.0]
2    (1.999, 25.0]
3    (25.0, 100.0]
4    (25.0, 100.0]
5    (25.0, 100.0]
```

## Discriminative  vs Generative

> 분류하여 예측 하는 모델에는 두 가지 방식이 있다. Discriminative, Generative

<br>

### Discriminative

> 입력 데이터들이 있을때 label data를 구별해내는 방식

> 어떤 입력값(input) x가 주어졌을 때 그 결과값(label) y일 확률을 알아내는 것

![discriminative](https://user-images.githubusercontent.com/33630505/59189720-d4f9e880-8bb5-11e9-97e4-69ec7a2a5d09.JPG)

<span style="background-color: skyblue">대표 알고리즘</span>
```
1. Logistic Regression
2. Conditional Random Field
3. Support Vector Machine
4. Linear Regression
```

**장점** 데이터가 충분할 경우 성능이 좋음


**단점** 데이터가 실제 어떤 모습인지 본질을 이해하기 어려움


<br>

<hr>
#### SVM(Support Vector Machine)

> SVM은 수학적으로 증명 가능하고 초평면을 경계로 분류하는 알고리즘 이라고 볼 수 있다 <br>
> 선형, 비선형 둘다 성능 좋지만 최적화를 고려 안해 속도가 느리다는 단점이 있다

<br>

![svm](https://user-images.githubusercontent.com/33630505/59195422-e8617f80-8bc6-11e9-8f3e-e05d569ec4d9.JPG)

<hr>

### Generative

> 입력값과 결과값이 주어질때, 일정한 분포 규칙속에 존재한다는 가정을 한다.

> 관측 데이터 결합확률 분포를 통해 확률 모델을 만들어낸다. 즉 주어진 데이터를 보고 분포 규칙을 생성해 낸다.

![generative](https://user-images.githubusercontent.com/33630505/59195390-d384ec00-8bc6-11e9-8dc7-dd27c882753f.JPG)

<br>

![generative2](https://user-images.githubusercontent.com/33630505/59195454-fadbb900-8bc6-11e9-955e-51f5048ee8ec.JPG)

<span style="background-color: skyblue">대표 알고리즘</span>
```
1. Naive Bayes
2. Gaussian discriminant Analysis
3. Gaussian Mixture Model
```

**장점** 데이터 자체의 특성을 파악하기에 좋다, 데이터를 생성해 새로운 결과물을 얻어낼 수 있다.


**단점** 데이터가 많은 경우 Discriminative에 비해 성능이 떨어 질수 있다.



Generative & Discriminative: [naver blog](https://m.blog.naver.com/PostView.nhn?blogId=2feelus&logNo=221078340870&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
<br>

선형, 비선형 모델 : [blog](https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2-3-7-%EC%BB%A4%EB%84%90-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0/)<br>

## LogisticRegression을 제일처음에 하는 이유

```
LogisticRegression은 데이터가 선형분류로 성능이 좋은지 안좋은지를 보고
데이터가 선형 데이터인가 비선형 데이터인가 판별하는데
기준이 될 수 있기 때문에 시간 절약을 할 수 있다

선형분류와 비선형분류 알고리즘 둘다 성능이 비슷한 경우 선형데이터라고 간주하고
선형분류 알고리즘 위주로 학습시키는데 사용하고

선형분류 알고리즘의 성능이 현저하게 낮은 경우 비선형 데이터라고 간주하고
그때부터는 비선형 알고리즘 위주로 학습시키는데 사용하면 시간을 절약할 수 있다
```

<br>

**복습시간**  18시 30분 ~ 22시 10분 / 총 3시간 40분




<a id = '24th'></a>
# 2019년 6월 12일 수요일 24th


## import를 하지 않고 외부 객체의 메소드를 사용 하는 방법


```python
import seaborn as sns
iris = sns.load_dataset('iris')

$whos
:
Variable   Type         Data/Info
---------------------------------
iris       DataFrame         sepal_length  sepal_<...>n\n[150 rows x 5 columns]
sns        module       <module 'seaborn' from 'C<...>s\\seaborn\\__init__.py'>

dir(iris)
:
['T',
 '_AXIS_ALIASES',
 '_AXIS_IALIASES',
 '_AXIS_LEN',
 ....
 'boxplot',
 'iloc',
 'index',
 'infer_objects',
 'info',
 'insert',
 'interpolate',
 'isin',
 .....
```

> DataFrame 객체는  Pandas 프레임워크에 정의된 클래스이다. 따라서 Pandas를 import하지 않고는 사용할 수 없다.

> 하지만 import seaborn만 했는데 iris 객체가 DataFrame 타입으로 나온다. 어떻게 된것일까?

```shell
!pip install seaborn

Requirement already satisfied: seaborn in c:\users\samsung\anaconda3\lib\site-packages (0.9.0)
Requirement already satisfied: numpy>=1.9.3 in c:\users\samsung\anaconda3\lib\site-packages (from seaborn) (1.16.2)
Requirement already satisfied: scipy>=0.14.0 in c:\users\samsung\anaconda3\lib\site-packages (from seaborn) (1.2.1)
Requirement already satisfied: pandas>=0.15.2 in c:\users\samsung\anaconda3\lib\site-packages (from seaborn) (0.24.2)
Requirement already satisfied: matplotlib>=1.4.3 in c:\users\samsung\anaconda3\lib\site-packages (from seaborn) (3.0.3)
Requirement already satisfied: pytz>=2011k in c:\users\samsung\anaconda3\lib\site-packages (from pandas>=0.15.2->seaborn) (2018.9)
.....
```

> seaborn을 설치하게되면 numpy, scipy, pandas 등 같이 설치하게 된다. 왜냐하면 seaborn을 사용하기 위해서는 모두 필요하기 때문이다.

> 설치가 되었다고 해서 import하지 않고 쓸수 있다는 말은 아니다. seaborn 패키지 자체에서 numpy든 pandas든 import해서 seaborn으로 dataset을 생성하면 DataFrame 형태로 반환하도록 설계되어 있어 DataFrame 객체가 네임스페이스에 들어 있게 되면 DataFrame이 사용할 수 있는 메소드는 전부 사용할 수 있게 되는 것이다.

## 상황에 따른 알고리즘 사용법

![algorithm](https://user-images.githubusercontent.com/33630505/59346577-926b1400-8d4d-11e9-893c-04293ef73f8c.JPG)

## 데이터의 양이 충분한지 판단하는 방법

> 데이터 분석시 info정보만으로 데이터의 양이 충분한지 안한지 가늠이 가지 않을때 Learning curve를 확인하여 데이터 양이 충분한지 판단한다.

>Learning curve란 학습시킬때마다 정확도가 어떻게 달라지는지 추세를 확인하여 training score와 cv score가 만나는 지점 즉, overfitting되기 전 적당한 trade-off 지점을 확인할 수 있는 데이터 양이라고 한다면 데이터가 충분하다는 말

```python
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn_evaluation import plot
!pip install sklearn-evaluation

iris = load_iris()
data = pd.DataFrame(iris.data, columns=list('ABCD'))
target = pd.DataFrame(iris.target, columns=['target'])
iris2 = pd.concat([data, target], axis=1)
knn = KNeighborsClassifier()
train_size, train_score, test_score = learning_curve(knn, iris2.iloc[:,:-1], iris2.iloc[:,-1], cv = 10)
plot.learning_curve(train_score, test_score, train_size)
```

![learning curve](https://user-images.githubusercontent.com/33630505/59351013-699c4c00-8d58-11e9-8ada-647b976d4949.JPG)


## Learning curve & LogisticRegression  
```python
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
data = make_classification(1000,5)

d = pd.DataFrame(data[0])
ta = pd.DataFrame(data[1])
train_size, train_score, test_score = learning_curve(lr, d, ta, cv=10)
plot.learning_curve(train_score, test_score, train_size)
```

![logistic learning curve](https://user-images.githubusercontent.com/33630505/59358328-4f1d9f00-8d67-11e9-83ce-43b6da0756ef.JPG)


## 하이퍼 파라미터 찾기 (GridSearchCV)

> GridSearch를 활용하여 for문을 쓰지 않고 하이퍼 파라미터 찾기

```python
from sklearn.model_selection import GridSearchCV

# iris2는 위에서 다룬 예제를 대체한다
x_train, x_test, y_train, y_test = train_test_split(iris2.iloc[:,:-1], iris2.iloc[:,-1])
para_grid = {'n_neighbors': range(2,21), 'weights':['uniform', 'distance']}
gri = GridSearchCV(KNeighborsClassifier(), para_grid)
gri.fit(x_train, y_train)  # cross validation이기 때문에 전체 데이터로 fit 시켜야함
gri.best_estimator_
gri.best_params_
gri.param_grid
gri.best_score_
pd.DataFrame(gri.cv_results_).T

: GridSearchCV(cv='warn', error_score='raise-deprecating',
       estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform'),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'n_neighbors': range(2, 21), 'weights': ['uniform', 'distance']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=13, p=2,
           weights='distance')   
{'n_neighbors': 13, 'weights': 'distance'}
{'n_neighbors': range(2, 21), 'weights': ['uniform', 'distance']}
0.9821428571428571
```


![gri_results](https://user-images.githubusercontent.com/33630505/59363332-f43c7580-8d6f-11e9-8858-226de6ce3354.JPG)


**LogisticRegression** LogisticRegression알고리즘은 target data가 2개 이상일때만 Learning curve가 가능하다.


**Cross-validation & Learning curve** Cross-validation으로 성능 체크할때 n개로 나누어 체크를 하는데 이때 자동으로 데이터를 섞고나서 평가를 하기 때문에 데이터가 정렬 되어 있어도 섞어서 평가를 한다. 그런데 Learning curve로 학습 추세를 확인 할때는 데이터를 순서대로 학습시키기 때문에 최소 클래스 2개가 필요한 LogisticRegression알고리즘을 사용할 때는 shuffle 옵션을 True로 줘야 한다.


**복습시간**  19시 ~  22시/ 총 3시간





<a id = '25th'></a>
# 2019년 6월 13일 목요일 25th


## Supervised Learning Process

![supervised learning process](https://user-images.githubusercontent.com/33630505/59368687-bba19980-8d79-11e9-91bc-63d9a8d3988e.JPG)


<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Raw Data Collection</span> <br>

<p style = "border: 1.2px solid black; border-radius: 7px; display: block; padding: 10;">데이터 수집, 적합한 데이터 format으로 불러오기.
	    기초 통계분석하기 위해 보통 DataFrame 형태로 불러오거나 변환해준다.</p>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Pre-Processing</span> <br>	    

<p style = "border: 1.2px solid black; border-radius: 7px; display: block; padding: 10;">Tidy Data인지 확인한다.
	    Tidy Data가 아닐 경우 변수는 열로 관측치는 행으로 구성할 수 있도록 melt로 행, 열 변환해준다. </p>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Sampling</span> <br>

<p style = "border: 1.2px solid black; border-radius: 7px; display: block; padding: 10;">Train-Test-Split 하거나 데이터 양이 많지 않아 대략적인 성능을 알고 싶을 때는 Cross Validation. 보통 Big Data를 다룬다는 가정이 있기 때문에 Train-Test-Split을 한다.</p>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Pre-Processing</span> <br>

<p style = "border: 1.2px solid black; border-radius: 7px; display: block; padding: 10;">info를 통해 데이터 양이 충분한지, 열 이름에 공백이나 특수문자는 없는지, 데이터 타입이 모두 숫자인지, 불러드릴 수 있는 크기인지, label data를 포함하고 있는지 등을 체크한다.
	    이때 데이터 양이 충분한지 여부를 확인하고 싶을때는 Learning Curve를 확인한다.
	    데이터 양이 적다고 판단이 되어 데이터 수집을 해야하는데 데이터 수집할 형편이 되지 않는다면 차원 축소를 고려해본다.
	    차원 축소는 Scaling, 수작업 등으로 한다.

	    </p>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Learning Algorithm Training</span> <br>	    


<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Hyperparameter Optimization</span> <br>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Post-Processing</span> <br>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Final Model</span> <br>

## Pandas-Profiling

### 설치

```shell
!pip install pandas-profiling
```

### 예제

```python
from sklearn.datasets import load_wine
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns

data = load_wine()
data1=pd.DataFrame(data.data, columns=data.feature_names)
data2 = pd.DataFrame(data.target, columns=['target'])
data3 = pd.concat([data1,data2], axis=1)
ProfileReport(data3)
```

![overview](https://user-images.githubusercontent.com/33630505/59434512-a59de280-8e26-11e9-8053-1d2431cea98c.JPG)

> ProfileReport를 사용해서 자기만의 전처리 방식을 자동화 할 수도 있다.


## 차원 축소 3가지 방법

```
1. Feature Scaling
2. Feature Selection
3. Dimensionality Reduction
```

### Feature Scaling

#### 13개 차원에서 5개 차원으로 축소
```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=['target'])
wine_data = pd.concat([wine, target], axis=1)

pca = PCA(5)
wine_pca = pca.fit_transform(wine_data.iloc[:,:-1])
wine2 = pd.DataFrame(wine_pca)
wine2_data = pd.concat([wine2, wine_data.target], axis=1)

# 13차원
cross_val_score(KNeighborsClassifier(), wine_data.iloc[:,:-1], wine_data.iloc[:,-1], cv=10)
# 5차원
cross_val_score(KNeighborsClassifier(), wine2_data.iloc[:,:-1], wine2_data.iloc[:,-1], cv=10)
:
array([0.68421053, 0.61111111, 0.66666667, 0.55555556, 0.66666667,
       0.55555556, 0.77777778, 0.66666667, 0.82352941, 0.75      ])

array([0.68421053, 0.61111111, 0.66666667, 0.55555556, 0.66666667,
       0.55555556, 0.77777778, 0.66666667, 0.82352941, 0.75      ])
```

> 차원 축소 전과 축소 후 성능 비교후 성능이 축소 전과 비슷하다면 상관성이 높다는 의미로 차원을 축소해도 괜찮다.

> 데이터의 양이 차원에 비해 작을때 차원 축소로 성능 향상을 하기도 한다.


<span style="background-color:red">밑에 부터는 복습 자세하게 다시하기</span>

## Pipeline

> pipeline은 ...

### Pipeline만드는 두가지 방법

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">Pipeline</span> <br>

<span style = "border: 1.2px solid rgb(45, 164, 164); background-color: rgb(45, 164, 164); color: white">make_pipeline</span> <br>

### Pipeline
```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mino
%matplotlib inline
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

data = load_breast_cancer()
X, y = pd.DataFrame(data.data), pd.DataFrame(data.target, columns=['target'])
cancer = pd.concat([X, y], axis=1)

t = cross_val_score(KNeighborsClassifier(),
                    cancer.iloc[:, :-1],
                    cancer.iloc[:, -1],
                    cv=10)

X_train, X_test, y_train, y_test = train_test_split(cancer.iloc[:, :-1], cancer.iloc[:, -1])
pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())])
pipe.fit(X_train, y_train)

: Pipeline(memory=None,
         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('knn',
                 KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                      metric='minkowski', metric_params=None,
                                      n_jobs=None, n_neighbors=5, p=2,
                                      weights='uniform'))],
         verbose=False)
```

## 표준화

## GridSearchCV + Pipeline 하는 방법


**복습시간**  19시 ~ 22시 / 총 3시간  




<a id = '26th'></a>
# 2019년 6월 14일 금요일 26th

## Unsupervised Learnling

## k-means

> 근처 값의 평균을 내어 n개로 묶는 clustering 방법

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

km = KMeans(3)  # 3가지로 묶는다
vars(km.fit(iris_data.values))  

:
{'n_clusters': 3,
 'init': 'k-means++',
 'max_iter': 300,
 'tol': 0.0001,
 'precompute_distances': 'auto',
 'n_init': 10,
 'verbose': 0,
 'random_state': None,
 'copy_x': True,
 'n_jobs': None,
 'algorithm': 'auto',
 'cluster_centers_': array([[6.85      , 3.07368421, 5.74210526, 2.07105263],
        [5.006     , 3.428     , 1.462     , 0.246     ],
        [5.9016129 , 2.7483871 , 4.39354839, 1.43387097]]),
 'labels_': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
        0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2]),
 'inertia_': 78.85144142614601,
 'n_iter_': 5}
```

k-means : [github blog](https://ratsgo.github.io/machine%20learning/2017/04/19/KC/) <br>
<br>

### k-means로 cluster 성능 파악하기

```python
import numpy as np

iris.target  # target data (정답)
:
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
km.labels_   # cluster로 묶은 답
:
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
       0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2])

np.where(km.labels_==1)  # 0 ~ 49 / 100% 맞춤
:
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       dtype=int64),)
np.where(km.labels_==2)  # 50 ~ 99 / 101,106,112 ~ 149 / 2개 틀림  
:
(array([ 50,  51,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
         64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
         91,  92,  93,  94,  95,  96,  97,  98,  99, 101, 106, 113, 114,
        119, 121, 123, 126, 127, 133, 138, 142, 146, 149], dtype=int64),)
np.where(km.labels_==0)  # 100 ~ 149 / 52, 77 / 14개 틀림
:
(array([ 52,  77, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112,
        115, 116, 117, 118, 120, 122, 124, 125, 128, 129, 130, 131, 132,
        134, 135, 136, 137, 139, 140, 141, 143, 144, 145, 147, 148],
       dtype=int64),)
```

## dbscan

> 묶음 갯수 파악하기

```python
from sklearn.cluster import DBSCAN, dbscan  # 둘다 같은 기능
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
dbs = DBSCAN()
dbs.fit(iris_data.iloc[:,:-1])
vars(dbs.fit(iris_data.iloc[:,:-1]))

:
DBSCAN(algorithm='auto', eps=0.5, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=5, n_jobs=None, p=None)

{'eps': 0.5,
 'min_samples': 5,
 'metric': 'euclidean',
 'metric_params': None,
 'algorithm': 'auto',
 'leaf_size': 30,
 'p': None,
 'n_jobs': None,
 'core_sample_indices_': array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
         13,  16,  17,  19,  20,  21,  23,  24,  25,  26,  27,  28,  29,
         30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  42,  43,
         44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
         58,  61,  63,  65,  66,  67,  69,  70,  71,  72,  73,  74,  75,
         76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  88,  89,
         90,  91,  92,  94,  95,  96,  97,  99, 101, 102, 103, 104, 110,
        111, 112, 115, 116, 120, 121, 123, 124, 125, 126, 127, 128, 132,
        133, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 149],
       dtype=int64),
 'labels_': array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
         1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
        -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,  1,
         1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
         1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       dtype=int64),
 'components_': array([[5.1, 3.5, 1.4, 0.2],
        [4.9, 3. , 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
	......
```

```
min_samples는 영역 안의 최소 데이터 갯수
eps는 영역 크기
```
<br>

## Agglomerative Clustering


### Dendrograms

```python
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(X, 'ward')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix, #
    color_threshold=0,
)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
```

![dendrogram](https://user-images.githubusercontent.com/33630505/59520588-44990c00-8f05-11e9-99bb-06d03b26ff40.JPG)


## mglearn으로 clustering 시각화 해서 보기

### 설치

```shell
!pip install mglearn
```
### k-means방식으로 clustering 하는 과정
```python
import mglearn
mglearn.plot_kmeans.plot_kmeans_algorithm()
```

![kmeans](https://user-images.githubusercontent.com/33630505/59517495-ab66f700-8efe-11e9-8de2-fb47d3d01680.JPG)

### k-means boundaries

```python
import mglearn
mglearn.plot_kmeans.plot_kmeans_boundaries()
```

![boundaries](https://user-images.githubusercontent.com/33630505/59517493-ab66f700-8efe-11e9-87f4-74532ab636a1.JPG)

### agglomerative

```python
import mglearn
mglearn.plot_agglomerative.plot_agglomerative_algorithm()
```

![agglomerative](https://user-images.githubusercontent.com/33630505/59517492-aace6080-8efe-11e9-944c-f61807ea32e0.JPG)

### dbscan

```python
import mglearn
mglearn.plot_dbscan.plot_dbscan()
```

![dbscan](https://user-images.githubusercontent.com/33630505/59519335-9db37080-8f02-11e9-8caa-822cf5e52152.JPG)

## dbscan + k-means

<hr>
<span style="background-color: red">알고리즘 만들기는 다음시간에 계속</span>

## 알고리즘 만들기

### Duck typing 방식
```python
class MyEstimator:
    def __init__(self):
        print('a')
    def fit(self, X,y):
        print('b')

my = MyEstimator()
my.fit(data.iloc[:,:-1],data.iloc[:])

from sklearn.dummy import DummyClassifier
dum = DummyClassifier() # 사람처럼 분류하는 알고리즘
```

### BaseEstimator 상속 방식


**Dummy 알고리즘** Dummy 알고리즘과 내가 만든 알고리즘과 비교해서 성능이 좋지 못하다면 자신만의 알고리즘을 만들 필요가 딱히 없음...




**복습시간**  19시 45분 ~ 24시 / 총 4시간 15분  



<a id = '27th'></a>
# 2019년 6월 17일 월요일 27th

## 영화 추천 모델 만들기

### Collaborative filtering

> 나와 비슷한 사람을 찾아 내가본 영화를 제외한 비슷한 사람이 본 영화 추천

### 필요한 데이터 불러오기

```python
import pandas as pd

data = pd.read_csv('u.data', delimiter='\t', header=None, engine='python', usecols=range(3),names=['user_id','movie_id','ratings'])
items=pd.read_csv('u.item', delimiter='|', header=None, engine='python', usecols=range(3), names=['movie_id','title','year'])

data.head(4)
:
user_id	movie_id	ratings
0	196	242	3
1	186	302	3
2	22	377	1
3	244	51	2

items.head(4)
:
	movie_id	title	year
0	1	Toy Story (1995)	01-Jan-1995
1	2	GoldenEye (1995)	01-Jan-1995
2	3	Four Rooms (1995)	01-Jan-1995
3	4	Get Shorty (1995)	01-Jan-1995
```

### DESCR, README 등 도메인 정보 확인하기


```
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of
	         user id | item id | rating | timestamp.
              The time stamps are unix seconds since 1/1/1970 UTC

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.
```

### 불러온 정보 필요한 형태로 변형하기

```python
# 유저 아이디 + 영화 아이디 + 평점 + 영화 이름 + 개봉년도 DataFrame 만들기
user_movie_rate=pd.merge(data,items)

# user index, item columns로 만들기
user_item = data.set_index(['user_id','movie_id']).unstack().fillna(0)
```


### 회원간의 상관성 보기 (어떤 연관성 전략을 세울지 고민)

```python
# user_item에서 user가 index이기 때문에 corr하기 위해 Transform 해야함
user_item_corr = user_item.T.corr()
```

![user_corr](https://user-images.githubusercontent.com/33630505/59600997-c4a9b680-913d-11e9-983e-e5482a173c99.JPG)


### 연관성이 높은 3명 뽑기 (세부 전략 세우기)

> 회원 번호 2를 나라고 가정

```python
user_item_corr.loc[2].sort_values(ascending=False)[:4]
:
user_id
2      1.000000
701    0.570307
931    0.495166
460    0.485913
Name: 2, dtype: float64
```

### 나와 비슷한 사람 영화 목록 - 나의 영화 목록

```python
# 나의 영화 목록
my_movie_list = user_movie_rate[user_movie_rate.user_id==2]
my_movie_list = my_movie_list.movie_id
my_movie_list=set(my_movie_list)
my_movie_list.__len__()
:
62

# 나와 비슷한 사람 영화 목록
other = user_movie_rate[user_movie_rate.user_id.isin([701]).movie_id.value
other_movie_list = set(other)

# 나와 비슷한 사람 영화 목록 - 나의 영화 목록
reco_movie_to_me = other_movie_list - my_movie_list
```

### 최종 추천 영화 목록 출력하기

```python
reco_movie_to_me=user_movie_rate[user_movie_rate.movie_id.isin(reco_movie_to_me)].sort_values('ratings', ascending=False)

final_reco_movie_to_me = set(reco_movie_to_me.movie_id.values)
final_my_reco_movie
:
{124, 326, 328, 333, 344, 689, 690, 750, 751}

# 최종 추천 영화 목록
list(map(lambda x:set(user_movie_rate.title[user_movie_rate.movie_id==x].values),final_my_reco_movie))
:
[{'G.I. Jane (1997)'},
 {'Conspiracy Theory (1997)'},
 {'Game, The (1997)'},
 {'Amistad (1997)'},
 {'Tomorrow Never Dies (1997)'},
 {'Jackal, The (1997)'},
 {'Seven Years in Tibet (1997)'},
 {'Apostle, The (1997)'},
 {'Lone Star (1996)'}]
```

## Pandas format 대표값 설정 없이 그대로 변형하는 4가지 방법

```
1. stack
2. unstack
3. melt
4. pivot
```

### pivot

```python
import pandas as pd

data = pd.read_csv('u.data', delimiter='\t', header=None, engine='python')
data.rename({0:'user_id',1:'movie_id',2:'ratings'}, axis=1, inplace=True)
data=data.pivot('user_id','movie_id','ratings')
data.fillna(0)
```

![pivot](https://user-images.githubusercontent.com/33630505/59602677-4a2f6580-9142-11e9-93a6-c1cf399baa65.JPG)

## Surprise


### 설치
```shell
!pip install surprise
```

### Surprise를 활용하여 예상 별점 예측하기

```python
from surprise import Dataset, Reader, SVD, KNNBasic
import pandas as pd

data = pd.read_csv('u.data', delimiter='\t', header=None, engine='python', usecols=range(3),names=['user_id','movie_id','ratings'])  

sur_data = Dataset.load_from_df(data, Reader(rating_scale=(1,5)))

kb = KNNBasic()
svd = SVD()

kb.fit(sur_data.build_full_trainset())
svd.fit(sur_data.build_full_trainset())

# {124, 326, 328, 333, 344, 689, 690, 750, 751} 위 예제에서 회원아이디 2인 사람의 영화 추천목록

svd.predict(2,344)
:
Prediction(uid=2, iid=344, r_ui=None, est=3.7619267139014876, details={'was_impossible': False})
svd.predict(2,124)
:
Prediction(uid=2, iid=124, r_ui=None, est=4.160187263892665, details={'was_impossible': False})
kb.predict(2,124)
:
Prediction(uid=2, iid=124, r_ui=None, est=4.065428928759065, details={'actual_k': 40, 'was_impossible': False})
kb.predict(2,344)
:
Prediction(uid=2, iid=344, r_ui=None, est=3.696881271344415, details={'actual_k': 40, 'was_impossible': False})
```

## Scikit으로 예상 별점 예측하기

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(3)
knn.fit(data.iloc[:,:-1],data.iloc[:,-1])
:
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=None, n_neighbors=3, p=2,
          weights='uniform')

knn.predict([[2,344]])
:
array([3.33333333])

knn.predict([[2,124]])
:
array([4.])

#####################################
knn = KNeighborsRegressor(40)
knn.fit(data.iloc[:,:-1],data.iloc[:,-1])
:
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=None, n_neighbors=40, p=2,
          weights='uniform')


knn.predict([[2,344]])
:
array([3.075])

knn.predict([[2,124]])
:
array([3.9])
```

## plot_knn_regression (mglearn)

```python
import mglearn

mglearn.plot_knn_regression.plot_knn_regression()
```

![knn_regression](https://user-images.githubusercontent.com/33630505/59603791-49e49980-9145-11e9-9970-0ea5d8318d34.JPG)

## recommendation.pdf 내용 추가

**복습시간**  19시 10분 ~ 21시 17분 / 총 2시간 7분



<a id = '28th'></a>
# 2019년 6월 18일 화요일 28th


## Surprise vs Scikit

### 차이점 2가지

```
1. Train_test_split
2. 평가척도
```
```
Scikit에서는 Train_Test_Split으로 데이터를 나누었지만 Surprise에서는 Fold로 랜덤하게 쪼개준다.
그리고 Scikit에서 평가척도는 score하나 뿐이었지만 Surprise에서는 평가척도로 여러가지가 있다.
예를 들어 rmse(root mean square error) => 평균 제곱근 편차

Fold => train,test default로 5쌍으로 쪼개어진 generator를 반환한다.
```


### Surprise 예제

```python
import pandas as pd
from surprise import SVD, KNNBasic, Dataset, Reader, dump
from surprise.accuracy import rmse

data = Dataset.load_builtin('ml-100k')

for trainset, testset in data.folds():
    algo_knn.fit(trainset)
    predictions_knn = algo_knn.test(testset)
    rmse(predictions_knn)
:
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9753                       
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9685
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9870
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9858
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9739

# 평균 제곱근 편차가 0.9739면 어떻다는 거니... 생각해보자..
```


## os vs sys

> os는 파일관련 처리할 때 사용하고 운영체제 내 폴더파일을 다룰때도 사용한다. <br>
> 참고로 os는 위험한애임.. <br>
> sys는 파이썬 관점에서 경로를 확인할때 등에 사용되는 모듈 이다. <br>
> 자세한건 더 공부하면서 추가해보자.

```python
import os

os.path.expanduser
: <module 'ntpath' from 'C:\\Users\\SAMSUNG\\Anaconda3\\lib\\ntpath.py'>

import sys

sys.path
: ['C:\\Users\\SAMSUNG\\Anaconda3\\lib\\site-packages\\win32\\lib',
 'C:\\Users\\SAMSUNG\\Anaconda3\\lib\\site-packages\\Pythonwin',
 'C:\\Users\\SAMSUNG\\Anaconda3\\lib\\site-packages\\IPython\\extensions',
 'C:\\Users\\SAMSUNG\\.ipython']
```


## Validation_curve

> GridSearchCV로 하이퍼 파라미터를 찾을때 같이 사용함으로써 적절한 하이퍼 파라미터를 찾기 위해 참고하면 좋다.

```python
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn_evaluation import plot

knn = KNeighborsClassifier()
train_scores, test_scores=validation_curve(knn, iris.iloc[:,:-1], iris.iloc[:,-1], 'n_neighbors', [3,4,5,6,7], cv=10)
plot.validation_curve(train_scores, test_scores, [3,4,5,6,7], 'n_neighbors')
```

![validation_curve](https://user-images.githubusercontent.com/33630505/59681973-2a667300-9210-11e9-8e1e-384e6233675d.JPG)


## Statsmodel로 regression분석하기

> Linear regression 분석은 머신러닝에서 해설분야를 담당하고 예측하는데 쓰지는 않는다.

### R방식

#### 설치
```shell
!pip install statsmodels
```

#### 예제

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset("Guerry", "HistData").data

results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)',data=data).fit()
results2 = smf.ols('Lottery ~ Literacy + Instruction',data=data).fit()

results.summary()
results2.summary()
```

#### results summary
![summary](https://user-images.githubusercontent.com/33630505/59684068-4704aa00-9214-11e9-8bd1-4d6417b831f0.JPG)

#### results2 summary
![summary2](https://user-images.githubusercontent.com/33630505/59684100-5552c600-9214-11e9-97e9-9b89f399f6c2.JPG)

### Python 방식

```python
import numpy as np
import statsmodels.api as sm

nobs = 100
X = np.random.random((nobs, 2))
X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
Iy = np.dot(X, beta) + e

results = sm.OLS(Iy, X).fit()

print(results.summary())
```

![python_summary](https://user-images.githubusercontent.com/33630505/59684963-102f9380-9216-11e9-8603-4d2679cf351c.JPG)


**복습시간**    19시 ~ 22시  / 총 3시간





<a id = '29th'></a>
# 2019년 6월 19일 수요일 29th


## 버전관리 2가지 방법

```
1. version-information
2. watermark
```

### version-information

```python
# 설치 방법
!pip install version-information

%load_ext version_information   # import 처럼 version_information을 쓰겠다고 명시해주는 구문

%version_information numpy, pandas, seaborn, scikit-learn, statsmodels # numpy, pandas, seaborn, 등의 버전 명시
```

> watermark 방식보다 이쁘게 나온다

![version_information](https://user-images.githubusercontent.com/33630505/59765932-1be48e00-92da-11e9-92f5-29c8becce363.JPG)


### watermark

```python
%load_ext watermark

%watermark -a 지혁 -d -p numpy,pandas,seaborn
```

> version_information은 한글이 깨지지만 watermark는 한글도 지원한다

![watermark](https://user-images.githubusercontent.com/33630505/59765934-1be48e00-92da-11e9-9e6f-08ea3a07d2a7.JPG)


## Feature-selection

> pre-processing의 일종으로 column을 줄여야겠다는 판단이 들었을때 하는 전처리. <br>
> 성능을 높이기 위한 전처리로, 연산 속도를 향상 시키는 방법으로 사용한다. <br>
> 이때 정확도 성능을 낮추지 않는 선에서 feature-selection을 진행한다.

### 3가지 방식
```
1. Filter
2. Embeded
3. Wrapper
```

### Filter

> 통계값을 보고 경험적으로 도메인 지식을 통해 column을 걸러낸다.

#### 예시

```python
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# 데이터 불러와서 DataFrame 형태로 만들기
data = load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=['target'])
boston_target = pd.concat([boston, target], axis=1)

# 기초 통계분석 생략

# pairplot으로 clustering 경향 살피거나 도메인 지식 활용하여 영향력이 가장 없는 column 걸러내기

boston_target_raw=boston_target.copy() # 원본 데이터 copy해두기
boston_target.drop(columns=['AGE'])  # 가구당 나이는 집값에 영향이 크지 않다고 판단하여 걸러 내본다.
cross_val_score(LinearRegression(), boston_target.iloc[:,:-1], bost_target2.target, cv=10).mean()

: 0.20252899006055775

# 원본 데이터의 정확도
cross_val_score(LinearRegression(), boston_target_raw.iloc[:,:-1], bost_target_raw.target, cv=10).mean()

: 0.20252899006055775
```

> AGE column을 걸러냈을 때와 걸러내기 전의 정확도가 같기 때문에 age는 영향력이 없는 column! <br>
> 따라서 빼도 되는 feature!

### wrapper

> 통계값과 머신러닝 기법을 동시에 사용하여 기준을 두고 ranking을 구해 n개 column 뽑는 방법.


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfe = RFE(model, 7)
X_rfe = rfe.fit_transform(boston_target_raw.iloc[:,:-1], boston_target_raw.target)  # filter 예시에 있는 boston data 사용
X_rfe
:

array([[ 0.   ,  0.538,  6.575, ...,  1.   , 15.3  ,  4.98 ],
       [ 0.   ,  0.469,  6.421, ...,  2.   , 17.8  ,  9.14 ],
       [ 0.   ,  0.469,  7.185, ...,  2.   , 17.8  ,  4.03 ],
       ...,
       [ 0.   ,  0.573,  6.976, ...,  1.   , 21.   ,  5.64 ],
       [ 0.   ,  0.573,  6.794, ...,  1.   , 21.   ,  6.48 ],
       [ 0.   ,  0.573,  6.03 , ...,  1.   , 21.   ,  7.88 ]])

model.fit(X_rfe, boston_target_raw.target)
:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)

vars(rfe)

:
{'estimator': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
          normalize=False),
 'n_features_to_select': 7,
 'step': 1,
 'verbose': 0,
 'estimator_': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
          normalize=False),
 'n_features_': 7,
 'support_': array([False, False, False,  True,  True,  True, False,  True,  True,
        False,  True, False,  True]),
 'ranking_': array([2, 4, 3, 1, 1, 1, 7, 1, 1, 5, 1, 6, 1])}
```

### Embeded

> 알고리즘으로 자동으로 영향력이 어느 정도인가 분류 해주는 방법.


```python
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

iris = sns.load_dataset('iris')
dt = DecisionTreeClassifier()
dt.fit(iris.iloc[:,:-1], iris.iloc[:,-1]) # classification에 한정해서 숫자로 바꾸지 않았을때 자동으로 바꿔줌
:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

dt.predict([[3,3,3,3]])
: array(['virginica'], dtype=object)

dt.feature_importances_
: array([0.        , 0.01333333, 0.06405596, 0.92261071])  # 각각의 숫자는 영향력의 크기를 나타낸다
```

## Ensemble

> 여러가지 알고리즘을 동시에 사용하여 최적의 성능을 낼수 있는 알고리즘을 생성한다

### RandomForest

> 랜덤포레스트는 분류, 회귀 분석 등에 사용되는 앙상블 학습 방법의 일종으로, <br>
> 훈련 과정에서 구성한 다수의 결정 트리로부터 분류 또는 평균 예측치를 출력함으로써 동작한다.<br>
> 성능이 좋고 overfitting이 잘 안일어난다. <br>

<br>

```python
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(iris.iloc[:,:-1], iris.iloc[:,-1])

: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf.feature_importances_
: array([0.03122967, 0.02095218, 0.57202362, 0.37579453])
```

## MLxtend

### 설치

```shell
!pip install mlxtend
```

## Staking

## Data부터 Model 학습까지




## Cross-validate 3가지

```
1. Cross_val_score
2. Cross_validate
3. Cross_val_predict
```

## Fit_transform 하는 3가지

```python
1. pre-processing
2. feature-extraction
3. RFE
```

## Column 줄이는 3가지 방법

```
1. filter
2. PCA
3. RFE
```

**복습시간**    18시 45분 ~  22시 20분 / 총 3시간 35분  
