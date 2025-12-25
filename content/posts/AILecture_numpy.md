---
title: AI 이노베이션 스퀘어 수업(기본반) - numpy
date: 2019-04-29
draft: false
description: AI 이노베이선 스퀘어에서 배운 AI 공부 정리
categories:
- AI
- Numpy
tags:
- AI
- Lecture
- Numpy
slug: AILecture_numpy
---


<a id = '12th'></a>
# 2019년 5월 20일 월요일 열두번째 수업

## 정보의 진화 단계

![dikw](https://user-images.githubusercontent.com/33630505/58008628-c1fc7700-7b27-11e9-9e69-6f39608cd81d.JPG)


DIKW : [blog](http://blog.naver.com/PostView.nhn?blogId=gracehappyworld&logNo=221481622524&categoryNo=17&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search)

## 인공지능의 시작과 배경

Physical Labor과 Cognitive Labor의 한계를 극복하기 위해 인공지능으로 자동화 시키는 분야가 발달하게 됨 <br>
그러나 요즘은 보통 물리적 노동보다는 인지적 관점에서 관심이 쏠리고 있다 <br>
아직 인공지능은 이해하는 능력은 부족하지만 인식하는 능력은 사람의 영역 그 이상까지 왔다 <br>

### 지능

지식을 <br>

1. 이해
2. 인식
3. 추론
4. 학습
5. 생성
6. 해결
7. 결정
<br>
할 수 있는 능력

## 인공지능
```
'인간을 대체할 수 있는 기계 또는 지능을 갖춘 존재로부터 의사소통, 상황의 상관관계 이해 및
 결론 도출 등 인간의 행동을 모방할 수 있는 기술'

좁은 의미에서 기계학습이라 볼 수 있다
기계학습은 데이터를 넣어주면 프로그래밍된 논리나 규칙을 바탕으로
스스로 학습하여 문제해결을 하는 알고리즘을 생성한다

deep learning은 인간 신경망을 모델화하여 스스로 데이터 세트를 예측하는 기술이다
deep learning은 인식분야에서 정확도가 높지만 인식이외에 기능은 좋지 않다
deep learning은 정형화 데이터에서 성능이 좋지 않고 비정형 데이터에서 성능이 좋다
```

### AI, Machine learning, deep learning

![ai](https://user-images.githubusercontent.com/33630505/58009466-7f3b9e80-7b29-11e9-9515-20b4583abaf6.JPG)


### 인공지능 영역의 분류

![ai](https://user-images.githubusercontent.com/33630505/58367430-26e61180-7f1a-11e9-97a3-9c163f0b52d4.JPG)

## 인공지능의 문제해결 전략

```
인간의 인지적 작업을
어떻게 Computing Model로 만들어내고
그것을 Machine에서 구현하여
그 작업을 자동으로 효율적으로 할 수 있게 할 것인가?

Computing Model
- Theory of computation
  컴퓨터 과학의 한 갈래로, 어떤 문제를 컴퓨터로 풀 수 있는지, 또
  얼마나 효율적으로 풀 수 있는지
- Programmable
```

## 파이썬은 연산속도가 느린데 도대체 왜 파이썬으로 AI를 하는가?

### Numpy가 있기 때문에!

<span style="background-color:rgb(56, 188, 182)">Numpy는 속도가 빠르고 사용하기가 쉽다! </span><br>
Numpy는 벡터 연산, 행렬 연산을 효율적으로 쉽게 만들 수 있다<br>  
Numpy는 속도 개선의 최적화를 하지 않아도 되기 때문에 AI에서 Numpy를 쓰는 것이다 <br>
<br>


## Numpy

> Numarray와 Numeric이라는 오래된 Python 패키지를 계승해서 나온 수학 및 과학 연산을 위한 파이썬 패키지이다.

<br>

### Numpy 속도가 빠른 이유

```python
1. C나 Fortran으로 만들어져 속도가 빠르다
2. Array기반으로 처리하기 때문에 속도가 빠르다
3. 데이터를 1열로 저장해, 효율적인 자료구조 형태를 갖기 때문에 빠르다
4. Homogeneous한 Type만을 저장하기 때문에 타입 체크 비용이 들지 않아 빠르다
5. 데이터를 메모리에 한번에 올려 처리하기 때문에 속도가 빠르다
6. 데이터 구조가 Structured array방식이기 때문에 데이터 접근이 빠르다
```

### Numpy는 벡터 기반이다

- 1차 Vector (Numpy에서 vector는 방향이 없다고 간주)
- 2차 Matrics  
- 3차 Tensor

<br>


### python 속도 개선을 위한 방법

```
1. Computing Power
- GPU
- Parallel Computing

2. Compiler
- Cython
- PyPy ....

3. Library
- Numpy

4. Algorithm/ Data Structure
```

<br>

### Vectorization

```
loop없이 벡터연산으로 속도 향상을 하는 방법
요즘은 cpu자체에서 vector processor를 지원
함수형 패러다임 + 선형대수 기법
```

![array](https://user-images.githubusercontent.com/33630505/58010431-4ef4ff80-7b2b-11e9-81fb-7fb7bba0cd19.JPG)
<br>

**Numpy Tip1** python list와 numpy list는 차이가 있다. python은 linked list , type check로 인해 속도가 느리고, numpy 에서는 type이 통일되어 있어 속도가 빠르다

![list](https://user-images.githubusercontent.com/33630505/58010552-8bc0f680-7b2b-11e9-9c2b-ff3cb5012eaa.JPG)

<br>

## Python 문법으로 벡터화 하기 vs Numpy

```python
def x(a,b):
    return [i+j for i,j in zip(a,b)]
x([1,2,3],[4,5,6])
: [5, 7, 9]

@np.vectorize
def z(a,b):
    return a + b
z([1,2,3],[4,5,6])
: array([5, 7, 9])
```

<br>

### Numpy 사용하기

```python
import numpy as np

a = np.array(0)
b = np.array([1,2,3])
c = np.array([[1,2],[3,4]])
d = np.array([[[1,2],[3,4],[5,6]]])
a
b
c
d
type(a)

: array(0)
  array([1,2,3])
  array([[1, 2],
       [3, 4]])
  array([[[1, 2],
        [3, 4],
        [5, 6]]])
  numpy.ndarray

a = np.arange(5,25).reshape(4,5)

np.max(a)
np.min(a)

: 24
  5

np.argmax(a)
np.argmin(a)  # Function 방식
: 19
  0

a.argmax()   # Method 방식
: 19
```
<br>

### Factory Method

> 객체를 만들어내는 부분을 서브 클래스로 위임해 캡슐화 하는 패턴 <br>
> 타입에 따라 다르게 동작하고 싶을때 사용하는 패턴이다
<br>

```python
import numpy as np

np.array([1,2,3])
: array([1, 2, 3])

np.array((1,2,3))   # 메소드 방식(Factory Method)
: np.array(['a', 1])

np.ndarray(['a',1]) # 인스턴스 방식(Homogeneous한 타입의 mutable이기 때문에 타입이 다르면 에러가 난다)
: TypeError

np.ndarray(shape=(2,2), dtype=float, order='F') # 인스턴스 방식은 랜덤으로 값이 채워진다
: array([[1.49769904e-311, 0.00000000e+000],
       [0.00000000e+000, 5.02034658e+175]])       
```
<br>

### Endianness

> 컴퓨터의 메모리와 같은 1차원 공간에 여러 개의 연속된 대상을 배열하는 방법 <br>
> 여기서 바이트를 배열하는 방법을 Byte order라고 한다

![endian](https://user-images.githubusercontent.com/33630505/61672256-e510f600-ad25-11e9-8c01-b43f98be8de0.png)

```python
import numpy as np

np.array(['a', 1]) # little-endian
: array(['a', '1'], dtype='<U1')

dt = np.dtype(">i4") # big-endian
dt.byteorder
: '>'

a = np.array(['a', 1])
a.flags
: C_CONTIGUOUS : True  # C 저장 방식
  F_CONTIGUOUS : True  # Fortran 저장 방식
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False

b = np.array([[1,2],[3,4]], order = "C")
b.flags
: C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False

c = np.array([[1,2],[3,4]], order = "C")
c.flags
: C_CONTIGUOUS : False
  F_CONTIGUOUS : True
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
```
<br>

#### Big-endian

> 큰 단위가 앞에 오는 경우 <br>
> 최상위 바이트 (MSB - Most Significant Byte)부터 차례로 저장하는 방식

![big_endian](https://user-images.githubusercontent.com/33630505/68069380-f8127800-fda2-11e9-8309-5ac9655162c4.JPG)
<br>



#### Little-endian

> 작은 단위가 앞에 오는 경우 <br>
> 최하위 바이트 (LSB - Least Significant Byte)부터 차례로 저장하는 방식

![little-endian](https://user-images.githubusercontent.com/33630505/68069383-02cd0d00-fda3-11e9-98b6-bffa911e22d2.JPG)
<br>

### Big-endian VS Little-endian

```
빅 엔디언은 사람이 숫자를 읽고 쓰는 방법과 같기 때문에 디버깅 과정에서 메모리의 값을 보기 편하다는 장점이 있다.
ex) 0x12345678 => 12 34 56 78로 표현

리틀 엔디언은 메모리에 저장된 값의 하위 바이트들만 사용할 때 별도의 계산이 필요 없다는 장점이 있다.
ex) 32비트 숫자인 0x2A(16진수)를 표현하면 2A 00 00 00 가 되는데, 하위 바이트를 사용하려고 한다면 앞의 한 바이트만 떼어 내면 된다.
    (빅 엔디언에서는 하위 바이트를 얻기 위해서는 3바이트를 더해야 한다는 단점이 있다)    

보통 변수의 첫 바이트를 그 변수의 주소로 삼기 때문에 이런 리틀 엔디언의 성질은 종종 프로그래밍을 편하게 해준다.
또한 가산기가 덧셈을 하는 과정은 LSB로부터 시작하여 자리 올림을 계산해야 하므로 리틀 엔디언에서 가산기 설계가 조금 더 단순해진다.
(오늘날의 프로세서는 여러개의 바이트를 동시에 읽어들여 동시에 덧셈을 수행하는 구조를 갖고 있어 사실상 차이가 없다)

※ 엔디안 방식은 데이터를 전송하는 네트워크 층에서 중요하게 여겨진다.
   서로 다른 방식의 데이터 저장방식을 갖고 통신을 하게되면 엉뚱한 값을 주고 받기 때문이다.

빅 엔디언 =>  Unix의 Risc계열의 프로세서가 사용하는 바이트 오더링
             네트워크에서 사용하는 바이트 오더링
	     앞에서부터 스택에 PUSH
	     비교연산에서 리틀 엔디언보다 속도가 빠르다
리틀 엔디언 => Intel 계열의 프로세서가 사용하는 바이트 오더링
              뒤에서부터 스택에 PUSH
	      계산연산에서 빅 엔디언 보다 속도가 빠르다
```
<br>

출처: [tistory](https://genesis8.tistory.com/37), &nbsp; [위키백과](https://ko.wikipedia.org/wiki/%EC%97%94%EB%94%94%EC%96%B8)

<br>

### 특수 행렬 만들기

```python
import numpy as np

# 영행렬 만들기
z = np.zeros([3,3])
z
: array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])

# 단위행렬(항등행렬) 만들기
y = np.eye(3)
y
: array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

# 전치행렬 만들기
t = np.array([[1,2,3],[4,5,6],[7,8,9]])
t.T
: array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]])

# 일행렬 만들기
o = np.ones((5,4))
o
: array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])

# 원하는 숫자로 행렬 채우기
f = np.full((3,3),3)
f
: array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]])

# 원하는 행렬 shape 복사해서 일행렬 만들기
l = np.ones_like(f)
l
: array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]])

# 대각행렬 만들기

np.diagonal([[1,2],[3,4]])
: array([1, 4])

# 상부 삼각행렬

np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
: array([[1, 2, 3],
       	[0, 5, 6],
       	[0, 0, 9],
       	[0, 0, 0]])

# 하부 삼각행렬

np.tri(4)
: array([[1., 0., 0., 0.],
       	[1., 1., 0., 0.],
       	[1., 1., 1., 0.],
       	[1., 1., 1., 1.]])

np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
: array([[ 0,  0,  0],
       	[ 4,  0,  0],
	[ 7,  8,  0],
       	[10, 11, 12]])

np.linspace(0,30)
: array([ 0.        ,  0.6122449 ,  1.2244898 ,  1.83673469,  2.44897959,
          3.06122449,  3.67346939,  4.28571429,  4.89795918,  5.51020408,
          6.12244898,  6.73469388,  7.34693878,  7.95918367,  8.57142857,
          9.18367347,  9.79591837, 10.40816327, 11.02040816, 11.63265306,
          12.24489796, 12.85714286, 13.46938776, 14.08163265, 14.69387755,
          15.30612245, 15.91836735, 16.53061224, 17.14285714, 17.75510204,
          18.36734694, 18.97959184, 19.59183673, 20.20408163, 20.81632653,
          21.42857143, 22.04081633, 22.65306122, 23.26530612, 23.87755102,
          24.48979592, 25.10204082, 25.71428571, 26.32653061, 26.93877551,
          27.55102041, 28.16326531, 28.7755102 , 29.3877551 , 30.        ])

np.logspace(1,100)
: array([1.00000000e+001, 1.04811313e+003, 1.09854114e+005, 1.15139540e+007,
         1.20679264e+009, 1.26485522e+011, 1.32571137e+013, 1.38949549e+015,
         1.45634848e+017, 1.52641797e+019, 1.59985872e+021, 1.67683294e+023,
         1.75751062e+025, 1.84206997e+027, 1.93069773e+029, 2.02358965e+031,
         2.12095089e+033, 2.22299648e+035, 2.32995181e+037, 2.44205309e+039,
         2.55954792e+041, 2.68269580e+043, 2.81176870e+045, 2.94705170e+047,
         3.08884360e+049, 3.23745754e+051, 3.39322177e+053, 3.55648031e+055,
         3.72759372e+057, 3.90693994e+059, 4.09491506e+061, 4.29193426e+063,
         4.49843267e+065, 4.71486636e+067, 4.94171336e+069, 5.17947468e+071,
         5.42867544e+073, 5.68986603e+075, 5.96362332e+077, 6.25055193e+079,
         6.55128557e+081, 6.86648845e+083, 7.19685673e+085, 7.54312006e+087,
         7.90604321e+089, 8.28642773e+091, 8.68511374e+093, 9.10298178e+095,
         9.54095476e+097, 1.00000000e+100])

# Random
np.empty((3,3))
: array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
         [0.00000000e+000, 0.00000000e+000, 4.36754031e-321],
         [8.70018274e-313, 6.79038653e-313, 1.24610994e-306]])
```

**array** 는 몇차원 데이터인지 통칭하는 단어이다. 그리고 return 값에 array가 나오면 Numpy형태라는 뜻. 파이썬에서 사용하는 형태를 Numpy로 바꿔준다. 벡터를 만드는 방식이기도 하다



### 행렬 연산

```python
import numpy as np

a = np.array([1,2,3,4,5]) # broadcasting
a + 3
: array([4, 5, 6, 7, 8])

a.dot(a) # 내적
: 55

np.sum(a)
: 15

t = np.array([[1,2,3],[4,5,6],[7,8,9]])
np.sum(t, axis = 1) # 행연산
: array([ 6, 15, 24])

np.sum(t,axis=0) # 열연산
:array([12, 15, 18])

A = np.array([[1,2],[3,4]])
B = np.array([[1,2],[3,4]])

A @ B  # At sign 연산자 (행렬곱)
: array([[ 7, 10],
       [15, 22]])
```
<br>

### Python, Numpy 속도 비교

```python
%timeit np.sum(np.arange(10000000))
: 24.6 ms ± 2.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit sum(range(10000000))
: 352 ms ± 6.34 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

<br>

### 데이터 정보 확인하는 방법

```
1. dtype
2. size
3. shape
4. ndim
```
```python
import numpy as np
a = np.arange(10)

a.dtype
a.size
a.shape
a.ndim

: dtype('int32')
  10
  (10,)
  1
```

<br>

## Stride

![ndarray](https://user-images.githubusercontent.com/33630505/58012662-f2481380-7b2f-11e9-833e-4966c0e17241.JPG)

```python
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a.dtype
: dtype('int32')
a.strides
: (12, 4)

# 8bit = 1byte
# dtype에서 int 32bit라고 나왔기 때문에 byte로 바꾸면 4byte가 되는데
# 데이터 하나당 4byte를 차지한다고 보면된다
# 따라서 strides에서 맨 앞을 4byte로 나누어주면 그 갯수만큼 하나의 묶음으로 생각한다
# 즉, 한 행이 3개 데이터로 구성 되어 있다는 뜻
```

**stride만 바꾸어 shape 자유자재로 바꾸기**<br>

```python
import numpy as np

a = np.arange(10).reshape(2,5)
a
: array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

a = np.arange(10).reshape(-1,5)
a
: array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
# -1은 자동으로 알아서 하라는 뜻 / 행렬의 크기를 모를때 유용        
```
<br>

## reshape vs resize

```python
a=np.array([[0,1],[2,3]])

# reshape

a. reshape(4,1)
: array([[0],
         [1],
         [2],
         [3]])

a. reshape(1,4)
: array([[0, 1, 2, 3]])

# resize

np.resize(a,(2,3))
: array([[0, 1, 2],
         [3, 0, 1]])

np.resize(a,(1,4))
: array([[0, 1, 2, 3]])

np.resize(a,(2,4))
: array([[0, 1, 2, 3],
         [0, 1, 2, 3]])
```

## 데이터 형태 변환하기

```python
import numpy as np

a.astype('float32')
: array([[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 9.]], dtype=float32)

a.astype('int64')       
: array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]], dtype=int64)   

a.astype('bool')
: array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]])
```

<br>


## array는 sequence type

> sequence type은 indexing, slicing이 가능하다!


```python
import numpy as np

n = np.arange(10).reshape(5,2)
n
: array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])

n[:,1]
: array([1, 3, 5, 7, 9])

n[:][1]
: array([2, 3])

n[1,:]
: array([2, 3])

n[n>3]
: array([4, 5, 6, 7, 8, 9])
```

<br>

### Numpy Indexing

```
1. 일반 indexing
2. 콤마
3. Fancy indexing
4. Masking
5. 조건문 (where)
```

### 예제

```python
import numpy as np

a = np.arange(25).reshape(5,5)
a
: array([[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24]])  

# 일반 indexing

for x in range(3,5):
    for y in range(3,5):
        print(a[x][y], end=" ")
    print()
: 18 19
  23 24

# 콤마

a[3:,3:]
: array([[18, 19],
        [23, 24]])

# Fancy indexing

a[[3,4],3:]
: array([[18, 19],
         [23, 24]])

# Masking

a[(a > 17) & (a < 20) + (a > 22)].reshape(2,2)
: array([[18, 19],
        [23, 24]])

b = np.array([[False,False,False,False,False],
              [False,False,False,False,False],
              [False,False,False,False,False],
              [False,False,False,True,True],
              [False,False,False,True,True]])
a[b].reshape(2,2)
: array([[18, 19],
        [23, 24]])

# 조건문
a[np.where((a > 17) & (a < 20) + (a > 22))].reshape(2,2)
: array([[18, 19],
        [23, 24]])
```

### nditer

```python
import numpy as np

a = np.nditer([1,2,3])
next(a)
: (array(1), array(2), array(3))

b = np.nditer([[1,2],[3,4]])
next(b)
: (array(1), array(3))

c = np.array([[1,2],[3,4]])
d = np.nditer(c)
next(d)
: array(1)
```

`복습 시간` 17시 40분 ~ 19시 / 총 1시간 20분



<a id = '13th'></a>
# 2019년 5월 21일 화요일 13th


## Masking

> True, False를 활용해 인덱싱하는 방법

```python
import numpy as np

a = np.arange(10)

a > 3
: array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
a[a>3]
: array([4, 5, 6, 7, 8, 9])

a[(a > 3) & (a < 8)]
: array([4, 5, 6, 7])

a[[True, True, True, True, True, True, False, False, False, False]]
: array([0, 1, 2, 3, 4, 5])
```

## ix_

<span style="background-color:skyblue">Cartesian product 연산</span><br>

```python
import numpy as np

h = np.arange(25).reshape(5,5)
h
: array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
h[np.ix_([1,3],[0,1,2,3,4])]
:
array([[ 5,  6,  7,  8,  9],
       [15, 16, 17, 18, 19]])
```

## Namedtuple

> 이름 있는 튜플 만들기
sequence tuple 처럼 사용 가능
클래스처럼 이름으로 접근가능

```python
from collections import namedtuple

t = namedtuped('AttendanceSheet',['name','attendance'])
x=t('jh','yes')
x[0]
x[1]
x.name
x.attendance
type(x)

: jh
  yes
  jh
  yes
  __main__.AttendanceSheet
```

## broadcasting

> 벡터연산에서 자동으로 크기 맞춰주는 기법

```python
import numpy as np

a = np.arange(10)
a + 1

: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

![broadcasting](https://user-images.githubusercontent.com/33630505/58095909-e717e500-7c0e-11e9-9adb-d916ca172bf2.JPG)


## ufunc(universal function)

> 범용적인 함수 즉, python, numpy 둘다 있는 함수 but 차이가 있다

### 1개의 배열에 대한 ufunc 함수

```
abs,fabs => 절대값
ceil => 올림
floor => 내림
modf => 정수부분과 소수점 부분 분리
rint => 올림하거나 내림하거나 5를 기준으로
log, log10, log2, log1p => 로그 값 취하기
exp => exponential 지수함수 (정확히 어떻게 계산되는지는 모르겠음)
sqrt => 루트
square => 제곱
isnan => nan인지 체크
isinfinite => 유한한 수안자 체크
logical_not => 모르겠음
sign = > 0을 제외하고 다 1로 반환 (사실 정확하지 않음)
sin, cos, tan => sin, cos, tan값 계산
arcsin, arccos, arctan => 역삼각함수 계산

```
### 2개의 배열에 대한 ufunc 함수

```
add => 각 요소 더하기
subtract => 각 요소 빼기
multiply => 각 요소 곱하기
divide => 각 요소 나눈 값
floor_divide => 각 요소 나눈 몫
mod => 각 요소 나눈 나머지
power => 승 계산 ex) 2,3 => 2의 3 승 : 8
maximum, fmax => 더 큰 값
minimum, fmin => 더 작은 값
greater => 앞 값이 더 크면 True 작으면 False
greater_equal => 앞 값이 크거나 같으면 True 작으면 False
less => greater 반대
less_equal => greater_equal 반대
equal => 같으면 True
not_equal => 다르면 True
copysign => 모르겠음
```

### Python, Numpy ufunc

> python에서는 동시에 사용 못하지만 numpy에서는 한꺼번에 연산 가능

```python
import math
math.sqrt(4)
: 2.0

np.sqrt((4,9))
: array([2., 3.])
```
<hr>

![ufunc1](https://user-images.githubusercontent.com/33630505/58092007-052d1780-7c06-11e9-86ca-8b8a2dae0a74.JPG)
![ufunc2](https://user-images.githubusercontent.com/33630505/58092009-065e4480-7c06-11e9-9dbc-eb9665801c3a.JPG)
![ufunc3](https://user-images.githubusercontent.com/33630505/58092011-078f7180-7c06-11e9-826c-b477930b24a5.JPG)

```python
np.sqrt([4,9])
np.sqrt((4,9))

둘다 가능
```
**Numpy Tip1** mutable 성질이 중요하지 않으면 list, tuple 혼용 가능



### 배열 분할하기, 붙이기

```python
# split (분할하기)

a = np.arange(16).reshape(4,4)
a
: array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

np.hsplit(a,2) # 수평축으로 분할(세로, 사실상 수직) (np.split(a,2,axis=1))
: [array([[ 0,  1],
        [ 4,  5],
        [ 8,  9],
        [12, 13]]),
   array([[ 2,  3],
        [ 6,  7],
        [10, 11],
        [14, 15]])]

np.hsplit(a,(1,2))
: [array([[ 0],
        [ 4],
        [ 8],
        [12]]),
   array([[ 1],
        [ 5],
        [ 9],
        [13]]),
   array([[ 2,  3],
        [ 6,  7],
        [10, 11],
        [14, 15]])]

np.vsplit(a, 2)
: [array([[0, 1, 2, 3],
        [4, 5, 6, 7]]),
   array([[ 8,  9, 10, 11],
        [12, 13, 14, 15]])]

np.s_[a,b]
: (array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9]))

[1,2,3,4,5][2:5]
: [3, 4, 5]

[1,2,3,4,5][slice(1,5)]
: [2, 3, 4, 5]

np.arange(10)[np.s_[2:5]]
: array([2, 3, 4])

# stack (붙이기)

a = np.arange(5)
b = np.arange(5, 10)

np.stack((a,b), axis=1)
: array([[0, 5],
         [1, 6],
         [2, 7],
         [3, 8],
         [4, 9]])

np.stack((a,b), axis=0)
: array([[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]])

np.vstack((a,b))
: array([[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]])

np.hstack((a,b))
: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.column_stack((a,b))   # np.c_[a,b]
: array([[0, 5],
         [1, 6],
         [2, 7],
         [3, 8],
         [4, 9]])

np.row_stack((a,b))      # np.r_[a,b]랑 같음
: array([[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]])
```



<br>

### view & copy

> python은 기본적으로 shallow copy, numpy는 기본적으로 deep copy

```python
# python에서 deep copy하기
import copy

a = [[1,2,3]]
b = copy.deepcopy(a)
a[0][1] = 4
b
a

: [[1, 2, 3]]
  [[1, 4, 3]]

# numpy는 기본적으로 deep copy

a = np.array([[1,2,3],[4,5,6]])
b = a.copy()
a[0][0] = 4
b
a

: array([[1, 2, 3],
       [4, 5, 6]])
  array([[4, 2, 3],
       [4, 5, 6]])     
```

<br>

### ravel & flatten

> Ravel - Bolero (클래식/디지몬 어드벤처 극장판에서 나오는 노래)
ravel은 몇 차원이건 간에 모두 1차원으로 만들어 준다
그리고 view 방식이기 때문에 원래 값을 바꾸기 때문에 주의 해야한다
flatten은 copy 방식

```python
a = np.arange(10).reshape(2,5)

a.ravel()
a.flatten()
: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### newaxis

> return이 None이고 차원을 추가한다
곱하기 할때에도 활용하는 방법이다.(차원을 맞추어 계산해야 하기 때문)

```python
a = np.array([[1,2,3],[4,5,6]])
a
a.shape
: array([[1, 2, 3],
       [4, 5, 6]])
  (2,3)

# z축에 추가
b=a[:,:,np.newaxis]
b
b.shape
: array([[[1],
        [2],
        [3]],

       [[4],
        [5],
        [6]]])
   (2, 3, 1)	## 평면 두개    

# y축에 추가
c=a[:,np.newaxis]
c
c.shape
: array([[[1, 2, 3]],

       [[4, 5, 6]]])
  (2, 1, 3)     ## 평면 두개


# x축에 추가
d=a[np.newaxis,:]
d
d.shape
: array([[[1, 2, 3],
        [4, 5, 6]]])
  (1, 2, 3)     ## 평면 한개
```

<br>

#### elementwise product

```python
a = np.array([[1,1],[0,1]])
b = np.array([[2,0],[3,4]])

a*b
: array([[2, 0],
       [0, 4]])
```


`복습 시간` 18시 30분 ~ 21시 / 총 2시간 30분



<a id = '14th'></a>
# 2019년 5월 23일 목요일 14th

## newaxis 정리

<span style="color:orange">1차원: 방향이 없는 벡터(스칼라)형태의 데이터만 존재, [] 1개</span><br>
```python
import numpy as np

a = np.array([1,2,3])
a.shape

: (3,)  # 3개의 데이터가 하나로 묶여 있다고 생각
```
<span style="color:orange">2차원 : 행렬, 평면, [] 2개</span> <br>

```python
import numpy as np

a = np.array([1,2,3])
a[np.newaxis]     # x축 추가 행기준으로 묶기
a[np.newaxis].shape

: array([[1,2,3]]) # 가장 바깥 [] 소거하고 행갯수 세면 x축 데이터 갯수
                   # 그 다음 안 [] 소거하고 행갯수 세면 y축 데이터 갯수
  (1,3)
a[:,np.newaxis] # y축 추가 열기준으로 묶기 (np.expand_dims(a, 1))
a[:,np.newaxis].shape

: array([[1],      # 가장 바깥 [] 소거하고 열갯수 세면 x축 데이터 갯수
        [2],       # 그 다음 안 [] 소거하고 열갯수 세면 y축 데이터 갯수
        [3]])
  (3,1)
```

<span style="background-color:red">이 경우는 뭐지?</span> <br>

```python
import numpy as np
a = np.array([[1,2],[4,5,6]])
a
: array([list([1, 2]), list([4, 5, 6])], dtype=object)

a = np.arange(27).reshape(3,3,3)

np.swapaxes(a, 0, 2)
: array([[[ 0,  9, 18],
          [ 3, 12, 21],
          [ 6, 15, 24]],

         [[ 1, 10, 19],
          [ 4, 13, 22],
          [ 7, 16, 25]],

         [[ 2, 11, 20],
          [ 5, 14, 23],
          [ 8, 17, 26]]])

np.moveaxis(a, 0, 2)
: array([[[ 0,  9, 18],
          [ 1, 10, 19],
          [ 2, 11, 20]],

         [[ 3, 12, 21],
          [ 4, 13, 22],
          [ 5, 14, 23]],

         [[ 6, 15, 24],
          [ 7, 16, 25],
          [ 8, 17, 26]]])
```

<span style="color:orange">3차원 : 행렬 중첩, 평면 겹쳐서 직육면체처럼 [] 3개 </span><br>

```python
import numpy as np

a = np.array([[1,2,3],[4,5,6]])
a[np.newaxis]
a[np.newaxis].shape

: array([[[1, 2, 3],
        [4, 5, 6]]])   # 가장 바깥 []소거하고 []x2인 행갯수 세면 x축 데이터 갯수
                       # [[1,2,3]] 이라는 평면 1개   
  (1,2,3)              # 그 다음 안 [] 소거하고 []x1인 행갯수 세면 y축 데이터 갯수
                       # 마지막 [] 소거하고 행갯수 세면 z축 데이터 갯수

a[:,np.newaxis]
a[np.newaxis].shape    # 위와 동일

: array([[[1, 2, 3]],

       [[4, 5, 6]]])
  (2,1,3)      

a[:,:,np.newaxis]
a[:,:,np.newaxis].shape #

: array([[[1],
         [2],
         [3]]])
(2, 3, 1)
```


## tile

```python
import numpy as np

a = np.array([1,2,3])

np.tile(a,3)
: array([1, 2, 3, 1, 2, 3, 1, 2, 3])

np,tile(a,(2,3))
: array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
       [1, 2, 3, 1, 2, 3, 1, 2, 3]])

np.tile(a,[2,3])  # duck typing
: array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
       [1, 2, 3, 1, 2, 3, 1, 2, 3]])
```

## 파일 불러오기


### loadtxt, genfromtxt
```python
%%writefile a.csv
1,2,3
4,5,6

: Writing a.csv

x = np.loadtxt('a.csv', delimiter = ',')
x
: array([[1., 2., 3.],
       [4., 5., 6.]])

x = np.genfromtxt('a.csv')
: array([nan, nan])

# loadtxt는 delimiter를 이용해 문자열 구분을 하지 않으면 에러가 나지만
# getfromtxt는 nan이라는 출력값을 주고 에러를 발생시키지 않는다
```

### fromfile

```python
x = np.fromfile('a.csv', sep=',') # \n을 만나면 종료
x

: array([1., 2., 3.])
```


## File

### Flat

> 구조가 있는 파일

```
1. text file  
=> 확장자 상관없이 열 수 있다.
=> 데이터 교환시 유용함
2. binary file
=> 연결프로그램에 의존적
```


```
np.savez()

np.save()
```

### Raw

> 구조가 없는 파일

```
```


## linear algebra


#### WhyPythonIsSlow + open_with 내용 복습

https://docs.scipy.org/doc/numpy/user/quickstart.html 복습

### 설명 보기

```python
import numpy as np

np.lookfor('shape')
: 설명 ~
np.info('shape')
: 설명 ~
```
