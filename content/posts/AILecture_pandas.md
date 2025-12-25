---
title: AI 이노베이션 스퀘어 수업(기본반) - pandas
date: 2019-04-29
draft: false
description: AI 이노베이선 스퀘어에서 배운 AI 공부 정리
categories:
- AI
- Pandas
tags:
- AI
- Lecture
- Pandas
- Data Analysis
slug: AILecture_pandas
---

<a id = '15th'></a>
# 2019년 5월 24일 금요일 15th

## Pandas

> Numpy 기반으로 만들어진 데이터 조작, 분석을 위한 프레임워크 <br>
> Data Wrangling Tool, 데이터를 불러와 합치고, 간단한 전처리하고, 기초통계 분석하는 프레임워크<br>


<span style="color:orange">Pandas로 할 수 있는 2가지</span>
```
1. 기초통계분석 (EDA)
2. 전처리
- 반정형 데이터를 정형데이터로 바꿔준다
```

<br>

ETL vs Munging ETL(Extract Transform Load)는 개발자 입장에서 하는 파이프라인이고 Munging은 통계분석가 입장에서 하는 파이프라인이라고 생각하면 된다.


<br>

## 데이터 종류

```
1. 정형 데이터 : Dataframe 객체에 정확하게 컬럼에 집어 넣을 수 있는 데이터
2. 비정형 데이터
3. 반정형 데이터
```

## 데이터 타입 만드는 방법

### Numpy 방식 (structured array)
```python
import numpy as np

x = np.array([('jihyuk',25,73.0),('thor',35,85.0),('lion',10,30.0)],dtype=[('name','U10'),('age','i4'),('weight','f4')])  

x[0]
: ('jihyuk',25,73.0)
x[0]['name']  # dict의 key값으로 접근
: 'jihyuk'
```

### Python 방식 (namedtuple)

```python
from collections import namedtuple

x = namedtuple('Address',['name','age','weight'])
a = x('jh',25,'73.0')

a.name  # attribute
a.age
: 'jh'
  25
```

## Pandas로 기초통계분석하기

### [첫번째] 데이터 불러들이기

```python
import pandas as pd

data = pd.read_csv('/Users/SAMSUNG/Desktop/개인공부/AI/AI 이노베이션 스퀘어 기본과정/수업 내용/abc.csv',engine='python')
type(data)
: pandas.core.frame.DataFrame

# read 메소드는 flat file 또는 sql format을 dataframe형태로 불러들인다
# 첫번째 인자는 불러올 파일의 경로인데 현재 작업파일과 동일한 위치에 있다면 파일이름만 적어줘도 된다
# engine = 'python' 이나 encoding = 'cp949'를 인자로 넣어주지 않으면 unicodeerror가 뜬다
```
![read method](https://user-images.githubusercontent.com/33630505/58368737-26567680-7f2c-11e9-9581-e21370c90f49.JPG)

**filepath_buffer**는 read_csv 메소드의 첫번째 인자로 파일경로나, url이 올 수 있다  


**Dataframe** 객체는 Numpy에서 structured array방식을 따라 데이터 타입을 생성한다. pandas는 벡터, 행렬연산으로 속도를 빠르게 하기 위해 Numpy방식을 그대로 이어받아 사용한다. 그리고 DataFrame에서 각 열은 단일 데이터 형식만을 저장한다. 따라서 타입체크를 하지 않아 속도가 빠르다. 또한 DataFrame은 dict, attr 두가지 방법으로 모두 접근 가능하다. ex) dataframe.column, dataframe['column']


**Series** 객체는 Dataframe에서 1차원 데이터 한 행이나 한 열, 1차원이기 때문에 방향은 없다. Series는 dataframe 처럼 dictionary 형태로 구성되어 있고 key값으로 index가 자동 생성이 된다.


```python
import pandas as pd

data = np.read_csv('abc.csv', engine='python')
data.values

: array([['절도', ' 129 ', ' 217 ', ..., ' 1 ', ' - ', nan],
       ['불법사용', ' - ', ' - ', ..., ' - ', ' - ', nan],
       ['침입절도', ' 29 ', ' 38 ', ..., ' - ', ' - ', nan],
       ...,
       ['화재예방·소방시설설치유지및안전관리에관한법률', ' - ', ' - ', ..., ' - ', ' - ', nan],
       ['화학물질관리법', ' 1 ', ' - ', ..., ' 3 ', ' - ', nan],
       ['기타특별법', ' 26 ', ' 226 ', ..., ' 4 ', ' - ', nan]], dtype=object)

# dict으로 접근해서 values를 사용하면 numpy format인것을 확인할 수 있다       
```

### Series , Vector 차이점

Series, Vector 둘다 1차원 데이터에 방향도 없지만 Series는 index가 붙는다

<br>


### [두번째] 분석하고 그래프 그리기


### 분석하기전 5가지 확인 사항

> 데이터를 분석하기 전에 분석할 가치가 있는 데이터인지 부터 판단해야 한다!  데이터의 갯수가 충분한지 여부도 체크하고, 내가 불러들일 수 있는 크기의 데이터 양인지 체크하고 편향된 데이터값이 많은지 등등 알고서 분석에 적합한 데이터를 판별해야한다. (데이터가 많으면 많을 수록 분석, 예측시 성능이 좋아진다!

```
1. info : 데이터의 기본 정보를 보여준다
2. describe : 숫자형태인 데이터에 대해서 기본 통계값을 보여준다
3. head : default로 앞에서 5개 데이터만 불러온다 (앞에서부터 보고싶은 데이터 갯수 입력 가능)
4. tail : head와 반대로 뒤에서 부터 데이터를 불러온다
5. sample : 랜덤으로 하나의 데이터를 불러온다
```

### info, describe로 데이터의 숨겨진 의미 찾기

1. column 갯수 확인 => 차원의 저주 고려 <br>
2. 데이터 갯수 확인 => 큰 수의 법칙 고려 <br>
3. 미싱 데이터 찾기 => 미싱데이터를 포함하고 있으면 정확도  <br>
4. 데이터 타입 확인 => 적절한 타입을 썻는지 체크 (category, object는 각각 지원하는 기능이 다르다) <br>


<br>

#### data (도로교통공단_시도_시군구별_도로형태별_교통사고(2018) 공공데이터)

> pandas로 불러오면 rangeindex가 붙는다

![accident](https://user-images.githubusercontent.com/33630505/58369114-09707200-7f31-11e9-8b4b-cfcc10731401.JPG)


#### info

```python
import pandas as pd

data = pd.read_csv('load.csv', engine = 'python')
data.info()

: <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 2001 entries, 0 to 2000
  Data columns (total 9 columns):
  시도      2001 non-null object
  시군구     2001 non-null object
  도로형태    2001 non-null object
  발생건수    2001 non-null int64
  사망자수    2001 non-null int64
  부상자수    2001 non-null int64
  중상      2001 non-null int64
  경상      2001 non-null int64
  부상신고    2001 non-null int64
  dtypes: int64(6), object(3)
  memory usage: 140.8+ KB

# 교통사고 공공데이터  
```


#### describe

```python
import pandas as pd

data = pd.read_csv('load.csv', engine = 'python')
data.describe()
```
![describe](https://user-images.githubusercontent.com/33630505/58369143-5f451a00-7f31-11e9-8bf4-b9dcac83f844.JPG)


#### head, tail, sample

```python
import pandas as pd

data = pd.read_csv('load.csv', engine = 'python')
data.head(3)
data.tail(3)
data.sample(3) # replace = True 옵션을 주면 복원추출
```

![head](https://user-images.githubusercontent.com/33630505/58369306-78e76100-7f33-11e9-9abc-f94e94aac804.JPG)
![tail](https://user-images.githubusercontent.com/33630505/58369307-7a188e00-7f33-11e9-9254-1fdaa0245e32.JPG)
![sample](https://user-images.githubusercontent.com/33630505/58369313-90bee500-7f33-11e9-8549-92b5635e9a3d.JPG)

#### 표준편차 표로 보기 (boxplot)

![boxplot](https://user-images.githubusercontent.com/33630505/58369325-c237b080-7f33-11e9-9d04-2ec773255130.JPG)

### 왜도(skewness), 첨도(kurtosis)

**왜도** <br>
왜도는 데이터가 대칭이 아닌 정도를 나타낸다 <br>
왜도의 값이 음수이면 오른쪽으로 치우친 정도를 나타내고 <br>
왜도의 값이 양수이면 왼쪽으로 치우친 정도를 나타낸다 <br>
<br>
**첨도** <br>
첨도는 데이터가 중간값의 분포도의 정도를 나타낸다 <br>
첨도의 값이 3보다 작으면 완만한 분포를 나타내고 <br>
첨도의 값이 3보다 크면 뾰족한 분포를 나타낸다 <br>

```python
import pandas as pd

data = pd.read_csv('load.csv', engine = 'python')
data.skew()
data.kurtosis() # kurt

: 발생건수    3.765094
사망자수    3.820251
부상자수    3.778984
중상      3.541237
경상      3.847238
부상신고    5.351853
dtype: float64

발생건수    17.881821
사망자수    18.509412
부상자수    17.783772
중상      15.962331
경상      18.220330
부상신고    40.538785
dtype: float64
```

## 열 뽑는 4가지 방법

```
1. dictionary
2. attribute
3. fancy indexing
4. data type
```

**Pandas Tip1** 데이터 분석시 데이터 조작을 하기위해 할당을 하는 경우에 view방식으로 접근하게되면 원본 데이터도 변경될 수 있으므로 copy방식을 사용해야 한다



## 행 뽑는 방법

```
1.loc  # index 명으로 접근
2.iloc # index 숫자로 접근
```

## 문제해결 그리고 예측

<span style="background-color:orange">많은 데이터 확보 => 기초 통계분석 및 전처리 => 기계학습 및 딥러닝으로 예측 </span>


## Exploratory Data Analysis

> 수집한 데이터가 들어왔을 때, 이를 다양한 각도에서 관찰하고 이해하는 과정. <br>
> 한마디로 데이터를 분석하기 전에 그래프나 통계적인 방법으로 자료를 직관적으로 바라보는 과정이다. <br>

<br>

### nan값 제거
```python
import pandas as pd

data = pd.read_csv('abc.csv',engine='python')
data.iloc[4].dropna()
```


<hr>


`복습 시간`  2시간으로 추정



# 2019년 5월 26일 일요일 Jupyter Notebook 오류

> Numpy 예제 100선을 풀기 위해 파일을 불러오는 도중 해당 파일이 Not trusted 문제가 발생했다

## Not trusted

내가 작성한 파일이 아닌 다른 사람에 의해 만들어진 파일이라서 보안상 문제가 될 수 있어 발생한 오류 인것 같다<br>
그래서 열고자 하는 파일이 믿을만하다는 것을 알려주기 위해서 명령 프롬프트를 통해 신뢰할만한 파일이라고 직접 알려줘야 한다<br>

```shell
jupyter trust name.ipynb
```
위 명령어를 입력하자 not trusted 오류는 발생하지 않았다 <br>

## 페이지를 열기 위한 메모리가 충분하지 않음

크롬 브라우저에서 메모리가 부족하다는 것이다. <br>
그래서 쿠키정보를 삭제해보았다.

**해결!** <br>
그러나 힌트파일이 아닌 정답파일은 파일 자체 내용이 많아서 그런지 아직도 안열린다...


<a id = '16th'></a>
# 2019년 5월 27일 월요일 16th

## 유니콘이 되려면...

![unicon](https://user-images.githubusercontent.com/33630505/58414654-fedfe500-80b6-11e9-950d-03888fd83082.JPG)

**Data Wrangling** Raw data를 또 다른 형태로 수작업으로 전환하거나 매핑하는 과정. 즉, 여러가지 데이터 포멧을 내가 원하는 데이터 포멧으로 전환하여 사용하기 위한 과정. (Data Munging 이라고도 불린다)


## 그래프 그리기

### describe로 나오는 값들 그래프로 그리기
```python
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('file.csv', engine='python')
pd.plotting.boxplot(data)
```
![describe](https://user-images.githubusercontent.com/33630505/58414617-e374da00-80b6-11e9-9e72-168df4140f90.JPG)

### 정규분포가 되는지 확인하는 그래프 그리기

```python
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('file.csv', engine='python')
pd.plotting.scatter_matrix(data)
```
![scatter](https://user-images.githubusercontent.com/33630505/58416173-aeb75180-80bb-11e9-8cc3-5d8f83737d5d.JPG)

## matplotlib inline & notebook

```python
%matplotlib inline
data.boxplot()

%matplotlib notebook
data.boxplot()
```

### inline
![inline](https://user-images.githubusercontent.com/33630505/58416292-0f468e80-80bc-11e9-9789-12627d6fbdd7.JPG)

### notebook
![notebook](https://user-images.githubusercontent.com/33630505/58416310-1bcae700-80bc-11e9-90c0-54f9ba8cd9aa.JPG)


## seaborn으로 그래프 이쁘게 그리기

```python
import seaborn as sns

data = pd.read_csv('file.csv', engine='python')
sns.pairplot(data)
```

![seaborn](https://user-images.githubusercontent.com/33630505/58416408-63ea0980-80bc-11e9-818c-93fbe478439e.JPG)

## Header name 바꾸기 (전처리 과정중 일부)

```python
import pandas as pd

data = pd.read_csv('file.csv', engine='python')
data.rename({0:'sl',1:'sw',2:'pl',3:'pw','class':'class_'},axis=1,inplace=True)

# inplace True하면 자기자신이 바뀜
```

## 짝을 이뤄 그래프 그리기

> 열(column)에 object가 있을 때

```python
import pandas as pd

data = pd.read_csv('file.csv', engine='python')
data.rename({0:'sl',1:'sw',2:'pl',3:'pw','class':'class_'},axis=1,inplace=True)
sns.pairplot(data,hue='class_')
```

![hue](https://user-images.githubusercontent.com/33630505/58416593-115d1d00-80bd-11e9-8b8a-59429b3f8b51.JPG)


## Tidy Data

<kbd>Wide format</kbd> ⇒  <kbd>Long format</kbd>

> 분석하기 좋은 데이터. Tidy data 형태로 만들면 차원도 줄고, 유지보수하기도 좋다

**Tidy Data 특징**

```
1. 각 변수는 개별의 열(column)로 존재한다
2. 각 관측치는 행(row)으로 구성한다
3. 각 표는 단 하나의 관측기준에 의해서 조작된 데이터를 저장한다
4. 만약 여러개의 표가 존재한다면, 적어도 하나이상의 열이 공유되어야 한다
```

> 위 원칙들은 관계형 데이터베이스 원칙과 유사하다

※ 예시

변수 : 키, 몸무게, 성별 <br>
값 : 175, 73, 남자 <br>
관측치 : 사람  (값을 측정한 단위가 되는 기준) <br>

```python
import pandas as pd

data = pd.read_csv('file.csv')
data.melt(['iso2','year'], var_name='sp', value_name='값').dropna()
```
![tidy data](https://user-images.githubusercontent.com/33630505/58412407-58451580-80b1-11e9-869a-56ce832033bb.JPG)

**주의** <br>
Tidy Data화 하지 않으면 info, describe, 등.. 초기 작업시 엉망으로 값이 나온다

<br>
### 행 뽑기

```python
tb.loc[5:7]
```
![loc](https://user-images.githubusercontent.com/33630505/58419146-bd563680-80c4-11e9-99e6-fd1fa156b058.JPG)
```python
tb.iloc[1:3] # 파이썬 방식
```
![iloc](https://user-images.githubusercontent.com/33630505/58419147-bd563680-80c4-11e9-9aeb-d56d82b27083.JPG)
## 상관성 체크하기  (correlation)

> 두 변수간에 어떤 선형적 관계를 갖고 있는지 분석하는 방법이 상관 분석. 그렇다면 상관성 있다는 것은 얼마나 관계가 있는지에 대한 정도라고 볼 수 있다. 만약 상관성이 1에 가깝다면 두 변수는 매우 관련 이 있다. 예를 들어 키가 크면 몸무게가 많이 나가는 것처럼 서로 관계가 가까운것.

**양의 상관성**: 기준이되는 변수가 커지면 상대 변수도 같이 커진다 <br>
**음의 상관성**: 기준이되는 변수가 커지면 상대 변수는 작아진다 <br>

<span style='color: red'>상관 분석은 왜 하는거야?</span><br>
데이터 분석시 column이 많아지면 계산이 복잡해지는데 상관관계를 따져 <br>
상관성이 높은 것들은 분석 데이터에서 제외시켜 계산 복잡도를 크게 줄일 수 있기 때문이다.
<br>

```python
import pandas as pd

data = pd.read_csv('file.txt')
data.rename({0:'sl',1:'sw',2:'pl',3:'pw','class':'class_'},axis=1,inplace=True)
data.corr() # method = {'pearson', 'kendall', 'spearman'}
```
![corr](https://user-images.githubusercontent.com/33630505/58418421-a9a9d080-80c2-11e9-9188-af2eb88e45b1.JPG)

**공분산**
```python
data.cov()
```

## 문자열에 사용하는 것들

### Series에서 object(문자열) 빈도수 체크하기

```python
import pandas as pd

data = pd.read_csv('file.txt')
data.rename({0:'sl',1:'sw',2:'pl',3:'pw','4':'class_'},axis=1,inplace=True)
data['class_'].value_counts()

: Iris-versicolor    50
  Iris-setosa        50
  Iris-virginica     50
  Name: class_, dtype: int64


data['class_'].value_counts().plot.pie()
data['class_'].value_counts().plot.bar()
```
![pie](https://user-images.githubusercontent.com/33630505/58419044-78320480-80c4-11e9-8ebb-cca88b2feb3b.JPG)
![bar](https://user-images.githubusercontent.com/33630505/58419048-79fbc800-80c4-11e9-980c-5105d90b2777.JPG)


### nlargest, nsmallest, unique

```python
x = data['class_'].value_counts()

x.nlargest()
x.nsmallest()
data['class_'].unique()

: Iris-versicolor    50
  Iris-setosa        50
  Iris-virginica     50
  Name: class_, dtype: int64

  Iris-versicolor    50
  Iris-setosa        50
  Iris-virginica     50
  Name: class_, dtype: int64

  array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
```


## 기초 통계 분석시 알아두면 좋은 원칙 및 정리


```
1. Occam's Razor (오캄의 면도날)
- 같은 성능을 보일 때 간단한것을 택한다
2. Curse of dimensionality (차원의 저주)
- 차원이 커지면 커질수록 필요한 데이터의 양이 커져야 한다
3. Law of large numbers (큰 수의 법칙)
- 큰 모집단에서 무작위로 뽑은 표본의 평균이 전체 모집단의 평균과 가까울 가능성이 높다
- 모집단이 커지면 표본평균은 모평균을 더 정확히 추정할 수 있다
4. Central limit theorem (중심 극한 정리)
- 동일한 확률분포를 가진 독립 확률 변수 n개의 평균의 분포는 n이 적당히 크다면 정규분포에 가까워진다는 정리
```


## Indexing & Slicing (Select data)

내가 필요한 통계값 구하기 위해


## MultiIndex

![multiindex](https://user-images.githubusercontent.com/33630505/58419324-440b1380-80c5-11e9-9630-d4f56d60c460.JPG)

**Pandas Tip1** 예측 분석을 하려면 문자열을 숫자로 바꿔줘야한다 (Encoding)


<br>

**예시에 나오는 데이터 출처** : [archive](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)<br>

**복습시간** 18시 30분 ~ 21시 / 총 2시간 30분



<a id = '17th'></a>
# 2019년 5월 28일 화요일 17th

## loc, iloc + lambda

```python
import pandas as pd
import numpy as np

data = pd.DataFrame(np.random.randn(6,4), index = list('abcdef'), columns = list('ABCD')
data
: 	   A	            B               C               D
a	0.427092	1.122736	1.064223	-0.724660
b	0.091881	1.049868	1.263243	-0.193525
c	0.224007	-1.128729	-1.261087	2.461563
d	-0.859961	-0.450851	-0.098474	0.456542
e	0.339599	-0.946570	0.892721	-0.331624
f	1.691290	-0.565636	0.905357	-0.301717

data.loc[lambda x: x.B>0, :]
: 	   A	            B               C               D
a	0.427092	1.122736	1.064223	-0.724660
b	0.091881	1.049868	1.263243	-0.193525

data.loc[:, lambda x:['D','A']]
:           D	           A
a	-0.724660	0.427092
b	-0.193525	0.091881
c	2.461563	0.224007
d	0.456542	-0.859961
e	-0.331624	0.339599
f	-0.301717	1.691290

data.iloc[:,lambda x:[0,3]]
:           A	            D
a	0.427092	-0.724660
b	0.091881	-0.193525
c	0.224007	2.461563
d	-0.859961	0.456542
e	0.339599	-0.331624
f	1.691290	-0.301717

data[lambda x: x.columns[3]]
: a   -0.724660
  b   -0.193525
  c    2.461563
  d    0.456542
  e   -0.331624
  f   -0.301717
Name: D, dtype: float64
```

## columns

```python
import seaborn as sns

tips = sns.load_dataset('tips')
tips
:    total_bill	 tip	 sex   smoker	day	time	 size
0	16.99	1.01	Female	 No	Sun	Dinner	  2
1	10.34	1.66	Male	 No	Sun	Dinner	  3
2	21.01	3.50	Male	 No	Sun	Dinner	  3
3	23.68	3.31	Male	 No	Sun	Dinner	  2

tips.melt(tips.columns[:3])    #  열만 따로 뽑기
:    total_bill	 tip	sex	variable   value
0	16.99	1.01	Female	smoker	     No
1	10.34	1.66	Male	smoker	     No
2	21.01	3.50	Male	smoker	     No
3	23.68	3.31	Male	smoker	     No
```

## index

```python
import pandas as pd
data = pd.read_csv('billboard.csv',engine='python')
data.melt(data.columns[:7]).set_index('genre').loc['Rock']

: 	year	artist.inverted	     track	     time	    date.entered       date.peaked    variable        value
genre								
Rock	2000	Destiny's Child	 Independent         3:38            2000-09-23         2000-11-18    x1st.week      78.0
                                  Women Part I	     		                         	  
Rock	2000	  Santana	 Maria, Maria	     4:18	     2000-02-12	        2000-04-08    x1st.week	     15.0
Rock	2000	Savage Garden	I Knew I Loved You   4:07	     1999-10-23	        2000-01-29    x1st.week	     71.0
Rock	2000	Madonna	             Music	     3:45	     2000-08-12	        2000-09-16    x1st.week	     41.0
````

## Intersection

```python
a = {1,2,3}
b = {3,4}

a.intersection(b)
: {3}
a.intersection([3,4])
: {3}
a.intersection(range(3))
:{1,2}
```

## 새로운 연산자 만들기

```python
class x(int):
	def __add__(self, other):
		print('안더해줌')

x(3) + x(4)
: 안더해줌
```

## isin (predicate)

```python
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')

s.isin([2, 4, 6])
: 4    False
  3    False
  2     True
  1    False
  0     True
  dtype: bool

s[s.isin([2, 4, 6])]  
: 2    2
  0    4
  dtype: int64
```

## where

## split, strip


**복습시간** 12시 ~ 1시 30분 / 총 1시간 30분



<a id = '18th'></a>
# 2019년 5월 30일 목요일 18th


## 기초통계 분석시 그래프 그리는 3가지

```
1. boxplot
2. pairplot
3. heatmap
```

### boxplot

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
tips.boxplot()  # tips Dataframe의 attribute로 내장하고 있음
# or
pd.plotting.boxplot(tips)
```

### pairplot

> 짝을 이뤄 그리는 그래프

```python
import pandas as pd
import seaborn as sns


tips = sns.load_dataset('tips')
sns.pairplot(tips)  # tips Dataframe의 attribute로 내장하고 있지 않다
sns.pairplot(tips, hue='sex')
```

### heatmap

> 상관 분석시 그리는 그래프

```python
import seaborn as sns

tips = sns.load_dataset('tips')
sns.heatmap(tips.corr())
sns.heatmap(tips.corr(), cbar = False, annot = True) # 오른쪽 사이드바 제거, 평면에 상관계수 표시
```

## Dataframe은 Iterator/Generator 처럼 next연산을 할 수 있다

```python
import seaborn as sns

tips = sns.load_dataset('tips')
x = tips.items() # or x = tips.iteritems()
y = tips.iterrows()
type(x)
type(y)
next(x)
next(y)
: generator
  generator
  ('total_bill',
   0      16.99
   1      10.34
   2      21.01
   3      23.68
   4      24.59
   5      25.29
   (0, total_bill     16.99
   tip             1.01
   sex           Female
   smoker            No
   day              Sun
   time          Dinner
   size               2
   Name: 0, dtype: object)
```


## Pandas data type 종류

```
1. 숫자      => int64, float64
2. 문자      => object, category
3. 시간/날짜
```

**Dask**는 메모리의 제한으로 dataframe을 만들 수 없는 경우 도움을 줄 수 있는 패키지 이다. 파이썬으로 작성한 작업을 병렬화 할 수 있다.



```python
!dir

: 2019-05-25  오후 10:01             4,173 pandas.ipynb
  2019-05-25  오후 08:49           139,785 pandas2.ipynb
  2019-05-26  오후 02:51           220,247 pandas3.ipynb
  2019-05-27  오후 09:18           749,763 pandas4.ipynb

----------------------------------------------------------
# 오늘 받은 패키지

!pip install vincent
!pip install -q pdvega # -q 옵션은 설치시 나오는 메시지 생략
                       # -U 옵션은 최신 버전이 아닐 경우 업데이트
```
**Jupyter Tip1** jupyter에서 !(느낌표) 뒤에 cmd에서 작동하는 명령어를 치면 작동한다




<br>

**pdvega** <br>

```python
import pdvega

s = tips.groupby('smoker')
s.vgplot.bar()
```

![vgplot](https://user-images.githubusercontent.com/33630505/58626155-11545b80-830f-11e9-8744-da3eb42a2161.JPG)


## Aggregation analysis (집합 분석)

### Groupby의 내부적 구현 순서

```
1. iterrow 순회
2. split        => groupby
3. apply        => mean, max ... (통계적으로 대표할 수 있는 값 설정)
4. combine      => 결과 묶기
```

## Group 3총사

```
1. groupby
2. pivot table
3. crosstab (pd로 접근)
```

### 2. pivot table
```python
import seaborn as sns

tips = sns.load_dataset('tips')
tips.pivot_table(index='smoker', columns = 'sex', aggfunc = np.sum, margins = True)
# margin은 부분합을 보여줌

:  	size	                tip	                total_bill
sex	Male	Female	All	Male	Female	All	Male	Female	All
smoker									
Yes	150	74	224	183.07	96.74	279.81	1337.07	593.27	1930.34
No	263	140	403	302.00	149.77	451.77	1919.75	977.68	2897.43
All	413	214	627	485.07	246.51	731.58	3256.82	1570.95	4827.77
```

### 3. crosstab

```python
import seaborn as sns

tips = sns.load_dataset('tips')
a = pd.crosstab(tips.smoker, tips.sex, tips.tip, aggfunc = np.max)
a.index

# smoker가 index, sex가 column, tip이 value, aggfunc는 value의 대푯값

:
sex	Male	Female
smoker		
Yes	10.0	6.5
No	9.0	5.2

CategoricalIndex(['Yes', 'No'], categories=['Yes', 'No'], ordered=False, name='smoker', dtype='category')

b = pd.crosstab(tips.smoker,[tips.sex,tips.time],tips.tip,aggfunc=np.max)
b.index
# multi columns

:
sex	Male	        Female
time	Lunch	Dinner	Lunch	Dinner
smoker				
Yes	5.0	10.0	5.00	6.5
No	6.7	9.0	5.17	5.2

CategoricalIndex(['Yes', 'No'], categories=['Yes', 'No'], ordered=False, name='smoker', dtype='category')

c = pd.crosstab([tips.smoker,tips.sex],tips.time,tips.tip,aggfunc=np.max)
c.index

: multi index
        time	Lunch	Dinner
smoker	sex		
Yes	Male	5.00	10.0
        Female	5.00	6.5
No	Male	6.70	9.0
        Female	5.17	5.2

MultiIndex(levels=[['Yes', 'No'], ['Male', 'Female']],
           codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
           names=['smoker', 'sex'])

d = pd.crosstab([tips.smoker,tips.sex],tips.time,tips.tip,aggfunc=np.max).index
d.labels # or d.codes

: FrozenList([[0, 0, 1, 1], [0, 1, 0, 1]])
```


```python
# sequnce 방식
x.codes
: FrozenList([[0, 0, 1, 1], [0, 1, 0, 1]])

x.labels[0]

# attribute 방식
from collections import namedtuple
n = namedtuple('Jung', ['x','y'])
a = n(1,2)
a
: Jung(x=1, y=2)

a.x
a.y
: 1
  2
```
**Pandas Tip1** 데이터 형태가 []를 포함하면 sequence 방식 , xx = yy 가 있으면 attribute 방식


## reindex & resetindex

> reindex는 수동으로 index변경, resetindex는 0부터 자동으로 index 변경

### resetindex
```python
import seaborn as sns

tips = sns.load_dataset('tips')
n = tips[tips.sex == 'Male'].loc[:15] # 맨처음 index부터 15 index까지 행 뽑기
n.reset_index(drop=True)  # 기존 index 버리고 0부터 새로 생성

: 	total_bill	tip	sex	smoker	day	time	size
0	  10.34	        1.66	Male	No	Sun	Dinner	  3
1	  21.01	        3.50	Male	No	Sun	Dinner	  3
2	  23.68	        3.31	Male	No	Sun	Dinner	  2
3	  25.29	        4.71	Male	No	Sun	Dinner	  4
4	  8.77	        2.00	Male	No	Sun	Dinner	  2
5	  26.88	        3.12	Male	No	Sun	Dinner	  4
6	  15.04	        1.96	Male	No	Sun	Dinner	  2
7	  14.78	        3.23	Male	No	Sun	Dinner	  2
8	  10.27	        1.71	Male	No	Sun	Dinner	  2
9	  15.42	        1.57	Male	No	Sun	Dinner	  2
10	  18.43	        3.00	Male	No	Sun	Dinner	  4
11	  21.58	        3.92	Male	No	Sun	Dinner	  2

n.reset_index() # 기존의 index 삭제 X

:     index	total_bill	tip	sex	smoker	day	time	size
0	1	10.34	        1.66	Male	No	Sun	Dinner	  3
1	2	21.01	        3.50	Male	No	Sun	Dinner	  3
2	3	23.68	        3.31	Male	No	Sun	Dinner	  2
3	5	25.29	        4.71	Male	No	Sun	Dinner	  4
4	6	8.77	        2.00	Male	No	Sun	Dinner	  2
5	7	26.88	        3.12	Male	No	Sun	Dinner	  4
6	8	15.04	        1.96	Male	No	Sun	Dinner	  2
7	9	14.78	        3.23	Male	No	Sun	Dinner	  2
8	10	10.27	        1.71	Male	No	Sun	Dinner	  2
9	12	15.42	        1.57	Male	No	Sun	Dinner	  2
10	13	18.43	        3.00	Male	No	Sun	Dinner	  4
11	15	21.58	        3.92	Male	No	Sun	Dinner	  2

```

## 행, 열 위치 변환하기

### 기준 데이터
```python
tips.groupby(['sex','smoker']).mean()[['tip']]


		tip
sex	smoker
Male	Yes	3.051167
        No	3.113402
Female	Yes	2.931515
        No	2.773519

```

### stack
```python
tips.groupby(['sex','smoker']).mean()[['tip']].stack()

sex     smoker     
Male    Yes     tip    3.051167
        No      tip    3.113402
Female  Yes     tip    2.931515
        No      tip    2.773519

dtype: float64
# 1차원으로 바뀜
```

### unstack

```python
tips.groupby(['sex','smoker']).mean()[['tip']].unstack()

	tip
smoker	Yes	        No
sex		
Male	3.051167	3.113402
Female	2.931515	2.773519

```

### Column이 2개 이상일 때 그래프

#### Stacked = True (Column값을 쌓는다)
```python
tips.groupby(['day','sex']).mean()[['tip']].unstack().plot.bar(stacked=True)
```
![stacked_True](https://user-images.githubusercontent.com/33630505/58628687-5f6c5d80-8315-11e9-9c14-5bc64e11af28.JPG)

#### unstack(0) (index와 열의 조합)
```python
tips.groupby(['day','sex']).mean()[['tip','total_bill']].unstack(0).plot.bar(stacked=True)
```
![zero](https://user-images.githubusercontent.com/33630505/58629825-79f40600-8318-11e9-9a54-a760d7049719.JPG)

#### Stacked = False (Column값을 쌓지 않는다)
```python
tips.groupby(['day','sex']).mean()[['tip']].unstack().plot.bar(stacked=False)
```
![stacked_False](https://user-images.githubusercontent.com/33630505/58628716-714e0080-8315-11e9-812d-006a2a4efea9.JPG)

#### unstack(1) (header와 열의 조합)

```python
tips.groupby(['day','sex']).mean()[['tip','total_bill']].unstack(1).plot.bar(stacked=True)
```

![one](https://user-images.githubusercontent.com/33630505/58629826-79f40600-8318-11e9-81a2-e408173faa35.JPG)

**sci** Stack은 Column을 Index로 바꿔주고, Unstack은 Index를 Column으로 바꿔준다




**복습시간** 18시 13분 ~ 20시 21분/ 총 2시간 8분




<a id = '19th'></a>
# 2019년 5월 30일 금요일 19th


## Data Visualization

> 문자나, 숫자 보다 그림으로 혹은 그래프로 시각적인 정보가 사람에게는 더 명확하고 효율적으로 전달 되기 때문에 데이터 분석 결과를 시각화 할 수 있어야 한다

### 시각화 라이브러리
![lib](https://user-images.githubusercontent.com/33630505/58707963-5ea8f980-83f1-11e9-8a8e-33d77621fc9d.JPG)

### Python 시각화 라이브러리 분류
![pylib](https://user-images.githubusercontent.com/33630505/58707964-6072bd00-83f1-11e9-99ab-3f366ebac47e.JPG)

> Costumize하려면 Matplotlib을 사용해야 한다

**matplotlib, seaborn** matplotlib는 python, numpy format으로 데이터를 처리하고 seaborn은 pandas format으로 데이터를 처리한다. .value는 pandas 데이터 형태를 numpy format으로 바꿔준다



## Matplotlib

1. pyplot
2. pylab

> 이제 pylab은 쓰지 않는다

### 구성요소

![structure](https://user-images.githubusercontent.com/33630505/58708062-ab8cd000-83f1-11e9-842e-c69ea2654837.JPG)

![graph](https://user-images.githubusercontent.com/33630505/58708348-8187dd80-83f2-11e9-96df-29952d1ff993.JPG)

### 그래프 커스터마이징 하기

```python
import matplotlib.pyplot as plt

# canvas, figure, axes는 생략하면 자동으로 생성해서 그래프를 그려준다
# 단 생략하지 않으면 커스텀 할 수 있다

plt.figure(figsize=(10,5), facecolor='yellow')    
plt.axes(xlim=(0,10),ylim=(0,10))  # xlim,ylim은 최대 범위를 지정
plt.title('Title')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.grid(axis='y')
plt.plot([1,2,3,4,5,6],[2,0,4,7,3,10], color='black', marker='o')
```
![plotgraph](https://user-images.githubusercontent.com/33630505/58708975-117a5700-83f4-11e9-9978-19420df353c3.JPG)


### matplotlib에서 제공해주는 스타일

```python
plt.style.available

['bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark-palette',
 'seaborn-dark',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'seaborn',
 'Solarize_Light2',
 'tableau-colorblind10',
 '_classic_test']
```

**예시**

```python
import seaborn as sns

iris = sns.load_dataset('iris')
plt.style.use('ggplot')
sns.pairplot(iris, hue='species')
```

![ggplot](https://user-images.githubusercontent.com/33630505/58709958-366fc980-83f6-11e9-8d80-2c457a90c1fb.JPG)


**복습시간** 22시 ~ 22시 40분 / 총 40분



<a id = '20th'></a>
# 2019년 6월 3일 월요일 20th

## Folium

> 지도 그리는 python 패키지 또는 라이브러리. 분석에 필요한 단계구분도를 하기 위해서 사용한다.
> Google map에서 갖고옴.


### Folium 설치
```shell
pip install folium
```

### Folium Map

```python
import folium

mymap = folium.Map(location=[37.332268, 127.180961], zoom_start = 11, tiles='Stamen Toner')
folium.Marker([37.332268, 127.180961], popup='<i>Ji hyeok home</i>',
icon=folium.Icon(icon='cloud')).add_to(mymap)
folium.Marker([37.543148,126.949866], popup='<b>My location</b>').add_to(mymap)

folium.CircleMarker(
location=[37.332268, 127.180961],
radius=80,
popup='My area',
color='#3186cc',
fill=True,
fill_color='#3186cc'   
).add_to(mymap)

mymap.add_child(folium.LatLngPopup()) # 지도위 클릭시 위도, 경도 보여줌
mymap.add_child(folium.ClickForMarker(popup="ClickPoint")) # 지도위 클릭시 클릭위치에 표시됨
```

![map](https://user-images.githubusercontent.com/33630505/58799253-cce2fb80-863f-11e9-8f2a-fac434b50f71.JPG)

## file 불러오기

보통 pandas로 파일을 불러오지만<br>
파일 구성이 복잡하여 불러오지 못하는 파일은 open으로 불러와야 한다<br>
open으로 불러온 데이터는 text(객체의 의미를 갖지 못함)형태로 불러오고 <br>
이 text를 csv나 json형태로 불러와 의미 부여해줘야 한다 (csv, json만 가능) <br>
나머지는 pickle로?

```python
import json
from pprint import pprint

with open('seoul_municipalities_geo_simple.json', encoding='utf-8') as f:
    x = json.load(f)

pprint(x)
len(x)
len(x['features'])
x['features'][0]['properties']['name']
x['features'][0]['geometry']['type']
: {'features': [{'geometry': {'coordinates': [[[127.11519584981606,
                                              37.557533180704915],
                                             [127.11879551821994,
                                              37.557222485451305],
                                             [127.12146867175024,
                                              37.55986003393365],
                                             [127.12435254630417,
                                              37.56144246249796]
  2
  25
  '강동구'
  'Polygon'
```

## 단계구분도

```python
import json, folium
import pandas as pd

seoul_geo_json = open('seoul_municipalities_geo_simple.json',encoding='utf-8')
seoul_geo_json = json.load(seoul_geo_json)

data = pd.DataFrame.from_dict(seoul_geo_json['features']).properties

keys = data[0].keys()

data_list = {}
for key in keys:
    temp_list = []
    for inst in data:
        temp_list.append(inst[key])
    data_list[key] = temp_list

seoul_df = pd.DataFrame.from_dict(data_list)    
seoul_df.to_csv('seoul_map.csv')

seoul = folium.Map(location=[37.5665, 126.9780], tiles='Mapbox Bright')
seoul_geo_df = pd.read_csv('seoul_map.csv')
seoul.choropleth(
    geo_data=seoul_geo_json, # json
    name='choropleth',
    data=seoul_geo_df,  # pandas
    columns=['name', 'code'],
    key_on='feature.properties.name', # geo data와 pandas data 맞춰준다?
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='population'
)

seoul
```

![seoul](https://user-images.githubusercontent.com/33630505/58800748-9f984c80-8643-11e9-80aa-83ede9c45b0b.JPG)

## map 사용하여 특정 열 값 뽑아내기

```python
import json

seoul_geo_json = open('seoul_municipalities_geo_simple.json',encoding='utf-8')
seoul_geo_json = json.load(seoul_geo_json)

data=pd.DataFrame.from_dict(seoul_geo_json['features'])
t=pd.DataFrame.from_dict(data.properties)

t

: 	                                         properties
0	{'code': '11250', 'name': '강동구', 'name_eng': '...
1	{'code': '11240', 'name': '송파구', 'name_eng': '...
2	{'code': '11230', 'name': '강남구', 'name_eng': '...
3	{'code': '11220', 'name': '서초구', 'name_eng': '...
4	{'code': '11210', 'name': '관악구', 'name_eng': '...

t.properties.map(lambda x:x['name'])

:
0      강동구
1      송파구
2      강남구
3      서초구
4      관악구
```

## pandas format으로 불러들이는 방법 3가지

```
1. pd.read_csv
2. pd.DataFrame
3. pd.DataFrame.from_dict
```

<span style='coloc:red'>※ 보충 필요 </span>

## Machine Learning

## 기계학습시 거치는 단계

```
1. 방법 (알고리즘)
2. 하이퍼 파라미터

컴퓨터에게 방법(알고리즘)을 알려주고 스스로 학습을 하고
학습한 데이터를 기반으로 예측을 할때 비슷한걸 찾는다
이때 기계학습에는 전부 숫자데이터로 하기 때문에 근처 숫자 값을
찾아 예측하게 된다 (KNN 알고리즘)

하이퍼 파라미터는 근처 값 몇개를 찾아보고 예측을 할지 정해주는 것이다
하이퍼 파라미터를 몇개로 줘야 하는지는 성능이 좋은 것에 따라 지정해주면 된다

그리고 알고리즘, 하이퍼 파라미터 둘다 컴퓨터가 알아서 성능 좋은걸로 선택하게 할 수도 있다
```

**KNN** K-Nearest Neighbor 최근접 이웃 알고리즘



```python
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
print(data.DESCR) # 데이터 이해를 위해 보는 것

data_pd = pd.DataFrame(data.data, columns=data.feature_names) # 인스턴스 방식
data_target = pd.DataFrame(data.target, columns=['target'])
iris = pd.concat([data_pd, data_target], axis = 1) # data_pd + data_target 결합


knn = KNeighborsClassfier(3) # 근처 3개를 확인해라
knn.fit(iris.iloc[:,:-1], iris.iloc[:,-1])

: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=3, p=2,
           weights='uniform')

data.target_names
knn.predict([[3,3,4,3]])
knn.predict_proba([[3,3,4,3]])

: array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
  array([1])  # versicolor로 예측
  array([[0.        , 0.66666667, 0.33333333]])  # 가까운 값이 versicolor 2개, virginica 1개가 있었음

※ Bunch
# dictionary + attribute

type(data)
: sklearn.utils.Bunch

data.data
data['data']
# 둘다 접근 가능한 데이터 타입

dir(data)
: ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
```


Folium 활용 : [pythonhow](https://pythonhow.com/web-mapping-with-python-and-folium/)

**복습시간**   18시 30분 ~ 21시 / 2시간 30분
