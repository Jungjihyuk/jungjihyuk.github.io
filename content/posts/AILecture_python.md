---
title: AI 이노베이션 스퀘어 수업(기본반) - python
date: 2019-04-29
draft: false
description: AI 이노베이선 스퀘어에서 배운 AI 공부 정리
categories:
- AI
- Python
tags:
- AI
- Lecture
- Python
slug: AILecture_python
---

<a id = '1st'></a>
# 2019년 4월 29일 월요일 1st
이 수업은 기본반이라고 되어 있지만 사실상 `fundamental` 즉, 핵심적이고, 근본적인, 필수적인 것들을 다루기 때문에 어려운 내용도 포함 하고 있다.

<br>

## 왜 프로그래밍 언어가 많은 걸까?? <br>
: 언어마다 각각의 `장단점`이 있기 때문이다 <br>
그리고 더 구체적으로는 언어마다 `특화되있는 분야`가 있기 때문이다 <br>

<br>

## 왜 AI에는 파이썬인가?
```
Life is short, you need!
파이썬의 슬로건에서 보여주듯이
파이썬은 개발 속도가 빠르기 때문에 생산성이 좋다!

1. 생산성
- Efficient  --> 빠르고 체계적인 일처리가 가능하다  
- Effective  --> 원하는 결과를 가져다 준다

2. 멀티 패러다임
- 절차 지향, 객체지향, 함수지향 프로그래밍이 모두 가능하다

3. 연구자 친화적 언어, 개발자 친화적 언어  
- 언어중에서 속도가 느린 언어에 속하지만 개발속도는 빠르고 함수적 프로그래밍이 가능하다

그리고 이 파이썬은 크게 3가지 분야에서 유리하다.
첫번째 시스템 자동화 분야
두번째 웹
세번째 데이터 사이언스
```
※ 이 수업은 Data Science분야를 집중해서 다뤄볼 것이다. <br>
<br>


## Python의 특징

1. `다양한 패러다임`을 지원한다.
2. `글루` 언어다.
- 언어이면서 명세서이다.
- CPython / De facto -> 많이 쓰여서 표준이된것. (사실상 표준)

3. `Library & Tool` (많다)
4. Python: General Purpose(범용적) <-> R: domain specific(도메인용, 통계전문)
5. 인터프리터 언어이다.
6. 모바일 지원에 대해 약하다. (이는 절대 개선될 수 없다. 왜냐하면 apple과 google에서 swift, 및 자사 언어를 사용하기 때문에)
- 그러나 DS분야에서는 그나마 낫다.

7. 속도가 엄청 느리다. (Dynamic language)

<br>

※이번 수업은 c로 만든 python으로 사용한다 <br>

※python3.3 부터 유니코드로 인한 속도가 개선되었다. <br>
<br>


## Interpreter vs Compiler

![interpreter vs compiler](https://user-images.githubusercontent.com/33630505/56888295-93eac080-6aae-11e9-977b-2db7ce9cd580.JPG)
<br>
<br>

## REPL vs IDE vs Text Editor
<kbd>REPL</kbd>: &nbsp; Read-Eval-Print loop의 약자로 컴파일 과정없이 즉석으로 코드를 입력하고 실행결과를 바로 보는 대화형 환경을 말한다.<br>
<kbd>IDE</Kbd>: &nbsp; Integrated Development Environment의 약자로 개발통합환경 즉, 코딩을 위한 편집기, 컴파일, 링크, 디버그 등... 실행 모듈 개발에 필요한 기능을 한곳에 모아둔 프로그램을 말한다. <br>
<kbd>Text Editor</kbd>: &nbsp; 문서 편집기, 메모장이나 노트패드, atom 등을 text editor라고 하고 보통 코딩을 위한 text editor는 코딩을 더 편리하게 도와주는 기능이 탑재되어있다. <br>
atom같은 경우에는 원하는 경우 설치하여 터미널창을 추가할 수도 있고 각 언어별 자동완성기능 등 여러가지 편리한 기능들을 사용할 수있도록 도와주는 프로그램이다.

<br>

## Jupyter notebook

Python + Markdown 지원 REPL? <br>

Python의 REPL에서 여러줄의 코드를 실행하기 편하고 편집하기 유용한 버전으로 업그레이드 한 것이 IPython Notebook. <br>
IPython의 강점은 한 파일 내에서 쉽게 코드 cell 단위로 실행할 수 있다는 것이다. <br>
다만 IPython는 파이썬 전용이다. 그리고 한 번 실행하고 Python 버전을 바꾸려면 로컬 서버를 내렸다가 다시 올려야 하는 불편함이 있다. <br>
이러한 불편함을 극복하고 만들어진 것이 바로 Jupyter Notebook이다. <br>
Jupyter 이름에는 뜻이 숨어 있다. Ju는 Julia Py는 Python 그리고 R은 R 세 단어를 합친 단어이다. <br>
Jupyter notebook은 특이한 점이 웹에서 사용한다는 것이다. <br>

> 앞으로 이 수업은 파이썬 공식 문서를 이용할 것이다. 왜냐하면 정확한 정보를 제공하고 공식 문서를 보는 연습을 미리 해두면
나중에 공부할때 필히 도움이 될 것이다. &nbsp; [공식 문서](https://docs.python.org/3/)

<br>

## 자료형 숫자, 문자  

```
자료형 중 `숫자`는 크게 `4가지`가 있다.
1. Integer
2. Float
3. Boolean
4. Complex

Integer는 정수형으로 숫자의 범위가 없다. 따라서 오버플로우가 없다.
그리고 python에서 정수의 종류는 한 가지

Float는 부동소수로 숫자의 범위가 있다. 따라서 오버플로우가 있다.
※ 0.1 + 0.1 + 0.1 의 결과값은 0.3000000000000004로 나온다.
Why? --> 컴퓨터는 근사값을 계산하기 때문에 정확한 값을 출력하지 못한다.

※ a = 2e3  --> 2000.0
   b = 2e-3 --> 0.002

그리고 컴퓨터가 정수와 부동소수의 저장 방식의 차이 때문에 .의 유무로 정수인지 부동소수인지 판단한다.

Boolean은 True / False
True는 숫자 1에 대응되고 False는 숫자 0에 대응된다.
※ True 처럼 bold채로 쓰인 것은 키워드이다.

Complex는 복소수로 j로 표시한다.
ex) i = 1 + 3j
   : (1+3j)

`문자`는 크게 `3가지`가 있다.
1. String
2. Bytes
3. Bytearray

String은 python3 버전에서 unicode문자에 해당한다.
Byte는 문자 앞에 b를 붙이면 byte문자로 인식하고 ASCII코드에 해당한다.
ex) a = b'안녕'
    SyntaxError: bytes can only contain ASCII literal characters

```

`Python Tip1` Python에서는 변수를 식별자라고 명칭한다. 이때 식별자는 생성 규칙이 정해져 있다.
<br>

`변수 명명규칙`
```
1. 영문으로 써야한다.
2. 첫번째문자는 특수문자를 사용할 수 없다.
3. 미리 정의되어있는 문자는 사용할 수 없다.
```

<br>

`Python Tip2` 파이썬은 동적타입, 즉 타입을 지정해주지 않아도 자동으로 지정해주기 때문에 정수형인지
부동소수점인지 문자인지 등을 명시하지 않아도 타입이 지정된다.

<br>

### Python keword 종류
```python
import keyword
keyword.kwlist

['False','None','True',
 'and','as','assert',
 'async','await','break',
 'class','continue','def',
 'del','elif','else',
 'except','finally','for',
 'from','global','if',
 'import','in','is',
 'lambda','nonlocal','not',
 'or','pass','raise',
 'return','try','while',
 'with','yield']
```

<br>

### Namespace 보기
```python
%whos     # 여태까지 사용한 식별자를 보여줌

Variable   Type       Data/Info
-------------------------------
a          int        1
abc        bytes      b'abc'
b          int        2
c          int        3
f          bool       False
i          complex    (1+3j)
keyword    module     <module 'keyword' from 'C<...>conda3\\lib\\keyword.py'>
t          bool       True
w          str        ㅇㅇ
```

<br>

### 문자열 연산 및 예외 연산
```python
'정지혁' * 3
: 정지혁정지혁정지혁

'정' + '정'
: '정정'


'정' + 1
: TypeError: can only concatenate str (not "int") to str

# 데이터 타입에 따라서 지원되는 연산이 다르다.
```
<br>

### Markdown 문법
<span style="background-color: #fdbb5d">아주 기본적인 마크다운 문법을 배웠고 jupyter에서 작동하는 커멘드를 배웠다.</span><br>
<kbd>ESC</kbd> : &nbsp; 명령 커멘드<br>
<kbd>H</kbd>: &nbsp; Help (도움말)<br>
<kbd>ESC + Y</kbd>: &nbsp; Code mode<br>
<kbd>ESC + M</kbd>: &nbsp; Markdown mode<br>
<kbd>ESC + A</kbd> &nbsp; 현재 라인 위에 새로 추가하기<br>
<kbd>ESC + B</kbd> &nbsp; 현재 라인 밑에 새로 추가하기<br>
<kbd>ESC + L</kbd> : &nbsp; 해당 줄에 숫자 <br>
<kbd>Ctrl + Shift + P</kbd> : &nbsp; Help (도움말)<br>

`복습 시간`   18시 15분 ~ 19시 30분 / 총 1시간 15분


<hr>

<a id = '2nd'></a>
# 2019년 4월 30일 화요일 2nd

> 자료형은 총 20가지가 있지만 13가지를 배울 것이다.

## 자료형
```
1. 숫자형 [Integer, Float, Boolean, Complex]
2. 문자형 [String, Bytes, Bytearray]
3. List
4. Tuple
5. Set
6. Frozenset
7. Dictionary
8. Range
```

### 정수형을 표현하는 4가지 방법

1. 2진수
- 0b를 숫자 앞에 붙인다.
2. 8진수
- 0o를 숫자 앞에 붙인다.
3. 16진수
- 0x를 숫자 앞에 붙인다.
4. 10진수
- 아무것도 안쓰면 그냥 10진수이다.

<br>

### 정수형 숫자 범위

```python
import sys
sys.maxsize

결과: 9223372036854775807
정수형에서 오버플로우가 없다고 했는데..?
실제로 메모리에 할당 가능한 최대 숫자 크기가 저 값이고
초과되었을 때 동적으로 추가 할당이 된다.(?)

sys.int_info

sys.int_info(bits_per_digit=30, sizeof_digit=4)
32비트 python인 경우, 정수는 최대 30비트까지 허용된다?
```
<br>

### 진법 변환 builtin
bin() => 2진법으로 변환 <br>
oct() => 8진법으로 변환 <br>
hex() => 16진법으로 변환 <br>
2, 8, 16진법의 숫자를 입력하면 자동으로 10진수로 변환<br>
* bold채가 아닌데 색이 변하는 것은 builtin함수
<br>
<br>

### float

<span style="background-color: skyblue">부동소수는 소수점이 존재하는 수</span><br>
<span style="background-color: skyblue">예외적으로 무한대와 숫자가 아닌 경우도 포함한다</span><br>
<span style="background-color: skyblue">컴퓨터의 부동소수 연산은 정확하지 않은 근사값을 보여준다</span><br>
```python
float('inf')
=> inf (infinity)
float('nan')
=> nan (not a number)
# 무한대 개념은 딥러닝에서 미분의 개념을 꼭 알아야 하는데 이때 중요하게 작용한다.
  파이썬은 이처럼 숫자 체계가 범위가 넓기 때문에 인공지능에 이용되는 장점이 있다.

# 부동소수 연산   
a = 1.7976931348623157e+308
a
a + 1
a = a + 1

:
1.7976931348623157e+308
1.7976931348623157e+308
True
```

<br>

**.(점)은 부동소수점을 결정하는 리터럴**<br>

> 리터럴은 데이터 타입을 결정하는 문자를 일컫는 말이다.

<br>

```
import sys
sys.float_info

sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)
```
<br>

### float형 숫자의 연산
<kbd>+</kbd> , <kbd>-</kbd> ,<kbd> *</kbd> ,<kbd> /</kbd>, <kbd>//</kbd>,<kbd> %</kbd>, <kbd>**</kbd>가 있다.  

```
단, 음수 나눗셈은 주의 해야 한다.
ex)
10 // -3
=> -4
why? --> -4 + 0.66667 이런식으로 간주하기 때문에
```

### 자료형을 구별하는 기준

```
┌─ Mutable    # 추가, 삭제가 가능한 경우 / 특징 : 메모리 번지 안바뀜, 재할당할 필요없음
└─ Immutable  # 추가, 삭제가 불가능한 경우 / 특징 : 재할당으로 메모리 번지 바뀜

Container ┌─ Homogeneous    # 요소들이 서로 같은 타입인 경우
          └─ Heterogeneous  # 요소들이 서로 다른 타입이 가능한 경우
(요소가 1개 이상인 데이터 구조)

Sequence ┌─ Indexing   # 요소를 하나씩 출력
         └─ Slicing    # 요소를 한번에 여러개 출력
(순서가 있음)         
Lookup  ┌─ Mapping hash    # key값과 value를 갖는 Dictinary가 갖는 특징    
        └─ set  # 순서가 없는 고유한 원소들의 집합
(key값으로 이루어진 데이터 구조)
```

### Indexing

```python
a = '정지혁'
a[0]
: 정
```

### Slicing

```python
a = '정지혁'
a[0:3] or a[:3]
:'정지혁'
a[-3:] or a[0:]
:'정지혁'
a[0:3:2]  # 양수일때 (맨 뒤에 오는 숫자 - 1) 을 하면 건너 뛰는 문자의 개수를 나타낸다
:'정혁'
a[::-1]  # 음수일때 |맨 뒤에 오는 숫자 + 1|을 하면 건너 뛰는 문자의 개수를 나타내고 거꾸로 출력한다.
:'혁지정'
a[:-3:-1]
:'혁지'
a[:70] # 슬라이싱은 범위가 벗어나도 에러가 나지 않는다
:'정지혁'

b = '정지혁 천재'

b[slice(0,3)]
: '정지혁'
b[slice(0,-1,2)]
: '정혁천'
```

### List vs Tuple
`차이점` List는 수정, 삭제, 추가가 가능하고 인덱싱, 슬라이싱이 가능하지만, Tuple은 수정, 삭제, 추가가 불가능하고 인덱싱, 카운팅만 가능하다.


```python
x = [1]
type(x)
: list
y = (1)
type(y)
: int

#하나의 요소를 갖는 튜플을 생성하려면 콤마를 적어줘야한다.
y = (1,)
```

`Tuple Tip` 콤마가 뒤에 있다는 것은 튜플이라는 것을 알려주기 때문에 가독성도 높일 수 있고, 콤마를 써야만 tuple로 인식이 되는 경우가 있기 때문에 마지막에 콤마를 꼭 써주는 습관을 갖도록 하자. ex) (1,2,)


<br>

### Set vs Dictionary
`공통점` 둘다 집합의 성질을 띄어 중복허용이 불가능하고, 순서가 없다.


`차이점` set은 key값만 있는 반면 dictionary는 key, value쌍을 이뤄 mapping hash 타입을 이룬다.


```python
s = {}
type(s)
:dict
# python이 처음 만들었을 때는 set이 없었기 때문에 공집합을 만들면 Dictionary로 간주한다.

# 그렇다면 set은 공집합을 어떻게 만드나?
s = set()
type(s)
: set

# 집합은 고유의 연산자가 있다.
a = {1,2,3}
a - {2}
: {1,3}

a ^ {3}
: {1,2}

# set의 활용
# set은 공통된 메소드를 확인할때도 사용한다

set(dir(list())) & set(dir(tuple()))
:
{'__add__',
 '__class__',
 '__contains__',
 ....
 'count',
 'index'}
```

### __ missing__ & defaultdict

#### __ missing__

> 보통 dictionary에 존재하지 않는 key값에 접근할 경우 KeyError가 발생한다 <br>
> 그런데 missing 메소드를 재정의 해서 내가 원하는 return 값을 주면 에러가 발생하지 않는다 <br>
> 즉, missing 메소드를 구현하면 KeyError가 나는 상황에서 missing 메소드를 호출하게 된다 <br>

```python
class MyDict(dict):
    def __missing__(self, key):
        self[key] = rv = []
        return rv

m = MyDict()

type(m)
: __main__.MyDict

m.update({'x':1})  # 해당 키가 있으면 value를 수정하고 없으면 추가한다
m
: {'x': 1}

m['y']
: []

m
: {'x': 1, 'y': []}

m['y'].append(2)
: {'x': 1, 'y': [2]}

class MyDict2(dict):
    def __missing__(self, key):
        self[key] = rv = {}
        return rv

m2 = MyDict2({'a':1})

m2['b']
: {}

m2.update({"c":[1,2,3]})
m['b'].update({'x':1,"y":2})

m
: {'a': 1, 'b': {'x': 1, 'y': 2}, 'c': [1, 2, 3]}
```

#### defaultdict

> 없는 key값에 접근했을 때 에러가 나지 않고 default로 원하는 타입의 값 자동으로 할당해주는 dict type

```python
from collections import defaultdict

m = defaultdict(list)
type(m)
: collections.defaultdict

m
: defaultdict(list, {})

m['x'].append(1)
m
: defaultdict(list, {'x': [1]})

m['y']
: defaultdict(list, {'x': [1], 'y': []})

[m['y'].append(x) for x in range(1,11)]
: [None, None, None, None, None, None, None, None, None, None]

m
: defaultdict(list, {'x': [1], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
```

출처: [tistory](https://greatkim91.tistory.com/189)<br>

<hr>

`Internals` Python은 -5부터 256까지 숫자는 재할당을 해도 메모리 번지가 바뀌지 않도록 인터널 기법을 사용한다. 많이 쓰이는 작은 정수들을 미리 할당해 놓음으로써 메모리 공간과 연산비용을 많이 아낄 수 있게된다.


```python
from sys import intern

dir(intern)
:
['__call__',
 '__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__name__',
 '__ne__',
 '__new__',
 '__qualname__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__self__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__text_signature__']


※ 주의
# interning된 값을 재할당할때는 메모리 번지가 바뀌지 않는다

a = 100
id(a)
:
140706464961408
a = 100
id(a)
:
140706464961408
```


파이썬 내부 동작 원리 : [mingrammer](https://mingrammer.com/translation-cpython-internals-arbitrary-precision-integer-implementation/)<br>

`Garbage Collection`은 메모리 관리 기법 중 하나로, 프로그램이 동적으로 할당했던 메모리 영역 중에서 필요 없게 된 영역을 해제하는 기능이다.


<br>
<hr>


### xxx = xxx
<span style="background: orange">식별자 or 변수 = 식(Expression), statement </span><br>
<kbd>Expression</kbd>: 연산결과가 하나의 값으로 만들 수 있는 것 <br>
ex) 3 + 1 , float('inf') 등 ...  

### Expression vs Statement

<kbd>Expression</kbd>은 값 또는 값들과 연산자를 함께 사용해서 표현해 하나의 값으로 수렴하는 것 <br>
이때 함수, 식별자, 할당 연산자 [], 등을 포함한다. <br>
그리고 Expression은 평가가 가능하기 때문에 하나의 값으로 수렴한다. <br>

* 여기서 평가라는 의미를 정확하게 알고 있어야 한다. 평가란 컴퓨터가 이해할 수 있는 기계 이진 코드로 값을 변환하면 리터럴의 형태가 달라도 그 평가값은 같게 되는 것을 말한다. (표현식의 값을 알아낸다.)

ex)
```python
a, b = 4, 5
eval('1+2+3')
eval('a+b')
:6
:9
```
[참고](https://shoark7.github.io/programming/knowledge/expression-vs-statement.html)<br>

<br>
<kbd>Statement</kbd>는 예약어와 표현식을 결합한 패턴이다. <br>
그리고 <span style="color: orange">실행가능한</span> 최소의 독립적인 코드 조각을 일컫는다. <br>
<span style="color: orange">Statement</span>는 총 5가지 경우가 있다. <br>
1. 할당 (Assignment)
2. 선언 (Declaration)
3. 조건 (if)
4. 반복문 (for, while)
5. 예외처리

**for문으로 변수 할당하기**
```python
# statement
for i in range(1, 20+1):
   exec('r' + str(i) + '=' +str(i))

# r1, r2 ...r20 까지 1,2,...20 할당
```

<br>

### 기타 내장 함수

```
dir(obejct) : 어떠한 객체를 인자로 넣어주면 해당 객체가 어떤 변수와 메소드를 가지고 있는지 출력을 합니다.
type(object) : 어떠한 객체를 인자로 넣어주면 해당 객체의 타입을 반환한다.
len(object) : 어떠한 객체를 인자로 넣어주면 해당 객체의 요소 갯수를 반환한다.
id(object) : 해당 객체가 저장되어 있는 메모리 위치를 반환한다.
```

<br>

### id함수로 보는 메모리 할당

```python
x = 10
y = 10
hex(id(x))
:'0x7ffb006b9460'
hex(id(y))
:'0x7ffb006b9460'
# -5 ~ 256사이 숫자이기 때문에 재할당을 해도 메모리 주소가 변하지 않는다.

x = 1000
y = 1000
hex(id(x))
: '0x1bc813b7650'
hex(id(y))
: '0x1bc813b73d0'

x = 2000
hex(id(x))
: '0x1bc813b7db0'

# 새로운 식별자로 같은 값에 binding을 하거나 reassignment를 하게되면 메모리 주소값이 변경된다.  
```
※ 메모리 잘 쓰는 방법: &nbsp; [wikidocs](https://wikidocs.net/21057)
<br>

### 비교 연산자

```python
1 == 3
: False
1 != 3
: True

'정지혁' < '천재'
: True

[1,2] < [3]
: True

# 첫번째 요소 끼리 비교후 같으면 그 다음 요소 비교

[1,2] < 'a'
: TypeError

# 비교할 때는 같은 타입끼리 해야 한다.

a = 1000
b = 1000

a is b
: False

# 같은 메모리 번지인지까지 따진다.

'ㅈ' not in '정지혁'
: True

1 in [1,2,3]
:True

1 in '정지혁'
: TypeError

# 정수형과 리스트를 in 연산을 했을 때는 타입이 달라도 가능하지만
# 정수형과 문자형은 불가능하다
# 기본적으로 멤버쉽 연산자(in)는 container에서만 쓸 수 있고 문자열 일때는 문자열 끼리만 가능하다.
```

### Error
1. NameError
- 값을 할당하지 않은 식별자를 사용했을 때 발생하는 에러
2. IndexError
- 인덱스 값에서 벗어난 인덱스를 사용했을 때 발생하는 에러
3. TypeError
- 서로 다른 타입간의 연산을 했을 경우 발생하는 에러
4. KeyError
- dictionary에서 없는 key값으로 접근할때 발생하는 에러
5. AttributeError
- 존재하지 않는 Attribute에 접근하려고 할 때 발생하는 에러
6. UnboundLocalError
- 할당하기 전에 지역변수를 참조했을 때 발생하는 에러

`복습 시간`   18시 35분 ~ 20시 5분/ 총 1시간 30분  


<a id = '3rd'></a>
# 2019년 5월 2일 목요일 3rd


### 할당문의 종류 6가지
```
1. 기본 할당문
- a = 1
2. Chain assignment
- a = b = 1
3. Augmentation
- a += 1
4. Pakcing
5. Unpacking
- Container 쪼개기
6. Global , nonlocal
- global 이용하여 전역번수 접근, 변경하기 # 추천하지 않는 기능
- nonlocal은 함수 중첩시 함수와 함수 사이 변수
```

* Unpacking 방법은 빅데이터 처리시 많이 쓰인다.

```python
# Chain assignment #

a = b = 2
a
: 2
b
: 2

b = 3
a
: 2
# 변수 어느 하나 재할당 한다고 같이 바뀌지 않는다

-- 주의 --
x = y = [1000,2000]
y.append(3000)
x
: [1000,2000,3000]
y
: [1000,2000,3000]
# 그러나 mutable일때는 같이 변하므로 주의해야 한다!

# Augmentation #
a = 0
a += 1  # (재할당)

# 다른 언어처럼 선증가, 후증가가 지원되지 않는다
++a  (X)
a++  (X)

그러나 # ++a는 에러가 나지 않는다
Why?
# 앞에 있는 +를 부호연산자라고 간주하기 때문이다

# Packing #

x = 1,2,3
y = 1,

# Unpacking #

x,y = 1,2  (O)
x,y,*z = 1,2,3,4 (O)
*x, = 1,2,3 (O)
*a, = '정지혁' (O)
*u, = {'a':1,'b':2,'c':3,'d':4,'e':1} (O)  # 단, 키 값만 리스트 형태로 반환
*x, y = range(10) (O)

x, = 1,2 (X)  # 왼쪽 식별자와 오른쪽 식의 갯수를 맞춰줘야 함
*x, y, *z = 1,2,3,4,5 (X)
*x = 1,2,3 (X)  

# 오른쪽에 오는 식은 Container면 모두 가능
# * (별표)는 나머지를 리스트로 반환, 그리고 * 두 개이상 못씀

#Global  nonlocal#

a = 1

def name():
   global a
   a += 1
   return a
name()
: 2

def name():
   a = 1
   a += 1
   return a
# 위 함수와는 같지 않음 / Why? 밑 함수에서는 a를 그냥 재할당한 것임

def name(t):
	a = t
	print(a)
	def name2():
		nonlocal a
		a += 1
		print(a)
	return name2()

name(3)
: 3
  4
```
Packing & Unpacking: &nbsp; [blog](https://python.bakyeono.net/chapter-5-5.html)

<br>

### 조건의 형태 3가지

```
1. if, else & and, or
2. if, elif, else
3. 중첩 if
```

### if문 예시
```python
a = 6
if 0 < a < 10:
   print(True)
else:
   print(False)
```


### AND & OR
```
1. A and B  
- A가 참(Truely)이면 B 체크 => B 반환
- A가 거짓(Fasly)이면 B 체크 X => A 반환
2. A or B
- A가 참이면 B 체크 X => A 반환
- A가 거짓이면 B 체크 => B 반환
```

<br>

### 반복문 2가지

```
1. for
2. while
```

> 여기서 개념 하나 추가 Iterable! Iterable은 순회, 반복 가능한 것을 말한다.
그래서 for문에 쓸 수 있다.
보통 container이면 Iterable ※ 아닌것도 있지만 아주 나중에 배운다.


<span style="color: red">container이지만 Iterable이 아닌것이 set인가?? iterable의 조건중 __ iter__랑 __ getitem__ 두가지가 있어야 되는데 set은 __ iter__ 한가지 밖에 없다 왜지? </span>

```python
# for #

for i in {'a':1,'b':2}.values():
   print(i)

for i, j in {'a':1,'b':2}.items():
   print(i,j)

# in 뒤에 오는 것은 Container, 여러개의 요소를 갖고 있는것은 반복문이 가능하다

# while #

i = 0
while i < 10:
   i+=1
   print(i)
   if i == 5:
      break   # 탈출문   
else:
   print("완료")

# continue는 continue 밑은 실행하지 않고 넘어간다
```
* for문은 반복횟수를 알때 주로 사용하고, while문은 반복횟수가 정해지지 않았을 때, 모를때 주로 사용한다.

* 모든 for문은 while문으로 바꿀수 있지만, 모든 while문은 for문으로 바꾸기 어렵다.


### Enumeration

> for문에 index가 필요할때 iterable 객체를 enumerate로 감싸면 index값이 같이 출력된다

```python
a = ['a','b','c','d','e']
for index, value in enumerate(a):
    print(index, value)
:
0 a
1 b
2 c
3 d
4 e
```

### else문 3가지 쓰임
1. 조건문에서     => 조건에 맞지 않는 경우  
2. 반복문에서     => 반복문이 정상 완료 되고나서 실행
- 0번도 수행이라고 간주하기 때문에 else문이 실행될 수 있다
3. 예외처리할 때  

<hr>

### Dictionary view

1. key
2. values
3. items


#### 구문 실행시 실행시간 알아보기

```python
%%timeit

for i in range(3,10,2):
    for j in range(1,10):
        print(i,"*",j,"=", i*j)
    print()

# 391 µs ± 31.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

#### Python에서 중요한 두 가지 (Two A)
1. Abstraction
2. Automation

python 내부 구조 확인 가능 사이트: [pythontutor](http://pythontutor.com)

<br>

`복습 시간`   17시 45분~ 18시 43분/ 총 58분




<a id = '4th'></a>
# 2019년 5월 3일 금요일 4th

## 선언문 2가지

```
1. 함수 선언
2. 클래스 선언
```

```python
def name(arg1, arg2 = 2):
   pass       # 에러가 나지 않게 형태만 갖추기

name(3)       # 함수 호출(콜)
: (3,2)
```

`Argument`  인자로 어떤 데이터 타입도 올 수 있다. 단 튜플을 넣을 때 괄호를 꼭 써줘야 한다  

> Python의 또다른 장점 'return'문은 생략 가능하다

<br>

## Parameter vs Argument

Parameter | Argument
--------|-------
선언문에서 괄호 안 | 호출문에서 괄호안
키워드 방식이 온다 | 식이 들어갈 수 있다

<br>

### Parameter 사용법

```
1. Positional + keyword
2. Only Positional
3. Only Keyword
4. Variable Positional
5. Variable keyword
6. Variable Positional + Variable keyword
```

```python
# Positional + Keyword #
def name(a,b):
   ''' 함수 설명 '''   # Docs String => Shift + Tab 하면 설명이 그대로 나온다
   return a, b        #                참고로 함수 설명에 ( / )가 있을 경우 Positional 방식이라는 뜻
name(1,2)
:(1,2)

name(3, b = 4)
:(3,4)

name(a=1,b=2)
:(1,2)

# Only Keyword #
def name(*,a,b):   
   return a + b

# Variable Positional#
def name(*a):
   return a[0],a[1:3],a[3:]
name()
:()
name([1,2,3],(4,5,6),{'a':1,'b':2})
:([1, 2, 3], ((4, 5, 6), {'a': 1, 'b': 2}), ())

# !주의! 할당할때 *는 list를 반환하고 함수에서 *는 tuple로 반환한다

# Variable Keyword #
def name(**a):
   return a

name(a = ['a',2], b = {'a':'a','b':2}, c = range(3),d = 5)
:{'a': ['a', 2], 'b': {'a': 'a', 'b': 2}, 'c': range(0, 3), 'd': 5}


# Variable Positional + Variable Keyword #
def name(*b, **a):
   return b, a
name(2,3,4,5, a = 9, b = 3, c = [1,2])    # !주의! keyword를 쓰기 시작하면 끝까지 keyword를 써야한다
:((2, 3, 4, 6), {'a': 9, 'b': 3, 'c': [1, 2]})
```

`Positional Only` 사용자가 직접 만드는 함수의 파라미터에는 Positional only 방식을 사용할 수 없다. 하지만 기본 내장 함수 중에 파라미터 부분에 / 표시가 되어 있는 경우는 Positional Only 방식을 사용하라는 의미 이다. (shift + tab으로 설명 볼때), (앞으로 python 3.8부터는 positional only 방식을 지원한다고 한다.)


`positional only` : [python](https://discuss.python.org/t/pep-570-python-positional-only-parameters/1078)

> Python에서는 Parameter로 받아 올때 Type을 지정해주지 않는다.
Why? Python 철학중 EAFP라는 것이 있는데 이는 '허락보다는 용서를 구하기 쉽다'로
부부관계를 예시로 설명하면 이해하기 쉽다.
결혼하고 나면 보통 남자든 여자든 비싼 사치품을 사는 것이 쉽지 않다.
이때 사치품을 사려고 하는 입장의 사람은 결정해야한다.
사고 혼날 것인가.
허락을 받을 것인가.
전자가 더 실행하기 쉽고 빠르다는 것이 Python의 철학인 것이다.

<br>

`Python Tip1` 파이썬은 오버로딩이 안된다.(@연산자로 오버로딩 가능해졌다?) 즉, 같은 함수 이름을 여러개 정의하여 매개변수를 달리하여 사용하는 기법이 허용이 되지 않는다.
파이썬의 단점 중에 속도가 느리다는 점이 있었는데, 오버로딩을 지원하지 않음으로써 속도를 개선했다. (단, 오버라이딩은 사용 가능하다. (매소드 재정의))


### multipledispatch

> 파이썬으로 오버로딩 지원해주는 패키지

```python
from multipledispatch import dispatch

@dispatch(int, int)
def add(x,y):
    return x + y

@dispatch(object, object)
def add(x,y)
    return "%s + %s"%(x,y)

add(1,2)
: 3

add(1,'hello')
: '1 + hello'
```


* 오늘의 명언
<span style="color: orange">자동이 많으면 제약이 많다.</span>

<br>

## 함수의 특징 3가지

```
1. return이 반드시 있어야 한다
- python에서는 return을 생략하면 None을 반환하도록 되어 있다
2. 함수 안에 또 다른 함수를 선언할 수 있다
3. Global, Local
- 함수 안에 없는 값을 return 하게 될 경우 가까운 Global 식별자를 return
- Global 식별자이름 하게 되면 접근, 수정이 가능하다
- 함수 밖에서 함수 안의 식별자에 접근, 수정이 불가능하다
```

<br>

### 함수안의 함수
```python
def a():
   def b():
      return 3
   return b

a()
:<function __main__.a.<locals>.b()>
a()()
:3

def a():
   def b():
      return 3
   return b()

a()
:3

# 둘의 차이가 있다 위에것은 함수를 리턴하는 것이고 밑에꺼는 함수 안에서 함수를 호출하여 값을 리턴
```


`Python Tip2`  return 값이 있는지 없는지 확인하는 방법 1. type() 2. 값을 저장해서 확인하기
<br>

`Python Tip3`  함수를 리스트에 넣어서 계산을 편리하게 할 수도 있다.
<br>

`Python Tip4`  jupyter에서는 두 가지로 호출할 수 있는 함수인지 판단 가능 1. callable 2. 출력
<br>

`Python Tip5`  Python keyword는 총 35개다.
<br>

```python
import keyword

cnt = 0
for x in keyword.kwlist:
   cnt += 1
print(cnt)
:35
```


### 신기한 함수
```python
import matplotlib.pyplot as plt

plt.plot([1,2,3,4,5],[9,3,7,3,9])

# 설명에서 plot[x] => 여기서 [] 대괄호는 리스트가 아니고 옵션이라는 뜻
```
![result](https://user-images.githubusercontent.com/33630505/57130958-1e486280-6dd6-11e9-9ee5-2426f4064ba1.JPG)


#### 추가 공부
1. 동적 프로그래밍에 대해서 찾아보기
2. 피보나치 다른 방법 공부해보기
3. 과제란에 올려주시는거 문제 풀어보기
4. 여태까지 공부했던건 tree형태로 가지치기 map 그려보기
5. Python 철학 처음부터 끝까지 정독해보기

`복습 시간`   17시 28분 ~ 19시/ 총 1시간 32분   



<a id = '5th'></a>
# 2019년 5월 7일 화요일 5th

<First class function/Higher order function 관계 그림 수정>

일급 객체: [git blog](https://gyukebox.github.io/blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EB%8A%94-%EC%9D%BC%EA%B8%89-%EA%B0%9D%EC%B2%B4first-class-citizen/), [tistory](https://rednooby.tistory.com/113)<br>


## First class function

> 함수를 값처럼 사용할 수 있다.

<br>

## Higher-Order-Function

> 함수를 리턴값으로 쓰고, 함수를 인자로 쓰는 함수

<br>

## 함수의 인자로 함수가 들어가는 경우

```
1. map
2. filter
3. reduce  => 여러개 값을 하나의 값으로 축약
```

### 예제
```python
# map
def a(x):
   return x + 1

list(map(a,[1,2,3]))
:[2,3,4]

temp = []
for i in [1,2,3]:
   temp.append(i+1)
else:
   print(temp)
:[2,3,4]

# filter
def b(x):
   return x > 3

list(filter(b,[1,2,3,4,5,6]))
:[4,5,6]

# reduce
from functools import reduce

reduce(lambda x,y:x+y,[1,2,3,4,5])
: 15

add5 = lambda n: n+5

reduce(lambda l, x: l+add5(x), range(10),0)
: 95
# 0 + (0 + 5) => 5 / 5 + (1 + 5) => 11 / .....  

reduce(lambda l, x: l+add5(x), range(10))
: 90
# 0 + (1 + 5) => 6 / 6 + (2 + 5) => 13 / .....

# 초기값 list (O)
reduce(lambda l, x: l+[add5(x)], range(10),[])
: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# 초기값 tuple (O)
reduce(lambda x,y : x+(y,), [1,2,3,4], ())
: (1, 2, 3, 4)

# 초기값 set (X)
reduce(lambda x,y : x+{y}, [1,2,3,4], set())
: TypeError

# 초기값을 주는지 안 주는지에 따라 결과값이 달라진다
```

> filter는 predicate function => True or False를 되돌려 주는 함수

<br>

`Python Tip1`  shift + tab 했을 때 나오는 * iterables와 iterable는 차이가 있다. 별표가 있는 것은 iterable 여러개가 오고 별표가 없는 것은 한개만 온다

<br>

### 별표의 총 7가지 방법

```
1. Unpacking 방법에서 나머지
2. Only keyword
3. Variable Positional
4. Variable Keyword
5. Unpacking (벗겨내기, list 쪼개기)
6. Unpacking (dictionary)
7. import에서 모두
```
<br>

### Annotation

```python
def xx(x:int) -> int:
   return x

xx.__annotations__
{'x': int, 'return': int}
# 타입을 표시해줌

xx(3.0)
xx('hi')
# 둘다 가능
```
<br>

### 인자에 default값 넣는 꼼수
```python
n = 0
def a(n):
   return n
a(n or 3)
# 인자에 디폴트를 사용할 수는 없지만 이렇게 흉내는 낼 수 있다
```
<br>

## 식의 종류

1. `조건식`
- 3 if a > 0 else 6
2. `함수식` (lambda: 익명함수)
- (lambda x: x + 1)(2)
- (lambda x, y=1: x + y)(3)
- (lambda * x: x)(4,1,2)
- list(map(lambda a:a+8, [1,2,3,4,5]))
- lambda 파라미터 : return 값
3. `반복식`
- (x for x in range(10))
- Haskell에서 가져옴

![haskell_comprehension](https://user-images.githubusercontent.com/33630505/60572427-8b657d80-9db0-11e9-82ae-5318f85e08b8.JPG)
<br>

```python
# 조건식 + 반복식

integer = [1,-1,4,0,44,-34,-42,14]
['양수' if i > 0 else ('음수' if i < 0 else 0) for i in integer]
list(map(lambda i: '양수' if i > 0 else ('음수' if i < 0 else 0), integer))

: ['양수', '음수', '양수', 0, '양수', '음수', '음수', '양수']

# 조건식 + 반복식 vs 반복식 + 조건식

li1 = [x if x%2==0 else None for x in range(10)]
li1
: [0, None, 2, None, 4, None, 6, None, 8, None]

li2 = [x for x in range(10) if x%2==0]
li2
: [0, 2, 4, 6, 8]

# 조건식이 왼쪽에 있을때 오른쪽에 있을때 사용법에 있어서 차이가 있으니 조심하자
# 기본적으로 파이썬은 왼쪽에서 오른쪽으로 실행!
```

<br>

`Python Tip2`  Local, Argumentation은 stack에 저장되고 Parameter는 heap영역에 들어간다
<br>

`Python Tip3`  default값에 mutable값을 넣으면 값을 공유한다?, 값이 고정된다?


<br>

```python
import time
time.time()
1557232234.682229 # 수행할 때마다 값이 변한다

def a(t=time.time()):
   return t
a()
1557232228.6397958 # 값이 고정된다
```
<br>

`Python Tip4`  bytearray와 frozenset은 리터럴이 없다

<br>

### Return의 3가지 형태

```
1. 자기는 변하지만 return이 None
- ex) append, extend
2. 자기 자신이 변하지 않고 return 값이 있다
- ex) count, index
3. 자기 자신도 변하고 return 값도 있다
- ex) pop
```

`mutable`에 사용하는 함수중에서 return 값이 None인 경우가 종종 있다. ex) append, extend


<br>

```python
# 1
def xx(y, x=[]):
   return x.append(y)
# 사실 엉터리 코딩 x.append(y)는 None을 리턴하고,
# 함수 밖에서 x 리스트에 접근도 할 수 없기 때문에 but 인자에 기본값 x list 말고
# 외부에 선언된 list 넣으면 list 확인 가능
# default값을 mutable로 사용하면 heap영역에 들어간다

# 1-1
def xx(list):
    return list.append(3)
x = [1,2]
xx(x)
x
: [1,2,3]

# 2
def xx(y, x=[]):
   x.append(y)
   return x
# xx함수를 호출 할때마다 x 리스트가 계속 변한다

def xx(y):
   x = []
   x.append(y)
   return x
# xx함수를 호출하면 원소가 하나인 리스트 반환
```
<br>

## 파이썬 변수의 유효 범위(Scope)

> 유효 범위 규칙(Scope Rule)은 변수에 접근 가능한 범위, 변수가 유효한 문맥범위를 정하는 규칙

### LEGB

```
1. Local : 함수 내 정의된 지역변수
2. Enclosing Function Local : 함수를 내포하는 또다른 함수 영역
- 함수 안의 함수가 있는 경우 함수와 함수 사이
3. Global : 함수 영역에
4. Built-in : 내장 영역
- 함수 안의 함수가 있는 경우 함수안의 함수에서 함수 밖의 변수를 사용?

우선순위 => L > E > G > B
```

#### LEGB 우선순위 확인 예제
```python
x = 'global'
def outer():
    #x = "local"

    def inner():
        #nonlocal x
        #x = "nonlocal"
        #print("inner:", x)
        return x

    inner()
    #print("outer:", x)
    return x

outer()
```

`Python Tip5`  함수 중첩은 3번 이상 하지 않는 것이 좋다.


<hr>

### 신기한 기능

```python
import seaborn as sns
tips = sns.load_dataset('tips')

tips.head(10) # 앞에 10개만 보여줘
tips.tail(10) # 뒤에 10개만 보여줘
tips.sample(10, replace = True) # 랜덤으로 10개 보여줘
```
#### 랜덤 10개
![python](https://user-images.githubusercontent.com/33630505/57372344-584ba700-71d0-11e9-86f1-21da17a4fc73.JPG)


#### 추가공부

1. 변수, 인자와 힙, 스택간의 관계


`복습 시간`  19시 10분 ~ 21시 40분 / 총 2시간 30분  



<a id = '6th'></a>
# 2019년 5월 9일 목요일 6th


## 함수형 패러다임

멀티 프로세싱 기법에 최적화된 패러다임 <br>
빅데이터 처리시 효율적이다 <br>


### 함수형 프로그래밍의 특징

```
1. 코드가 간결해진다
- 내부 구조를 몰라도 input, output만 알면 사용 가능
2. 수학적으로 증명이 가능하다
3. for, while문을 자제하고 iter를 사용한다
4. 수학적 증명이 필요하기 때문에 구현이 어렵다
- 단, python에서는 multi paradiam이기 때문에 적절히 혼용 가능
5. 디버깅, 테스트가 용이하다
6. 모듈성, 결합성이 있다.
7. mutable은 처리하지 않는다
```

## 반복을 줄이는 5가지 방법
(for문을 최대한 쓰지 않고)

```
1. Iterator
- 메모리 효율적, 속도 빠름
- 실행시에 메모리 할당
2. Generator
3. Comprehension
- for를 쓰지만 for를 쓰지 않는 기법
4. Recursive function
- 메모리 효율, 속도면에서 성능이 좋지않아 사용안함
5. map, filter, reduce
```

`iterable` 1. iterator로 바꿀 수 있는 2. 순회, 반복가능 (요소 하나씩 뽑아냄) 3. for 뒤에 사용할 수 있는 container


<br>

## 왜 함수형 패러다임에서 반복을 줄여야 하는가?

<span style="background-color: orange">for문 처럼 대입하는 것은 함수형 패러다임에 맞지 않고 수학적 증명과는 거리가 있기 때문이다. </span>

### Iterator

<span style="background-color:orange">데이터 스트림을 표현하는 객체, next()메소드를 사용하여 다음 요소를 가져온다</span>

```python
a = [1,2,3]
b = iter(a)
next(b)

:1
# 실행할 때마다 index 0번지 부터 하나씩 뽑아낸다

# iterator를 객체화하면 iterator의 성질도 잃고 객체화 하기 전 iterator의 요소 전부를 뽑아냄으로 주의!
a = [1,2,3]
b = iter(a)
next(b)
: 1
list(b) # tuple(b), set(b) 다 똑같음
: [2,3]
next(b)
: StopIteration
```

### Iterator vs Iterable

```python
from collections.abc import Iterable, Iterator

set(dir(Iterator))-set(dir(Iterable))
: {'__next__'}
```

### Generator

<span style="background-color:orange">Iterator를 생성해주는 Function, 그리고 일반 함수와 비슷해 보이지만 Generator는 yield를 포함한다는 점에서 차이가 있다</span>

두 가지 방법으로 만들 수 있다
1. generator 표현식(tuple)
2. yield

```python
# tuple로 만들기
a = (x for x in range(10))
a

: <generator object <genexpr> at 0x000001EBD55D0DE0>

# yield로 만들기
def x():
   yield 1
   yield 2
y = x()
next(y)

:1
```

**주의** iterator와 generator는 scope를 초과하면 StopIteration 에러가 뜬다.


**file**을 불러오면 generator 처럼 행동한다.


```python
x = open('file.txt')
next(x)
: '안녕\n'
next(x)
: '반가워\n'
```
<br>

### reversed

> 원본 데이터를 뒤집고 iterator로 만드는 함수                                              

```python
rev = reversed([1,2,3,4])
type(rev)
: list_reverseiterator

next(rev)
: 4

list(rev)
: [3,2,1]
```


### Iterator vs Generator & Generator vs Function

<span style="color: skyblue">Iterator vs Generator</span>

Iterator는 반복가능한 객체 그리고 데이터 스트림을 표현하는 객체라고 한다. <br>
예를 들어 list는 반복가능한 자료형 즉 iterable이지만 iterator는 아니다. <br>

```python
for x in [1,2,3]:
   print(x)
```
이처럼 in 다음에 iterable이 오면 반복해서 요소 하나씩 꺼낼 수 있긴 하다<br>
<span style="color:red">하지만</span> list는 iterator는 아니라고 했다<br>
그렇다면 어떻게 list가 iterator처럼 처리되는가? <br>
<span style="color:red">그 이유는</span> in 다음에 iterable이 오면 iter없이 iterator로 변환 해주기 때문이다
<br>
Generator는 iterator를 생성해주는 Function <br>
따라서 Generator와 iterator는 비슷하다 <br>
하지만 역할이 다르기 때문에 명칭도 다른것!<br>
<br>
<span style="background-color: orange">생성 방식에서 차이가 있고 Iterator는 객체 Generator는 함수</span><br>
<span style="background-color: orange">Generator는 tuple, yield로 만들고 Iterator는 liter() 함수로 만든다</span>
<br>

<br>
<hr>

<span style="color: skyblue">Generator vs Function</span>

**Generator** <br>

Iterator를 만들어주는 것 <br>
반복 가능한 객체를 만들어주는 함수<br>

```python
def generator():
   while True:
      yield from [1,2,3,4] # Cycling 기법
e = generator()
next(e)
:1       # 실행할 때마다 1부터 4까지 계속 반복해서 return

# yield는 return과 비슷하다고 생각하면 된다
```

일반적으로 **함수**는 사용이 종료되면 결과값을 호출한 곳에 반환해주고 함수 자체를 종료 시킨 후 메모리상에서 사라진다 <br>
하지만 yield를 사용할 경우 그 상태로 <span style="color:red">정지</span> 되며 반환 값을 next()를 호출한 쪽으로 전달한다<br>
함수 호출이 종료되면 메모리상의 내용이 사리지지 않고 다음 함수 호출까지 대기한다<br>
다음 함수 호출이 발생할 경우 <span style="color:red">yield이후 구문부터 실행된다</span>
<br>

여기서 generator를 사용하는 <span style="color:red">이유</span>를 알 수 있다 <br>
generator를 사용하면 호출한 값만 메모리에 할당되므로 메모리를 효율적으로 사용할 수 있게된다
<br>
이러한 기법을 <span style="color: orange">Lazy Evaluation</span>이라고 한다 <br>
Lazy Evaluation은 계산 결과 값이 필요할 때까지 계산을 늦추는 방식이다 <br>
Lazy Evaluation은 속도가 느리다는 단점이 있지만 파이썬에서는 <span style="color:red">내부적으로 최적화</span> 되어 있어 속도가 빠르다<br>

참고: [tistory](https://bluese05.tistory.com/56)<br>

<br>

### Comprehension

<span style="background-color:orange">Iterable한 객체를 생성하기 위한 방법</span>

```
1. List
2. Set
3. Dictionary
```

```python
# list

# for 앞에는 식이 오면된다
a = [(x,y) for x in range(10) for y in range(20)]
: [(0,0),(0,1),(0,2),......(9,19)]

# set
b = {x+1 for x in range(10) if > 5}
: {7, 8, 9, 10}

# dictionary
c = {x:1 for x in range(10) if x>5}
: {6: 1, 7: 1, 8: 1, 9: 1}

# generator 표현식 (tuple)
d = (x for x in range(10))
d
: <generator object <genexpr> at 0x000001EBD5645DE0>
```

### Recursive function

재귀함수 <br>

```python
def fib(num):
    if num < 3:
        return 1
    return fibB(num-1) + fibB(num-2)
fib(10)  # 10번째 항   (1 1 2 3 5 8 13 21 34 55)
: 55
```

### itertools

```
1. cycle
2. count
3. islice
4. chain
```

#### cycle

```python
from itertools import cycle

for x in cycle(iter([1,2,3])):
    print(x)
    if(x==3):
        break
:
1
2
3
```

#### count

#### islice

#### chain

### all, any

> all은 전부다 True일때 True를 반환하고 False가 하나라도 있으면 False를 반환한다. <br>
> any는 하나라도 True이면 True를 반환하고 전부다 False이면 False를 반환한다.

```python
# all
all_pred = lambda item, *tests: all(p(item) for p in tests)
: all_pred([1,2,3], sum, max)
: True
all_pred([0,1,2,3], sum, min)
: False

# any
any_pred = lambda item, *tests: any(p(item) for p in tests)
any_pred([0,1], sum, min)
: True

any_pred([0,1], min)
: False
```

`복습 시간`  18시 ~  19시 50분/ 총 1시간 50분



<a id = '7th'></a>
# 2019년 5월 10일 금요일 7th

<span style="background-color:orange">함수의 중첩으로 가능한 것들</span>
```
1. Closure
2. Decorator
```


## Closure

> 함수를 중첩시키고 함수를 리턴하는 함수. 클로저는 보통 함수를 쉽게 변형할 수 있는 기법이다.


```python
def add(x):
    def addd(y):
        return x + y
    return addd

# 3더해주는 함수
add(3)(10)
: 13

# 4더해주는 함수
add(4)(10)
: 14
```


Closure: [github blog](https://nachwon.github.io/closure/)


## Decorator

> 함수나 클래스를 추가, 수정해서 재사용 가능하게 해주는 것 <br>
> 데코레이터를 사용하려면 함수 중첩이 있어야 하고, 함수를 파라미터로 받아야 하고, 중첩된 inner 함수를 return 해야 한다.  

### 예제를 통한 decorator 이해하기

#### 중첩 X, 함수 리턴 X 일때
```python
def login_check(fn):
    id=input('id: ')
    if(id=='jh'):
        print("jh님 안녕하세요")
        fn()
    else:
        print('존재하지 않는 회원입니다')

@login_check
def home():
    print('jh님의 home page 입니다')

id: [jh                       ]  # home 함수 선언시 input창이 뜬다

: id: jh
  jh님 안녕하세요
  jh님의 home page 입니다

# 다른 아이디를 입력했을 때
id: [hj                       ]

: id: hj
  존재하지 않는 회원입니다.

callable(home)
: False
callable(login_check)
: True
```

> 결론 데코레이터 아님.

#### 함수 return X, 문자열 return일 때

```python
def login_check(fn):
    def inner():
        id = input('id: ')
        if(id=='jh'):
            print("jh님 안녕하세요")
            fn()
        else:
            print("존재하지 않는 회원입니다")
    print('ㅋㅋㅋㅋ')
    return 'a'


@login_check
def home():
    print('jh님의 home page 입니다')

: ㅋㅋㅋㅋ

callable(home)
: False
callable(login_check)
: True
```

> 결론 당연한 결과지만 login_check함수 안에서 inner 함수를 return 하지 않으면 inner함수를 사용할 방법이 없다. <br>
> 그리고 함수위에 '@login_check'을 사용하면 @밑으로 함수선언을 하게되면 @밑 함수는 '@login_check'의 인자로 들어가게 된다.<br>
> 결국 '@함수이름'의 return값이 함수이름이 아니게되면 @밑에 선언된 함수는 not callable이게 된다.
> 따라서 데코레이터 아님.

#### 함수 return X, 함수 호출 return 일때

```python
def login_check(fn):
    def inner():
        id = input('id: ')
        if(id=='jh'):
            print("jh님 안녕하세요")
            fn()
        else:
            print("존재하지 않는 회원입니다")
    return inner()

@login_check
def home():
    print("jh님의 home page 입니다")    
: id: [jh                       ]    
  id: jh
  jh님 안녕하세요
  jh님의 home page 입니다

: id: [hj                       ]   
  id: hj
  존재하지 않는 회원입니다

callable(home)
: False
```

> 함수 return X, 문자열 return일 때와 결과가 같다. 이로써 '@login_check'을 사용하면 밑에 선언된 함수는 <br>
> login_check함수의 인자로 들어가 login_check함수의 return 값을 반환한다는 사실이 확실해졌다.<br>
> login_check함수의 return 값은 inner함수 호출이고 inner함수의 호출값의 return은 None 이기 때문에 <br>
> home함수의 return 값은 None <br>
> 따라서 데코레이터 아님.

#### 어노테이션 X

```python
def login_check(fn):
    def inner():
        id = input('id: ')
        if(id=='jh'):
            print("jh님 안녕하세요")
            fn()
        else:
            print("존재하지 않는 회원입니다")
    return inner

def home():
    print("jh님의 home page 입니다")

home()
: jh님의 home page 입니다

login_check(home)()
:
id: jh
jh님 안녕하세요
jh님의 home page 입니다

login_check(home)()
:
id: hj
존재하지 않는 회원입니다
```

> 결론 home함수는 따로 작동하기 때문에 기능확장이라 할 수 없어서 데코레이터가 아님.


#### 함수 파라미터 O, 함수 중첩, 함수 이름 return O

```python
def login_check(fn):
    def inner():
        id = input('id: ')
        if(id=='jh'):
            print("jh님 안녕하세요")
            fn()
        else:
            print("존재하지 않는 회원입니다")
    return inner

@login_check
def home():
    print("jh님의 home page 입니다")

home()
: id [jh        ]
  id: jh
  jh님 안녕하세요
  jh님의 home page 입니다
home()
: id [hj        ]
  id: hj
  존재하지 않는 회원입니다
```

> 결론 데코레이터 맞음! (home함수에 login_check함수 기능 추가)

#### 데코레이터 기능 수정하기

```python
def trans_odd(fn):
    def inner(x):
    	print('홀수 입니다')
	result = fn(x+1)
	return result
    return inner

@trans_odd
def odd(x):
    if (x%2==0):
        return "홀수 입니다"
    else :
        return "홀수가 아닙니다"
odd(4)
: '홀수가 아닙니다'
odd(3)
: '홀수 입니다'
```


## Closure  vs  Decorator


## partial

> 부분을 대체하여 클로저 처럼 사용하는 함수

```python
from functools import partial

add3 = partial(add,3)  # add의 두개 인자중 하나를 3으로 대체
add3(7)
: 10
```


**Python Tip1** locals(), globals() 함수는 함수내에서 사용하면 메모리상에 올라간 local, global 변수를 각각 확인 할 수 있다


```python
x = 10
def a():
   y = 20
   def b():
      a = 1
      print(locals())
      b = 2
   return b()
a()
:{'a':1}
# b = 2는 print문 출력 이후에 있기 때문에 locals에 포함되있지 않다
 선언할 때는 모든 것이 메모리에 할당 되지만 실행할때는 순서대로 할당되기 때문이다
```

## name, namespace, module 그리고 __name__

<kbd>name</kbd>은 말 그대로 이름을 붙여주는 즉, 변수명 혹은 식별자라고 생각하면 된다<br>
<kbd>module</kbd>은 파이썬 코드를 담고 있는 파일이다, 좀 더 자세하게 말하면 클래스, 함수, 변수명의 리스트가 들어 있다고 보면 된다 <br>
<kbd>namespace</kbd>는 names(변수명들)을 담을 수 있는 공간이다, 모듈은 자신의 유일한 namespace를 갖고 있으며 모듈의 namespace이름은 보통 모듈이름과 같다. 그래서 동일한 모듈내에서 동일한 이름을 가지는 클래스나 함수를 정의할 수 없다. 또한 모듈은 각각 독립적이기 때문에 동일한 이름을 갖는 모듈을 갖을 수 없다. <br>

<br>
> import를 통해 namespace와 __name__에 대해서 자세히 알아보자

### import 방법은 3가지가 있다

```
1. import <module_name>
- 모듈의 name을 prefix로 사용함으로써 모듈의 namespace에 접근할 수 있다
2. from <module_name> import <name,>
3. from <module_name> import *
```

```python
1. import <module_name>
import sys
sys.path

# sys는 모듈, path는 sys모듈의 namespace에 담겨 있는 name
# 모듈 prefix sys를 통해 namespace path에 접근

2. from <module_name> import <name,>
from sys import path
path
sys.path

# 모듈 prefix를 사용하거나 사용하지 않고 둘다 접근 가능하다
# 단, del path를 하거나 name을 재정의 하게되면 모듈의 name을 사용할 수 없게된다
# 몇개의 name만 필요하고 name를 명확하게 구분할 수 있는 상황에서는 써도 무관하다

3. from <module_name> import *
from sys import*
sys.path
path

# 2번방법과 동일, 그러나 모듈에 있는 모든 name을 직접 현재 namespace로 가져오게된다
# 말할 것도 없이 전체를 import하면 name을 쓰는데 제약이 많이 생긴다
```
Namespace Binding: [slideshare](https://www.slideshare.net/dahlmoon/binding-20160229-58415344)<br>
<br>

**__name__**<br>

import를 하면 해당 모듈의 names를 namespace에 dict타입으로 할당하는 것을 보았다<br>
이때 import한 모듈의 __name__은 파일명이 된다 <br>
<br>
```python
from sys import *
sys.__dict__  # namespace 불러오기

'__name__': 'sys',
 '__doc__': "This module..........
```
<br>
이번엔 import를 하지 않고 main script(최상위 스크립트 환경)에서 직접 shell에서 실행하는 경우에 python interpreter가 최초로 파일을 읽어 실행하는 경우를 생각해보자 <br>
이때는 모듈이름 해당 파일 이름이 아닌 __main__가 된다 <br>

```python
__name__

'__main__'
```
<br>

> 따라서 만약에 '이 파일이 interpreter에 의해 실행되는 경우라면' 이라는 의미를 갖는다

```python
if __name__ == '__main__':
   main()

if __name__ == '__main__':
	print 'This program is being run by itself'
else:
	print 'I am being imported from another module'
```

__name__의 의미 : [tistory](https://pinocc.tistory.com/175)<br>


## Python의 모든 것은 Object(객체)이다

<span style="background-color: orange">Object</span>는 python이 data를 추상화 한 것이다<br>
쉽게말해 프로그래밍으로 구현될 대상, 현실에 존재하거나 상상가능한 대상을 특징지어 구현될 대상이라고 할 수 있다<br>
그리고 python 프로그램의 모든 data는 객체나 객체간의 관계로 표현된다<br>

> John von neumann's stored program computer model을 따르고 또 그 관점에서 코드 역시 객체로 표현된다

### 객체를 구현하려면?

객체를 구현하기 위한 설계도 및 틀을 <span style="color:red">Class(클래스)</span>라고한다<br>
<br>
Class를 실제로 프로그래밍할때 사용하려면 클래스 선언, 메모리 할당, mapping 3가지 단계가 필요하다<br>
이해를 돕기 위해 java의 경우를 예로 들겠다 <br>

```java
package test;

import test.Add;                                 // 1. 선언된 클래스 import (클래스 선언)

public class Test{
	public static void main(String[] args){
		Add add = null;                  // 참조변수 선언
		add = new Add();                 // 참조변수에 인스턴스에 대한 참조(참조값) 할당
	        //Add add = new Add(); 위와 동일 // 메모리에 생성되어 저장된 객체 처리 가능
		                                 // add는 레퍼런스 변수, 인스턴스를 가리키는값
						 // 참조변수를 사용하여 멤버변수, 메소드 접근가능
		System.out.print(add.sum(3,4));
	}
}
```

```
1. 클래스 사용을 위해 클래스 선언
2. 클래스 사용시 메모리에 생성
3. Index table에 참조변수와 메모리 연결을 위한 주소를 매핑하는 참조값이 만들어진다
4. 참조값은 JVM이 자동적으로 생성
5. 참조값을 사용하게되면 참조값에 연결된 메모리 즉, 인스턴스를 사용한다는 것

참조 ≒ 참조값(Hash code)
```

> 그렇다면 Python에서 객체의 의미를 살펴보자

```python
a = 1
print(type(a))
a = 3.2
print(type(a))
print(a)
a.__class__

: <type 'int'>
  <type 'float'>
  3.2
  int

# Python에서는 선언과 할당을 동시에 하면서
# 할당값에 의해 변수의 타입(객체의 타입)이 결정되고 naming된 변수 이름이 인스턴스가 되는 것이다
```


참조와 참조변수 : [tistory](https://dohe2014.tistory.com/entry/%EC%B0%B8%EC%A1%B0reference%EC%99%80-%EC%B0%B8%EC%A1%B0%EB%B3%80%EC%88%98reference-variable)<br>

**None & 객체** 객체가 있는지 없는지 구분할때 None을 활용해 확인할 수 있다


### 주의
```python
None == False
: False

# False는 0이라는 값과 매칭되어 있기 때문에 즉, 0이라는 객체이기 때문에
# 존재론적 관점에서 None이 아니다.
```


`복습 시간`  18시 22분 ~ 20시 / 총 1시간 38분


<a id = '8th'></a>
# 2019년 5월 13일 월요일 8th

## Class (클래스)

1. 값
2. 메소드

두 가지로 이루어져있다

```python
class A:
    x = 1
    def a(self, y):
        self.y = y
	return y

A.x      # 클래스로 클래스 변수 접근
a = A()  # 인스턴스 생성
a.x      # 인스턴스 a로 x(클래스 변수)접근
vars(a)  # 인스턴스 변수 확인
a.a(3)   # 인스턴스 a로 a() (메소드) 접근
vars(a)
A.a(a,5) # 클래스로 인스턴스 a와, 5을 인자로 넘겨주고 a함수 접근
vars(a)  # 인스턴스 변수는 인스턴스마다 고유로 갖을 수 있는 변수 이다

: 1
  1
  {}
  3
  {'y':3}
  5
  {'y':5}

dir(A)
dir(a)

: ['__class__','__dict__',.......,'a','x']
  ['__class__','__dict__',.......,'a','x','y']


class B:
    x = 2
    def b():
        print('Access class')

B.b()
b = B()
b.b()

: Access class
  TypeError

class B:
    x = 2
    def b(self):		
    print('Access instacne')

B.b()
b = B()
b.b()
B.b(b)

: TypeError
  Access instance
  Access instance
```

<br>
<hr>

### Instance (인스턴스)

클래스를 사용하려면 인스턴스화 해야한다 <br>
인스턴스는 클래스의 값과 메소드에 접근이 가능하다

```python
class A(object):                   # 기본적으로 class는 object라는 클래스를 상속받는다
	                           # 명시하지 않아도 default로 상속한다
	def __init__(self):
		print('init')	   # object는 공통적인 속성들을 모아둔 class(추상화, 상속)
# init은 내부적으로 인스턴스화 할때 호출함 / 생성자라고도 한다
# init의 추상화 내용이 마음에 들지 않는 경우 변경 가능함(다형성)  

A()    # 클래스를 호출하면 인스턴스 객체를 반환한다
: init
  <__main__.A at 0x23b6c143a90>
a = A()  # 클래스를 호출함으로써 인스턴스 할당
         # 인스턴스화를 하면 클래스에 정의되어 있는 기능 사용할 수 있다
```

<br>

### 클래스 변수 vs 인스턴스 변수

> 클래스 변수는 모든 인스턴스들이 어트리뷰트와 메서드를 공유한다. <br>
> 반면 인스턴스 변수는 각 인스턴스 별로 개별적인 값을 갖는다. <br>
> 인스턴스는 클래스내에 정의된 모든 것을 사용할 수 있지만, <br>
> 클래스는 여러 인스턴스를 생성하기 때문에 인스턴스 변수에 접근 할 수 없다.


### 예제로 살펴보기
```python
class A:
    a = 1                  # class variable, attribute
    def __init__(self, y):
	     self.y = y     # instance variable, attribute


# 접근 예제

class A(object):
    x = 1
    y = 2

    def add_a(self, x, y):
        sum_a = x + y
        self.sum_a = sum_a
        return sum_a

a = A()
a.add_a(3, 10)
vars(a)
a.sum_a = 0
vars(a)
A.sum  
A.x
vars(A)

: no output
  13
  {'sum': 13}
  {'sum': 0}
  AttributeError   # 클래스로 인스턴스 변수에 접근했기 때문
  1
  mappingproxy({'__module__': '__main__',
              'x': 1,
              'y': 2,
              'add_a': <function __main__.A.add_a(self, x, y)>,
              '__dict__': <attribute '__dict__' of 'A' objects>,
              '__weakref__': <attribute '__weakref__' of 'A' objects>,
              '__doc__': None})

class C:
    tricks = []

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

a = C('jh')
b = C('other')

a.add_trick('first')

a.tricks
b.tricks

: ['first']
  ['first']

# 인스턴스 별로 개별 리스트 변수를 만들고 싶다면 인스턴스 변수에 리스트 할당해야함

class D:
    def __init__(self, name):
        self.tricks = []
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

a = D('jh')
b = D('other')

a.add_trick(1)
a.add_trick(2)

b.add_trick('first')
b.add_trick('second')

a.tricks
b.tricks

: [1, 2]
  ['first', 'second']
```

<br>

**실행 순서** 메소드와 변수가 이름이 같을때 변수를 먼저 접근하기 때문에 주의 해야 한다



**self** 는 인스턴스라고 생각하면 된다. 메소드의 인자로 self가 있는건 self 자리에 인스턴스를 인자로 넘겨받아 해당 함수를 접근해서 사용 가능하게 된다는 의미로 받아들이면 된다


<br>
<hr>


**Type casting** a = list({1,2,3}) 리스트 클래스의 괄호 안에 dict타입을 넣게 되면 리스트 타입으로 변경된다. 파이썬에서는 타입 변경이 없기 때문에 타입 변경시에는 클래스 안에 인자로 넣어 인스턴스화 하여 바꿔준다. 단, 모든것이 되는것은 아니다  


```python
a = int('0')
a

: 0
# 원래는 문자를 정수형으로 타입 변환이 되지 않지만
# 문자형으로 된 숫자를 정수형 타입으로 변환되는 경우가 몇가지 있다
# 이는 init으로 바꿀 수 있게 해둔 것이다 (특수한 경우이다)

b = int('b')
: ValueError
# invalid literal for int() with base 10: 'b'
# 기본적으로 문자를 정수형으로 바꿀 수 없다.
```

<br>

### 클래스 밖에 있는 함수를 클래스의 지역 변수에 할당 할 수 있다

```python
def out(self, x):
    self.array.pop(x)
    return self.array

class Array:
    def __init__(self):
        self.array = []
    def add(self, x):
        self.array.append(x)

    def add5(self, x):   # 클래스 내에 다른 함수를 호출 가능
        self.add(x+5)
	      return self.array

    pop_method = out


array = Array()

array.add(1)
array.add(2)
array.add(3)

vars(array)
: {'array': [1, 2, 3]}

array.pop_method(0)
: [2, 3]

vars(array)
: {'array': [2, 3]}

array.array
: [2, 3]


array.add5(1)
: [2, 3, 6]
```

<br>

### classmethod, staticmethod

```python
class A:
    a = 1
    def __init__(self,y):
        self.y = y   
    def getx(self):
        return self.y

    @classmethod
    def getxx(cls):  # class method
        print('a')  

    @staticmethod    # 똑같이 함수 처럼 사용
    def y(cls):
        print('b')
```

### 객체 지향의 특징

```
1. 캡슐화
- 재활용 가능
- 파이썬에서는 기본적으로 외부에서 클래스 접근 가능
- descriptor로 클래스 접근을 막을 수도 있다
2. 추상화
- 구체적인것과 추상적인 것을 분리시킨다
- 특징을 뽑아 객체로 구현
3. 상속
- 추상화의 반대
- 추상화한 클래스를 물려받아 구체적으로 원하는 대로 바꾸어 사용 가능
4. 다형성
- 다양한 결과를 낼 수 있다
```

**Python Tip1** 유지보수를 해야 한다고 느끼면 객체지향 프로그래밍을, 멀티프로세스나 다양한 문제를 다양한 방식으로 풀고 디버깅을 편하게 해야 한다고 느끼면 함수형 프로그래밍을 하면 된다


**Python Tip2** 동적으로 인스턴스 변수, 메소드 추가 가능 but 좋지 않은 방식


```python
# ex)
class Dyn():
	name = 'jh'
	def name(self):
		print('nothing')
dy = Dyn()
dyn.d()
vars(dy)
dy.name
dy.name = 'jane'
dy.name
vars(dy)

: 'nothing'
  {}
  'jh'
  'jane'
  {'name': 'jane'}
```

<br>

## Object, Type

<span style="color: red; font-size: 30px;">Object</span>는 <span style="color: red; font-size: 30px;">최상위 객체</span><br>
<span style="color: red; font-size: 30px;">Type</span>는 <span style="color: red; font-size: 30px;">meta class</span><br>


<br>
<hr>

### Naming

#### 1. camelCase(카멜 표기법)

첫 문자는 소문자로 표기하고, 그 다음 단어의 첫 시작은 대문자로 표기한다 <br>
원래는 첫 문자는 대,소문자 구분없었지만 요즘은 소문자로 쓰는 방법이 카멜 표기법 이다<br>
ex) helloWorld
<br>
**함수명은 이 표기법을 권장한다. 단 ,소문자 + underscore를 쓰기도 한다**

#### 2. PascalCase(파스칼 표기법)

첫 문자를 대문자로 표기하고, 그 다음 단어의 첫 시작도 대문자로 표기한다 <br>
ex) HelloWorld
<br>
**클래스명은 이 표기법을 권장한다. 단, 이미 만들어져 있는 클래스는 소문자로 시작한다**

#### 3. snake_case(스네이크 표기법)

한 단어마다 _ (underscore)를 붙여 이어나가는 표기법이다 <br>
ex) hello_world
<br>
**모듈은 이 표기법을 권장한다. 내장 함수도 보통 스네이크 표기법을 따른다**

#### 4. 전부 대문자

전부 대문자를 쓸 경우 상수처럼 쓴다 (관례)<br>
ex) NUMBER = 10

<br>

c.f 패키지는 소문자로 구성한다


`복습 시간`  18시 30분 ~ 20시 20분 / 총 1시간 50분



<a id = '9th'></a>
# 2019년 5월 14일 화요일 9th


## Design pattern

20 ~ 30가지가 있다 <br>
주로 사용하는 coding 방식, 웹에서 주로 이용 (MVC 패턴 같은 것들)


## Class inheritance

```python
class Door:
    colour = 'brown'

    def __init__(self, number, status):
        self.number = number
        self.status = status

    @classmethod              
    def knock(cls):       
        print("Knock!")

    @classmethod
    def paint(cls, colour):  # 클래스, 인스턴스 모두 사용 가능하고
        cls.colour = colour  # 클래스 변수 바꾼다
                             # 클래스로 접근한다 / 클래스만 값에 접근 가능
    @classmethod
    def paint2(cls, number):  # number는 인스턴스 변수
        cls.number = number   # classmethod는 클래스 변수만 바꿀 수 있다
    def open(self):
        self.status = 'open'

    def close(self):
        self.status = 'closed'

class SecurityDoor(Door):      
    pass


Door.knock()
door = Door(1, 'open')
vars(door)
door.knock()
door.paint('red')  # 인스턴스로 클래스 변수 바꿀 수 있음 (@classmethod)
door.colour
Door.colour
door.paint2(2)
door.number
Door.paint2(3)     # 클래스만 인스턴스 변수 바꿀 수 있음 (@classmethod)
door.number
Door.number

: Knock!
  {'number': 1, 'status': 'open'}
  Knock!
  'red'
  'red'
  1
  1
  3
```

### mappingproxy


### 클래스를 상속 받으면 원래 클래스 변수를 공유한다

**※ 메모리 번지를 공유**

```python
sdoor = SecurityDoor(1, 'closed')

print(SecurityDoor.colour is Door.colour)  
print(sdoor.colour is Door.colour)

:True
 True
```

## Composition (합성)

> 상속을 하지 않고 클래스내에 객체를 불러와 다른 클래스의 일부 기능을 사용하는 방법.

```python
class SecurityDoor:
    locked = True

    def __init__(self, number, status):
        self.door = Door(number, status) # Door의 객체를 갖게 한다

    def open(self):
        if self.locked:
            return
        self.door.open()

    def __getattr__(self, attr):        # try except와 비슷
        return getattr(self.door, attr) # Door의 attr를 가져와라 (상속을 비슷하게 사용)


class ComposedDoor:
    def __init__(self, number, status):
        self.door = Door(number, status)

    def __getattr__(self, attr): # 없으면 가져와라
        return getattr(self.door, attr) # 바꾸지 않고 불러와서 사용할 때
```

```python
class A:
	x = 1

a = A()
hasattr(a,'x')
getattr(a,'x')

:True
 1
```

**※ Design pattern에서는 상속대신 합성을 주로 사용한다**

Composition : [blog](https://www.fun-coding.org/PL&OOP1-11.html)<br>

## 예외처리문

> Python의 철학중 양해를 구하기보다 용서를 구하기가 더 쉽다라는 것이 있다. 이처럼 Python에서 예외처리는 빠질 수 없는 부분이다.

```
하지만 DataScience 분야에서는 많이 사용하지는 않는다.
웹이나 자동화 등 사용자로부터 입력을 받거나 외부 조건에 의해 오류가 많이 날 수밖에 없는 환경에서는 굉장히 중요하다
따라서 오류가 나더라도 중단하지 않고 실행을 시켜야 하는 경우에 대비해서 예외처리문을 삽입하는 것이다
```

### 예외처리 구조

```python
a = {'a':1}
try:
    t = a['b']
except:   # 에러가 나면 실행 / 에러종류에 따라 여러개 만들 수 있음
    print('except')
else:     # 에러가 나지 않으면 실행
    print('else')
finally:  # 에러 상관없이 실행
    print('finally')

# try, except가 예외처리의 필수   
```

### 응용

```python
a = {'a':1}
try:
    t = a['n']
except :
    print('all')
except KeyError:
    print('except')

# SyntaxError => except 혼자 오는 것은 마지막에 와야 한다

a = {'a':1}
try:
    t = a['n']
except KeyError as f:   # 에러에 대한 상세 표시
    print(f)
except:
    print('except')

:'n'

a = {'a':1}
try:
    t = a['n']
except Exception as f:   # Exception 상속(상위 에러)
    print(f)
except:
    print('except')
```

**syntax error, runtime error** syntax error는 구문오류로 실행자체가 안된다, runtime error는 실행은 되지만 값이 나오지 않는다


### 에러를 강제로 발생시키는 방법

```python
def x(a):
	if a > 0 :
		raise
	else:
		raise SyntaxError

x(3)
x(-3)

: RuntimeError
  SyntaxError

a = 3
assert a > 4

AssertionError
# 해당 조건이 만족하지 않으면 에러 발생
```

## 다른 사람 것을 고쳐 쓰는 방법

```
1. 상속후 오버라이딩
2. 데코레이터

# 오버로딩은 같은 이름의 메소드를 가지면서 매개변수 유형은 다를때 서로 다른 메소드로 간주하는 것
# 파이썬에서는 오버로딩이 지원되지 않아 같은 이름의 메소드를 정의할경우 재할당이 된다
```

**Mangling** 맹글링은 소스 코드에서 선언된 함수 또는 변수의 이름을 컴파일 단계에서 컴파일러가 일정한 규칙을 가지고 변형하는 것을 말한다. 이는 보통 객체 지향에서 오버로딩시 함수의 이름이 같고 파라미터만 다를때 알아서 구별할 수 있도록 하는데 사용된다.



`복습 시간`  17시 30분 ~  19시, 20시 ~ 20시 30분 / 총 2시간



<a id = '10th'></a>
# 2019년 5월 16일 목요일 10th


## Duck typing

**미운오리새끼 이야기에서 유래가 되어 오리가 아닌데 오리처럼 행동을 하면 오리라고 간주한다는 개념이다** <br>
**타입을 미리 정하지 않고 실행되었을 때 해당 Method들을 확인하여 타입을 정한다**<br>

### 장점
- 타입에 대해 자유롭다
- 상속을 하지 않아도 상속을 한것처럼 클래스의 메소드 사용이 가능하다

### 단점
- 원치 않는 타입이 들어 갈 경우 오류가 발생할 수 있다
- 오류 발생시 원인을 찾기 어려울수 있다  

```python
class Airplane:
    def fly(self):
        print("Airplane flying")

class Whale:
    def swim(self):
        print("Whale swimming")

def lift_off(entity):
    entity.fly()

def lift_off2(entity):
    if entity.swim():
        entity.fly()    

airplane = Airplane()
whale = Whale()

lift_off(airplane)
lift_off2(airplane)
lift_off(whale)
lift_off2(whale)
: Airplane flying
  AttributeError
  Whale swimming
  AttributeError
```

## Duck typing vs Inheritance


## Polymorphism(다형성)


### Monkey patch

런타임상에서 함수, 메소드, 속성을 바꾸는 패치. <br>
런타임 실행중 메모리상의 오브젝트에 적용된다. <br>

```python
import matplotlib
len(dir(matplotlib))
: 109

mat1 = set(dir(matplotlib))

import matplotlib.pyplot as plt
len(dir(matplotlib))
: 172

mat2 = set(dir(matplotlib))

mat3 = mat2 - mat1
len(mat3)
: 63

mat3
: {'_cm',
   '_cm_listed',
   '_constrained_layout',
   '_contour',
   '_image',
   '_layoutbox',
   '_mathtext_data',
   '_path',
    ......
   'texmanager',
   'text',
   'textpath',
   'ticker',
   'tight_bbox',
   'tight_layout',
   'transforms',
   'tri',
   'units',
   'widgets'}  
```

### Runtime

어떤 프로그램이 실행되는 동안의 시간 <br>
그래서 런타임 에러는 '어떤 프로그램이 실행되는 동안 발생하는 에러를 말한다


## Meta class

<span style="color: bluesky; font-size: 30px;">Type</span><br>
**=> Class 행동을 지정할 수 있다** <br>
**ex) 인스턴스를 한개만 만들 수 있게 지정 (싱글톤)**
<br>

### Type의 3가지 활용

```
1. 메타클래스 자체로 사용할 때
2. 객체의 클래스 이름을 알아낼 때(함수 type())
3. 메타클래스, 클래스를 만들어줄 때
```

<br>

### 예제로 알아보기
```python
class MyType(type): # type을 상속받아 메타클래스를 만듦
    pass

class MySpecialClass(metaclass=MyType): # 안적어주면 type 상속
    pass

msp = MySpecialClass()
print(type(msp))
print(type(MySpecialClass))
print(type(MyType))

: <class '__main__.MySpecialClass'>   # 인스턴스 msp의 클래스명은 MySpecialClass
  <class '__main__.MyType'>           # 클래스 MySpecialClass의 메타클래스는 MyType
  <class 'type'>                      # 메타클래스 MyType의 메타클래스는 type

# 따라서 메타클래스를 만들기 위해서 type을 상속받아야 한다


# lambda 함수식 처럼 클래스도 선언없이 사용할 수 있다

A = type('Integer', (int,), {})  # int클래스를 상속받아 A라는 클래스를 생성

a = A(3)
a
type(a)
: 3
  __main__.Integer

B = type('List', (list,), {})
b = B([1,2,3])
b
type(b)
: [1,2,3]
  __main__.List

C = type('multi', (A, B), {})
: TypeError
# multiple bases have instance lay-out conflict
# => 상속받으려는 두 클래스의 속성이 비슷하여 충돌이 일어나는 경우 다중 상속이 불가하다.
# => 물론 해결하는 방법은 있는 듯 하다
```

<br>

## Singleton

> 인스턴스를 하나만 만들 수 있는 클래스 <br>
> 설정파일을 만드는 객체, 임시저장소 활용할때 쓴다?? <br>

```
1. Type을 상속받는다
2. __call__ (클래스 호출할때 사용함)
```

```python
class Singleton(type):
    instance = None
    def __call__(cls, *args, **kw):
        if not cls.instance:
             cls.instance = super(Singleton, cls).__call__(*args, **kw)      
	                    # super().__call__(*args, **kw) 동일
        return cls.instance

class ASingleton(metaclass=Singleton):
    pass

a = ASingleton()
b = ASingleton()
a is b
: True

# 싱글톤 인스턴스 a를 만들고 다시 b를 만들었더니
```
<br>

**isinstance & issubclass** isinstance는 어떤 객체가 특정 클래스인치 판별하는 predicate issubclass는 어떤 클래스가 특정 클래스의 상속을 받았는지 판별하는 predicate


```python
isinstance(1,(str,int))  # 두번째 인자는 tuple도 가능
issubclass(bool,int)

: True
  True
```

## __getattribute__ vs getattr vs __getattr__

instance.attribute(method) <br>
1. __getattribute__ 실행 (getattr)
2. attribute가 없으면 attribute error
3. __getattr__가 정의 되어 있으면 실행
<br>
instance.attribute(variable) <br>


참고 : [tistory](https://brownbears.tistory.com/187)
## as

1. import할때 명명법 바꾸기
2. 예외처리문에서 에러에 대한 상세표시(설명할 수 있는 다른 객체로 변화 시켜줌)  
3. with 구문 사용할때 파일 내용 할당하기

```python
with open('test.txt', 'w',encoding='utf-8') as f:
    f.write('ㅎㅎㅎㅎㅎ')

with open('test.txt', 'r', encoding='utf-8') as f:
    print(f.read())

: ㅎㅎㅎㅎㅎ
# __enter__, __exit__가 정의 되어 있으면 with 사용가능하다
```

**__all__** import할때 포함시키고 싶은 범위를 지정해주는 special method


special method : [slideshare](https://www.slideshare.net/dahlmoon/specialmethod-20160403-70272494), &nbsp; [git blog](https://corikachu.github.io/articles/python/python-magic-method)<br>

## _ 7가지 활용

```
1. _*
- from module import *에 의해 포함되지 않는 변수명명법

2. __*__
- magic or special method 명명법

3. __*
- mangling
- 클래스내에 __를 사용하여 변수명을 바꿔주는 방법
- 이때 외부에서 해당 변수에 접근을 하지 못하는 private 기능을 하는 것처럼 눈속임을 한다

4. _
- 숫자에 쓰는 언더바
- ex) a = 100_000

5. _ (이름이 중요하지 않지만 관례상 쓸때)

for i,_,k in zip([1,2,3],[4,5,6],[7,8,9]):
    print(i,_,k)

for i,_,_ in zip([1,2,3],[4,5,6],[7,8,9]):
    print(i,_,_)   
# 주의, 맨 마지막에 쓴 값 출력 (할당하지 않으면)

6. _method
- private

7. _  맨 마지막에 쓴 값 출력 (할당하지 않으면)
a = 3
a
-
: 3
  3
(8). _() # 다른 언어지원할 때
- 라이브러리 사용해야 해서 기본 7가지로 생각
```


##### 추가 복습

1. 다형성
2. 추상클래스
3. getattribute

`복습 시간`  20시 ~ 22시 30분/ 총 2시간 30분



<a id = '11th'></a>
# 2019년 5월 17일 금요일 11th

## Multiple inheritance(다중 상속)

말 그대로 상속을 2개 이상을 하는 것


**function 기법**
- 실행 순서를 직접 정할 수 있다
- 그러나 같은 값을 중복해서 출력하는 경우가 생긴다

```python
class x:
    def __init__(self):
        print('x')
class A(x):
    def __init__(self):
        x.__init__(self)
        print('A')
class B(x):
    def __init__(self):
        x.__init__(self)
        print('B')
class C(A,B):
    def __init__(self):
        B.__init__(self)   
        A.__init__(self)  
        print('C')

c = C()

: X
  B
  X
  A
  C
```

**super**

- super는 상속을 전부 실행하지 않는다
- 따라서 중복의 문제를 해결할 수 있다.
- 하지만 super와 function을 함께 사용하면 상속을 전부 실행하지 않는다

```python
class x:
    def __init__(self):
        print('x')
class A(x):
    def __init__(self):
        x.__init__(self)
        print('A')
class B(x):
    def __init__(self):
        x.__init__(self)
        print('B')
class C(B,A):
    def __init__(self):
        super().__init__()  # super는 부모 인스턴스를 반환 / 클래스 명,self 생략 가능
        print('C')

c = C()
: X
  B
  C
```

**only super**

- super는 상속 실행 순서를 자동적으로 지정해준다
- super를 사용할때 전부 super를 사용하면 중복 해결, 원하는 값 출력 가능하다
- 다이아 문제 해결

<br>

```python
class x:
    def __init__(self):
        print('x')
class B(x):
    def __init__(self):
        super().__init__()
        print('B')
class A(x):
    def __init__(self):
        super().__init__()
        print('A')
class C(A,B):
    def __init__(self):
        super().__init__()
        print('C')
c = C()
: x
  B
  A
  C
```

> 왜 실행 순서에 따라 출력되지 않을까?


### MRO (Method Resolution Order)

> 메소드 실행 순서를 확인하는 클래스 메소드(인스턴스로 사용 불가)

```python
C.__mro__
C.mro()

: (__main__.C, __main__.A, __main__.B, __main__.x, object)
  [__main__.C, __main__.A, __main__.B, __main__.x, object]
  # 반환 타입이 서로 다름
```

**실행 순서는 C -> A -> B -> x 인데 출력 결과는 왜 반대일까?**

<span style="background-color: orange">그 이유는 바로 super사용시 stack에 들어가기 때문이다 </span><br>

<br>

## 다중 상속시 절대 에러가 나지 않게 하는 방법

<span style="background-color: rgb(229, 68, 91)">Mixins</span>
**특수한 class를 만들어 충돌이 일어나지 않게 다중 상속을 한다**

```python
class FirstMixin(object):
    def test1(self):
        print("first mixin!!!")


class SecondMixin(object):
    def test2(self):
        print("second mixin!!!")


class TestClass(FirstMixin, SecondMixin):
    pass

t = TestClass()
t.test1
t.test2
: first mixin!!!
  second mixin!!!
```

**다이아 문제** 다중 상속시 어느 클래스의 메소드를 상속 받아야 할 지 모호한 경우


<br>

## ABC class (Abstract base class)

<span style="color:orange">추상적인 부분은 구현하지 않고 구체적인 부분에서 구현하도록 강제하는 기법 </span><br>

**추상 클래스 특징**

```
1. ABC class를 사용하면 duck typing 문제점을 보완할 수 있다
2. 상속받는 클래스는 추상메소드를 구현하지 않아도 클래스 기능은 동작한다
3. abc 모듈을 import 해야한다
```

추상클래스: [wikidocs](https://wikidocs.net/16075)<br>

```python
# sequence 타입의 조건 (sequence type은 iterable을 상속받아 만들어졌다)
class A:           
    x = 1
    def __getitem__(self,x):
        print('A')
    def __len__(self):
        print('B')
a = A()
a[0]
: A
```

### ABCMeta

```python
from abc import ABCMeta, abstractmethod

class Y(metaclass = ABCMeta):
	@abstractmethod
	def a(self):
		return 1
class A(Y):
	pass

a = A()

: TypeError
```

> 오버라이딩을 하지 않는 경우 에러가 발생하기 때문에 오버라이딩을 강제시킨다

```python
class A(Y):
    def a(self):
        return 1

a = A()
a.a()
:1
```

```python
from abc import ABCMeta, abstractclassmethod

class Y(metaclass = ABCMeta):
    @abstractclassmethod
    def a(self):
        return 1
class B(Y):
    def a(self):
        return 1

b = B()
b.a()
: 1

```

#### abstractmethod vs abstractclassmethod


<kbd>duck typing</kbd>+<kbd>meta class</kbd>+<kbd>abc</kbd> => <kbd>강려크하다</kbd>


### register

```python
from abc import ABCMeta

class MyABC(metaclass=ABCMeta):
    pass

MyABC.register(tuple)  # tuple처럼 사용

assert issubclass(tuple, MyABC)
assert isinstance((), MyABC)
```

> 좋지 않은 방법이기 때문에 내가 만든 클래스에서만 사용하도록 권장


<br>

## Descriptor

> 점(.)으로 객체의 멤버를 접근할 때, 먼저 인스턴스 변수(__dict__)에서 멤버를 찾는다. 없을 경우 클래스 변수에서 찾는다. 클래스에서 멤버를 찾고 객체가 descriptor 프로토콜을 구현했다면 바로 멤버를 리턴하지 않고 descriptor 메소드(__get__, __set__, __delete__)를 호출한다

**Descriptor 구현 방법 3가지**
```
1. get, set + composition

2. Properties

3. Decorator
```

#### 1. get, set + composition
```python
class RevealAccess:
    def __init__(self, initval = None, name='var'):
        self.val = initval
        self.name = name
    def __get__(self, obj, objtype):
        print('get')
        return self.val
    def __set__(self, obj, val):
        print('Updating',self.name)
        self.val = val + 10
    def __delete__(self, obj, val):
        print('안지워짐')
class Myclass:
    x = RevealAccess()
    y = 5

m = Myclass()
m.x
: get
m.x = 20
: Updating var
m.x
: get
  30
```
#### 2. Properties

```python
class C(object):  #getx, setx, delx 이름 상관없음
    def getx(self):
        print('AAA')
        return self.__x
    def setx(self, value):
        self.__x = value
    def delx(self):
        del self.__x
    x = property(getx, setx, delx, "I'm the 'x' property.")

d = C()
d.x
: AAA
  AttributeError
```

#### 3. Decorator

```python
class D:
    __X = 3   # 실제값은 __X에 저장
    @property
    def x(self):
        return self.__X

d = D()
d.x()
:TypeError


class D:
    def __init__(self):
        self._x = 0
    @property
    def x(self):
        return self.__x
    @x.setter    #이름은 똑같지만 다른 메모리번지에 할당해주는 역할
    def x(self, x):
        self.__x = x
```
<span style="background-color:red">Descriptor 부분은 다시 공부하고 정리하기 </span><br>

Descriptr : [slideshare](https://www.slideshare.net/dahlmoon/descriptor-20160403)<br>


### 싱글 디스패치, 멀티 디스패치

### epiphany

> 우연한 순간에 귀중한 것들과의 만남, 혹은 깨달음을 뜻하는 통찰이나 직관, 영감을 뜻하는 단어이다.
  python 공부도 그렇듯 계속해서 하다보면 저절로 내것이 될꺼라 믿는다.

epiphany : [brunch](https://brunch.co.kr/@altna84/216)<br>



`복습 시간` 22시 30분 ~ 1시 10분 / 총 2시간 40분  



<a id = 'debug'></a>
# 디버깅

## debug package, builtin_function

### pdb package
```python
x = 1

def y():
    import pdb; pdb.set_trace()
    x = x + 1
    print(x)

 y()
```
![pdb](https://user-images.githubusercontent.com/33630505/60567725-de860300-9da5-11e9-9f85-27249b1694d3.JPG)

### breakpoint

```python
x = 1

def y():
    breakpoint()
    x = x+1
    print(x)
```

![pdb](https://user-images.githubusercontent.com/33630505/60567725-de860300-9da5-11e9-9f85-27249b1694d3.JPG)
