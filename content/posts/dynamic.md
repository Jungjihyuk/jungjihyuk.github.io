---
title: Dynamic programming (Data Structure)
date: 2022-02-15
draft: false
description: About dynamic programming
categories:
- Data Structure
tags:
- Dynamic programming
- Data Structure
- Python
slug: dynamic
---


<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">Dynamic Programming(동적 계획법)</span> <br>


> 반복되는 연산을 피하기 위해서 테이블을 만들어 기록해두고 재사용하는 프로그래밍 기법 

<br>

<span style="font-size: 23px; color: rgb(194,147,67); padding: 2px;">[동적 프로그래밍의 종류] </span>

```python
1. 메모이제이션(Memoization) 
2. 동전 교환 문제(Coin Change Problem) 
3. 배낭 채우기 문제(Knapsack Problem) 
4. 최장 공통 부분 수열(Longest Common Subsequence) 
5. 최장 증가 부분 수열(Longest Increasing Subsequence) 
6. 편집 거리 알고리즘(Edit Distance)
7. 행렬 체인 곱셈(Matrix Chain Multiplication)
```

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">동적 프로그래밍의 장단점</span>

*장점* 

- 비교적 반복적인 연산이 많이 수행되는 문제에서 연산 속도를 비약적으로 증가시킬 수 있다 
- 필요한 모든 가능성을 고려해서 구현하기 때문에 항상 최적의 결과를 얻을 수 있다 
- 재귀호출로 인한 오버헤드 발생을 막을 수 있다

<br>

*단점*

- 다른 방법론에 비해 많은 메모리 공간이 필요하다 

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">동적 프로그래밍의 특징</span>

```python
1. 주어진 조건에서 최적의 값을 찾는 작업을 위한 프로그래밍 방법이다 
2. 분할 정복(Dvide and conquer) 알고리즘의 비효율성(재귀호출)을 개선한 방법 
  - 분할정복 기법과 동일하게 부분 문제의 해를 결합하여 문제를 해결한다
  - 분할정복과 다른점은 부분 문제들이 독립적이지 않다
1. 부분 문제를 Bottom-Up 방식으로 해결한다 
2. 부분 문제를 처음 해결하면 값을 저장한다 
3. 동일한 부분 문제가 발생하면 저장된 값을 불러온다
```

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">동적 프로그래밍을 사용하기 위한 선행조건</span>

```python
1. 큰 문제를 작은 문제로 나눌 수 있다 (간단한 부분 문제) 
2. 작은 문제에서 구한 정답은 그것을 포함하는 큰 문제에서도 동일하게 적용된다 (문제의 최적 부분구조)
3.  동일한 부분 문제를 풀어야 하는 시점이 존재한다 (중복된 부분 문제)
```

<br>

<span style="font-size: 23px; color: rgb(194,147,67); padding: 2px;">[동적 프로그래밍 구현하기] </span>


**피보나치 수열**

![fibo](https://user-images.githubusercontent.com/33630505/154040120-c94d0a49-f0b8-45e5-ab9d-9051528289e6.png)

위 와 같은 점화식을 만족하는 수열을 피보나치 수열이라고 한다 
피보나치 수열을 python의 재귀함수로 구현하면 다음과 같다

```python
import time

def fibo(x):
    if x == 1 or x == 2: 
        return 1 
    return fibo(x-1) + fibo(x-2)

for num in range(5, 50, 10):
    start = time.time()
    res = fibo(num)
    print(res, '-> 러닝타임:', round(time.time() - start, 2), '초')

5 -> 러닝타임: 0.0 초
610 -> 러닝타임: 0.0 초
75025 -> 러닝타임: 0.02 초
9227465 -> 러닝타임: 2.35 초
1134903170 -> 러닝타임: 244.81 초
```

> 35번째 부터 러닝타임이 기하 급수적으로 늘어난다 

<br>

이렇게 재귀함수의 긴 러닝타임의 단점을 극복하고 단계가 증가하더라도 러닝타임이 기하 급수적으로 증가하지 않는 방법이 있다 

<br>

그것은 바로 한번 실행한 결과를 메모리에 저장하고 다음에 또 호출되면 연산하지 않고 값을 불러와서 사용하는 방법이다

이러한 것을 **메모제이션 기법**이라고 한다 

<br>

```python
import time

d = [0] * 50

def fibo(x):
    if x == 1 or x == 2:
        return 1
    if d[x] != 0:
        return d[x]
    d[x] = fibo(x-1) + fibo(x-2)
    return d[x]

for num in range(5, 50, 10):
    start = time.time()
    res = fibo(num)
    print(res, '-> 러닝타임:', round(time.time() - start, 2), '초')

5 -> 러닝타임: 0.0 초
610 -> 러닝타임: 0.0 초
75025 -> 러닝타임: 0.0 초
9227465 -> 러닝타임: 0.0 초
1134903170 -> 러닝타임: 0.0 초
```

> 재귀함수 방법과 달리 단계가 늘어나도 속도가 느려지거나 하지 않았다 

<br>

<span style="font-size: 23px; color: rgb(194,147,67); padding: 2px;">[Memoization] </span>

> 어떠한 프로그램이 동일한 계산을 반복해야 할 때, 이전에 > 계산한 값을 메모리에 저장함으로써 동일한 계산의 반복 수행을 제거하여 프로그램 실행 속도를 빠르게 하는 기술이다 


![memoization](https://user-images.githubusercontent.com/33630505/154040552-44c0cfea-801d-4f0c-8356-040ffea08886.png)


이렇게 Memoization 방법은 큰 문제를 작은 문제로 나누어 위에서 부터 아래로 즉, **Top-Down** 방식으로 문제를 해결해 나간다. 한 번 연산된 결과는 메모리에 기록되어 있기 때문에 재호출되는 경우 기록된 메모리에서 불러오기만 하면 된다

<br>


```python
class Fibber(object):
    
    def __init__(self):
        self.memo = {}
        
    def fib(self, n):
        if n < 0:
            raise IndexError(
                'Index was negative.'
                'No such thing as a negative index in a series.'
            )
        
        # Base cases    
        elif n in [0, 1]:
            return n
        
        # 이미 계산한 값인지 아닌지 확인한다. 
        elif n in self.memo:
            print ("grabbling memo[%i]" % n)
            return self.memo[n]
        
        print("computing fib(%i)" % n)
        result = self.fib(n - 1) + self.fib(n - 2)
    	
        # Memoize
        self.memo[n] = result
        
        return result
```



<br>

<span style="font-size: 23px; color: rgb(194,147,67); padding: 2px;">[Bottom - Up] </span>

> Top - Down 방식과 달리 for문을 이용해서 처음값부터 다음 값을 계산해 나가는 방식이다 

<br>

```python
d = [0] * 100

d[1] = 1 # 첫 번째 항
d[2] = 1 # 두 번째 항
N = 99   # 피보나치 수열의 99번째 숫자는?

for i in range(3, N+1):
    d[i] = d[i-1] + d[i-2]

print(d[N])
```