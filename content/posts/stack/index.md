---
title: Stack (Data Structure)
date: 2022-02-04
draft: false
description: About stack
categories:
- Data Structure
tags:
- Stack
- Data Structure
- Python
slug: stack
image: stack.jpg
---


<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">Stack</span> <br>


> 스택은 한쪽 방향으로만 데이터를 저장하고 추출하는 형식의 자료 구조이다. 가장 최근에 저장된 값이 가장 먼저 추출되는 방식을 사용한다 (LIFO[Last In First Out] or FILO[First In Last Out]) 

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">스택의 연산</span>

![stack](https://user-images.githubusercontent.com/33630505/152550347-cc5c2925-125a-4ffe-8383-4d55673d0156.gif)


<br>

- pop : 스택에서 가장 마지막에 있는 데이터를 제거한다 (삭제) 
- push : 데이터를 가장 마지막 부분에 추가한다 (삽입) 
- peek : 가장 마지막 부분에 있는 데이터를 반환한다 (읽기) 
- isEmpty : 스택이 비어 있을 때 true를 반환한다 


<br>

### Stack overflow, underflow

```
Stack overflow는 스택의 최대길이까지 데이터가 차있을 때 스택에 데이터를 추가하려고 할 때 발생하는 에러 
Stack underflow는 스택에 아무것도 없을 때 데이터를 불러오려고 시도하면 발생하는 에러 
```

### 예시 

```python
class Stack(object):
    def __init__(self, limit = 5):
        self.stack = []
        self.limit = limit

    def __str__(self):
        return ' '.join([str(i) for i in self.stack])

    def push(self, data):
        # 스택의 최대길이에 도달하면 stack overflow 출력 
        if len(self.stack) >= self.limit:
            print('Stack overflow')
        else:
            self.stack.append(data)
        
    def pop(self):
        # 스택이 비어있으면 stack stack underflow 출력 
        if len(self.stack) <= 0:
            print('Stack underflow')
        else:
            return self.stack.pop()
        
    def peek(self):
        if len(self.stack) <= 0:
            print('Stack underflow')
        else:
            return self.stack[len(self.stack) - 1]
```

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">스택의 장단점</span>


*장점* 

- 구조가 단순해서 구현하기 쉽다 
- 데이터의 저장, 읽기 속도가 빠르다 

<br>

*단점* 

- 데이터 최대 갯수를 정해야 한다 (파이썬의 경우 재귀함수 호출은 1000번까지 가능)
- 저장공간이 비효율적이다 (최대 갯수를 정해야 하기 때문에)
- 데이터의 삽입, 삭제가 비효율적이다


<br>


<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">스택의 활용</span>

- 재귀 알고리즘
- DFS 알고리즘
- 작업 실행 취소와 같은 역추적 작업이 필요할 때
- 괄호 검사, 후위 연산법, 문자열 역순 출력 등

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">파이썬으로 구현하기</span>


### Factorial과 stack 

```python
def facto(n):
    # stack 할당 
    stack = []
    
    # n부터 1까지 수행 
    while n > 0:
        # n부터 1까지 stack에 추가 
        stack.append(n)
        n -= 1 
    result = 1 
    
    # stack에 있는 것을 1부터 뽑아서 n까지 누적 곱 수행 (가장 마지막에 넣은 값부터 뽑기 때문에 stack처럼 작동한다)
    while stack: 
        result *= stack.pop()
    return result

facto(5)
: 120 # 1*2*3*4*5 


📌 파이썬 재귀의 한계 

def recur(n):
    print(n)
    if n == 2968:
        return "limit"
    if n == 2969:
        return "exceed"
    return recur(n+1) 

recur(1) 
: 1
  2
  3 
  ...
  2964
  2965
  2966
  2967
  2968
  'limit'

# 파이썬의 재귀호출은 2968이 최대이다 

# 단, 재귀의 한계를 늘릴 수는 있다 

import sys 
sys.setrecursionlimit(4000)

recur(1) 
: 1
  2
  3 
  ...
  3964
  3965
  3966
  3967
  3968
  RecursionError 
```

<br>

### Array와 Stack 

```python
class Stack: 
    def __init__(self): 
        # 데이터 저장을 위한 리스트 준비 
        self.data = [] 
		
		def push(self, val): 
        # 가장 마지막에 값 추가 
        self.data.append(val) 
		
    def pop(self): 
        try: 
		        # 데이터가 있으면 가장 마지막 값 읽기 
		        return self.data.pop() 
		    except IndexError: 
		        # 데이터가 없으면 인덱스에러 발생
            print("Stack is empty") 
		
    def __len__(self): 
		    # len()로 호출하면 stack의 data 수 반환 
		    return len(self.data) 
		
    def isEmpty(self): 
        # 스택이 현재 비어 있는지 확인 
		    return self.__len__() == 0
```

<hr>

### Reference 

```python
1. https://cotak.tistory.com/69
2. https://ratsgo.github.io/data%20structure&algorithm/2017/10/11/stack/
3. https://data-marketing-bk.tistory.com/15
4. https://medium.com/@jch9537/stack-queue-linked-list-63ff4d54ec7d
```