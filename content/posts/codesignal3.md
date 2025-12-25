---
title: Interview Practice(Codesignal)
date: 2022-01-12
draft: false
description: Practice for coding test(python)
categories:
- Language
- Python
tags:
- CodeSignal
- 코딩연습
- python
- algorithm
slug: codesignal3
---


# Index 

```python
Data Structures 
Sorting & Searching 
Dynamic Programming 
Special Topics 
```

<span style="font-size: 35px; color: rgb(75,163,123); padding: 2px;">Data Structures</span> <br>


<span style="font-size: 28px; color: rgb(75,75,75); padding: 2px;">Arrays</span> <br>

> 단순히 데이터 원소들의 리스트 <br>
> 메모리 상에서 연속으로 저장된다 <br>

### 배열의 종류 

```python
정적 배열 
- 정적 배열을 선언하면 연속된, 정해진 크기의 메모리 공간을 할당한다 
- 메모리를 할당할 때 같은 원소 타입, 정해진 원소 개수에 맞춰서 분할한다 
- O(1)에 조회가 가능
- (배열 시작 주소 + 원소 타입에 따른 바이트 크기 * 원소 인덱스)로 해당 배열 원소 값의 주소 계산 및 메모리 접근이 가능하다 
- 원소의 삭제나 삽입 등의 작업은 최악의 경우 O(n)의 시간 복잡도를 가질 수 있다 
- 예를 들어 원소의 가장 첫번째 원소를 삭제한다고 가정하면 나머지 모든 원소를 하나씩 override해야 하기 때문에 n-1개만큼의 작업을 수행해줘야 한다
- 큐 구현에 사용되는 자료형이다 

동적배열 
- 크기를 사전에 지정하지 않고 자동으로 조정할 수 있도록 하는 배열을 말한다 
- python의 list, C++의 std::vector가 동적 배열에 해당한다
- 미리 초기값을 작게 잡아 배열을 생성하고, 데이터가 추가되면서 배열이 꽉 채워지게 되면 큰 사이즈의 배열을 새로운 메모리 공간에 할당하고 기존 데이터를 모두 복사한다
- 시간 복잡도 = O(n)
- 배열의 사이즈 증가는 대부분 2배씩 이루어진다
- The growth pattern is: 0,4,8,16,25,35,46,58,72,88
- 파이썬의 growth factor는 전체적으로는 1.125
- 더블링을 해야할 만큼 공간이 차게 되는 경우, O(n)의 비용이 발생하게 된다
- 분할 상환 분석에 따른 입력 시간은 여전히 O(1)이다

연결 리스트 
- 연결리스트는 배열과 달리 물리 메모리를 연속적으로 사용할 필요가 없다
- 동적으로 새로운 노드를 삽입 및 삭제하는 것이 간편하다
- 데이터를 구조체로 묶어서 포인터로 연결하는 개념으로 구현된다
- 연결리스트는 원소가 메모리 공간에 연속적으로 존재하지 않으므로 특정 인덱스(원소)에 접근하기 위해서는 전체를 순서대로 읽어야 한다 (시간복잡도: O(n))
- 시작 또는 끝 지점에 아이템을 추가하거나 삭제, 추출하는 작업은 O(1)에 가능하다
- 스택 또는 데크 구현에 쓰이는 자료형
```

#### Reference 

> 누구나 자료 구조와 알고리즘, 저자: 제이 웬그로우, 출판사: 길벗

<br>

## FirstDuplicate

> 입력 리스트에서 제일 먼저 중복을 이루는 값 찾아내는 함수.

<br>

### Example

```
a: [2, 1, 3, 5, 3, 2]
Output: 3

a: [1]
Output: -1

a: [3, 3, 3]
Output: 3

a: [1, 1, 2, 2, 1]
[1번]
Output: 1  
or
[2번]
Output: 2  
```

<br>

## [1번] with Python

```python
def firstDuplicate(a):
    temp = 99
    nums = list(set(a))

    if len(a) == len(nums):
        return -1

    for i in a:
        if a.count(i) > 1:
            for index, value in enumerate(a):
                if value == i and a.index(i)!= index:
                    if temp > index:
                        temp = index
    return a[temp]
```

`가장 먼저 중복을 이루면 해당하는 값을 출력한다.`

<br>

## [2번] with Python

```python
def firstDuplicate(a):
    temp = -1
    nums = list(set(a))

    if len(a) == len(nums):
        return -1

    for i in nums:
        if a.count(i) > 1:
            a.reverse()
            if temp < a.index(i):
                temp = a.index(i)
                a.reverse()
            else:
                a.reverse()
    a.reverse()
    return a[temp]
```
`입력값의 전체를 확인 했을 때 가장 마지막 중복값의 인덱스가 빠른 경우의 값을 출력`

<br>

## FirstNotRepeatingCharacter

> 가장 처음으로 오는 중복이 없는 문자

<br>

### Example

```
s = 'aabbcc'
ouput = '_'

s = 'abbc'
output = 'a'

s = 'a'
output = 'a'
```

<br>

## [1번] with Python

```python
def firstNotRepeatingCharacter(s):
    for x in list(s):
        if list(s).count(x) == 1:
            return x
    return '_'
```

`예외 상황없이 잘 되지만 속도가 매우 낮다.`

<br>

## [2번] with Python

```python
from collections import Counter

def firstNotRepeatingCharacter(s):
    for x, y in Counter(s).items():
        if y==1:
            return x

    return "_"
```

`1번 버전보다 속도가 더 빠르다.`

<br>

## [3번] with Python

```python
def firstNotRepeatingCharacter(s):
    for c in s:
        if s.index(c) == s.rindex(c):
            return c
    return '_'
```

`rindex는 가장 오른쪽에 있는 index값을 출력 하는 메소드!`

<br>

## rotateImage

> 2차원 배열을 입력값으로 받으면 오른쪽으로 90도 회전한 2차원 배열(행렬)을 반환 하는 함수

<br>

### Example

```
a = [[1,2,3],
     [4,5,6],
     [7,8,9]]
output = [[7,4,1],
          [8,5,2],
          [9,6,3]]

a = [[10,9,6,3,7],
     [6,10,2,9,7],
     [7,6,3,8,2],
     [8,9,7,9,9],
     [6,8,6,8,2]]
output = [[6,8,7,6,10],
          [8,9,6,10,9],
          [6,7,3,2,6],
          [8,9,8,9,3],
          [2,9,2,7,7]]                   
```

<br>

## [1번 with python]

```python
import numpy as np


def rotateImage(a):
    return list(np.array(a).T[:,::-1])
```

`2차원 행렬을 오른쪽으로 90도 회전 하는 것은 행렬을 전치한 후 뒤집으면된다.`

<br>

## [2번 with python]

```python
rotateImage = lambda a: zip(*a[::-1])
```

`2차원 행렬을 위아래로 뒤집은 후 열을 행으로 바꾸면 오른쪽으로 90도 회전한 것과 같아진다.`


<span style="font-size: 28px; color: rgb(75,75,75); padding: 2px;">Linked Lists</span> <br>


> 연결 리스트는 노드들이 한 줄로 연결되어 있는 방식의 자료 구조이다. <br>
> 여기서 노드란, 데이터와 다음 노드의 주소(포인터)를 담고 있는 구조체(메모리 공간)

<br>

### 연결리스트의 종류 

![linkedList](https://user-images.githubusercontent.com/33630505/149276486-09f0490d-6b5e-43f9-8d78-9b3fbadb7483.png)

<br>

```python
# 연결방향에 따라 연결리스트는 세 가지 종류로 나뉜다 

1. Singly linked list 
- 단방향으로 연결되어 있고 연결되어 있는 각 노드들은 다음 노드의 주소를 포함하고 있다 
2. Doubly linked list 
- 양뱡향으로 연결되어 이고 연결되어 있는 각 노드들은 이전, 다음 노드의 주소를 모두 포함하고 있다 
3. Circular linked list 
- 단방향, 양방향 모두 가능하고 처음과 끝이 연결되어 있는 경우이다 
```

### 배열 VS 연결리스트 

![arraylinkedlist](https://user-images.githubusercontent.com/33630505/149282671-50cf3f9d-53fc-4882-9e61-1573c082cffd.png)

<br>

### Singly linked list with python 

```python
class Node(object):
    '''
    데이터와 다음 노드의 주소(포인터)를 담고 있는 구조체(메모리 공간)
    '''
    def __init__(self, data):
        self.data = data # 데이터 
        self.next = None # 다음 노드의 주소 (포인터)

class SinglyLinkedList(object):
    def __init__(self):
        self.head = None  # 포인터    
    
    def append(self, node):          # 추가 
        if self.head == None:        # 헤드에 아무것도 없으면 
            self.head = node         # node 객체를 헤드로 (추가)
        else:                        # 헤드에 있으면 
            cur = self.head          # 현재 헤드를 포인터로 두고 
            while cur.next != None:  # 현재 헤드가 가리키는 것이 없을 때 까지 헤드를 옮김
                cur = cur.next       
            cur.next = node          # 끝나면 다음 헤드를 노드로 
    
    def getdataIndex(self, data):    # data의 인덱스 가져오기 
        curn = self.head             # 현재 헤드를 포인터로 둔다 
        idx = 0                      # 0부터 
        while curn:                  # 헤드가 있을 때 까지 
            if curn.data == data:    # 지금 헤드의 데이터와 data가 일치하면 
                return idx           # 그때의 index 반환 
            curn = curn.next         # 아니면 헤드 이동 
            idx += 1                 # index 증가 (헤드 이동을 위한)
        return -1                    # 없으면 -1 반환 

    def insertNodeAtIndex(self, idx, node): 
        '''
        지정된 인덱스에 새로운 노드 삽입 
        '''
        curn = self.head 
        prevn = None 
        cur_i = 0 
        
        if idx == 0: 
            if self.head:              # 현재 노드가 있다면 
                nextn = self.head      # nextn 변수에 현재 헤드 값 저장 
                self.head = node       # 현재 헤드에 삽입할 노드 저장 
                self.head.next = nextn # 삽입한 노드 다음 노드에 이전 헤드 값 저장 
            else:                     
                self.head = node       # 현재 노드가 없으면 그냥 추가 
        else:                          # index값이 0이 아닌 모든 경우에 
            while cur_i < idx:         # 삽입할 index이전까지 
                if curn:               # 현재 헤드에 노드가 있으면 
                    prevn = curn       # 현재 헤드를 이전 노드로 
                    curn = curn.next   # 현재 노드의 다음 노드를 현재 노드로 
                else:
                    break    
                cur_i += 1         
            if cur_i == idx:           # 현재 인덱스가 삽입할 인덱스와 같으면 
                node.next = curn       # 현재 헤드를 추가할 노드 다음으로 
                prevn.next = node      # 추가할 노드를 현재 헤드의 이전 노드로 
            else: 
                return -1 
        
    def insertNodeAtData(self, data, node):
        index = self.getdataIndex(data)
        if 0 <= index:
            self.insertNodeAtIndex(index, node)
        else:
            return -1 
        
    def deleteAtIndex(self, idx):
        curn_i = 0 
        curn = self.head 
        preven = None
        nextn = self.head.next 
        if idx == 0:
            self.head = nextn 
        else: 
            while curn_i < idx:
                if curn.next: 
                    preven = curn 
                    curn = nextn
                    nextn = nextn.next 
                else:
                    break 
                curn_i += 1 
            if curn_i == idx:
                preven.next = nextn 
            else:
                return -1 
            
    def clear(self):
        self.head = None  # 현재 헤드를 없앤다 
    
    def Lprint(self):
        curn = self.head 
        string = ""
        while curn:
            string += str(curn.data)
            if curn.next:
                string += "->"
            curn = curn.next 
        print(string )
    
if __name__ == "__main__":
    sl = SinglyLinkedList()
    sl.append(Node(1))
    sl.append(Node(2))
    sl.append(Node(3))
    sl.append(Node(4))
    sl.append(Node(5))
    sl.insertNodeAtIndex(3, Node(4)) 
    print(sl.getdataIndex(1))
    print(sl.getdataIndex(2))
    print(sl.getdataIndex(3))
    print(sl.getdataIndex(4))
    print(sl.getdataIndex(5))
    sl.insertNodeAtData(1, Node(0))
    sl.Lprint()         
```


#### Reference 

> 생활코딩 https://opentutorials.org/module/1335/8821 <br>
> 초보몽키의 개발공부로그 https://wayhome25.github.io/cs/2017/04/17/cs-19/ <br>
> Daim blog https://daimhada.tistory.com/72 <br>
> BaaaaaaaarkingDog blog https://blog.encrypted.gg/932 <br> 
> Dalkomit blog https://dalkomit.tistory.com/7 <br>


## RemoveKFromList

> Linked list에서 K와 일치하는 값 제거하기 

<br>

### Example

```python
l = [3, 1, 2, 3, 4, 5]
k = 3

solution(l, k) = [1, 2, 4, 5]

l = [1, 2, 3, 4, 5, 6, 7]
k = 10

solution(l, k) = [1, 2, 3, 4, 5, 6, 7]
```

## [1번 with python]

```python
# singly-linked list:
class ListNode(object):
  def __init__(self, x):
    self.value = x
    self.next = None

def solution(l, k):
    c = l
    while c:
        if c.next and c.next.value == k:
            c.next = c.next.next
        else:
            c = c.next
    return l.next if l and l.value == k else l
```

## IsListPalindrome

> Linked list가 palindrome인지 아닌지 판별하기 

<br>

### Example

```python
l = [0, 1, 0]

solution(l) = true 

l = [1, 2, 2, 3]

solution(l) = false
```

## [1번 with python]


```python
class ListNode(object):
  def __init__(self, x):
    self.value = x
    self.next = None

def solution(l):
    a = []
    while l != None: 
        a.append(l.value)
        l = l.next
    return a == a[::-1]
```


<span style="font-size: 28px; color: rgb(75,75,75); padding: 2px;">Hash tables</span> <br>

> 해시 테이블은 키(key) 1개, 값(value) 1개 쌍으로 이루어져 있는 자료구조 이다 <br>
> 프로그래밍 언어에 따라 서로 다른 이름으로 불린다 (해시, 맵, 해시 맵, 딕셔너리, 연관 배열) <br>
> 해시 테이블은 O(1)만에 값을 찾아낼 수 있는 빠른 읽기의 장점을 갖고 있다 

<br>

## Grouping dishses

> 음식, 식재료, 식재료,... 순으로 되어 있는 리스트를 식재료, 음식, 음식... 순으로 바꾸는 함수를 만든다 <br>
> 이때, 식재료는 두 가지 이상 음식에 포함되어 있는 것만 값을 반환해야 한다 <br>
> 추가로 사전식 순서를 따라 정렬한다 (lexicographically order) 

<br>

### Example

```python
dishes = [["Salad", "Tomato", "Cucumber", "Salad", "Sauce"],
            ["Pizza", "Tomato", "Sausage", "Sauce", "Dough"],
            ["Quesadilla", "Chicken", "Cheese", "Sauce"],
            ["Sandwich", "Salad", "Bread", "Tomato", "Cheese"]]

solution(dishes) = [["Cheese", "Quesadilla", "Sandwich"],
                            ["Salad", "Salad", "Sandwich"],
                            ["Sauce", "Pizza", "Quesadilla", "Salad"],
                            ["Tomato", "Pizza", "Salad", "Sandwich"]]
```

### [1번 with python]

```python
def solution(dishes):
    groups = {}
    for d, *v in dishes:
        for x in v:
            groups.setdefault(x, []).append(d)
    ans = []
    for x in sorted(groups):
        if len(groups[x]) >= 2:
            ans.append([x] + sorted(groups[x]))
    return ans
```



<span style="font-size: 35px; color: rgb(75,163,123); padding: 2px;">Sorting & Searching </span> <br>

