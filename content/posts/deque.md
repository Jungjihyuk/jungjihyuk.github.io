---
title: Deque (Data Structure)
date: 2022-02-09
draft: false
description: About Deque
categories:
- Data Structure
tags:
- Deque
- Data Structure
- Python
slug: deque
---

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">Deque</span> <br>

> 덱은 스택과 큐 연산을 모두 지원하는 자료 구조이고, 양쪽에서 삽입 삭제가 가능하다 (Double - Ended Queue)


<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">덱의 연산</span>

![deque](https://user-images.githubusercontent.com/33630505/153198431-5e113229-3c70-421d-90d5-e84bc212f456.gif)

<br>

- addFront : 앞부분에 데이터를 삽입 
- addRear : 뒷부분에 데이터를 삽입 
- deleteFront : 앞부분의 데이터를 삭제 
- deleteRear : 뒷부분의 데이터를 삭제 
- getFront : 앞부분의 데이터를 가져옴 
- getRear : 뒷부분의 데이터를 가져옴 
- isFull : 덱이 가득 찼는지 

<br>


<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">덱의 종류</span>

```python
1. 스크롤(Scroll) : 입력이 한쪽 끝으로만 가능하도록 제한된 덱 
2. 셀프(Shelf) : 출력이 한쪽 끝으로만 가능하도록 제한된 덱 

```

<br>


<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">덱의 장단점</span>

*장점* 

- 데이터의 삽입, 삭제가 빠르다 (시간복잡도 1)
- 크기가 가변적이다 
- 새로운 데이터 삽입시 메모리를 재할당 또는 복사하지 않고 새로운 단위의 메모리 블록을 할당하여 삽입한다 
- 탐색은 인덱스를 통해 가능하기 때문에 시간복잡도 1을 가진다 

<br>

*단점*

- 자료구조 중간에 있는 삽입, 삭제가 비교적 어렵다 
- 스택, 큐에 비해 구현이 어렵다 

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">파이썬으로 구현하기</span>

### Deque

```python
from collections import deque

d = deque()

d.append(4)
d.append(3)
d.append(5)
d.append(1)

d
:deque([4, 3, 5, 1])

d.pop()
:1 

d.popleft()
:4

d
:deque([3, 5])
```



<br>


### Reference 

```python
1. https://cotak.tistory.com/69
2. https://velog.io/@nnnyeong/자료구조-스택-Stack-큐-Queue-덱-Deque
3. https://gusdnd852.tistory.com/242
```