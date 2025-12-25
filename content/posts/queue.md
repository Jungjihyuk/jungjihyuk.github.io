---
title: Queue (Data Structure)
date: 2022-02-05
draft: false
description: About Queue
categories:
- Data Structure
tags:
- Queue
- Data Structure
- Python
slug: queue
---


<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">Queue</span> <br>

> 큐는 스택과 동일하게 데이터를 저장할 때는 가장 마지막에 저장된 데이터 다음에 새로운 값을 저장한다 
하지만 스택과 다른점은 데이터를 읽어올 때 가장 먼저 저장된 데이터부터 값을 불러온다 (FIFO[First In First Out]) 

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">큐의 연산</span>

![queue](https://user-images.githubusercontent.com/33630505/152647105-872bb5a8-8d8d-4af7-8eb1-063357f597df.gif)

<br>

- Enqueue : Rear 부분에 먼저 들어오는 데이터부터 차례대로 삽입한다 (삽입)
- Dequeue : Front 부분에 있는 데이터부터 삭제한다 (삭제)
- Front : 가장 앞에 있는 데이터
- Rear : 가장 뒤에 있는 데이터
- isEmpty : 큐가 비어 있을 때 True를 반환한다
- isFull : 큐가 꽉 차있으면 True를 반환한다
- Peek : 큐의 맨 앞에 저장된 데이터를 반환한다 (읽기)


<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">큐의 종류</span>

```python
1. 선형 큐(Linear Queue) 
    a. 선형 큐는 말 그대로 선형 자료구조 형태를 하는 큐이다 
    b. 가장 앞 부분을 front, 가장 뒤 부분을 rear로 지칭한다 
    c. 데이터 삽입시 rear가 가리키는 위치에 추가하고 rear를 1만큼 증가시킨다 
    d. 데이터 삭제시 front에 위치하는 데이터를 삭제하고 front를 1만큼 증가시킨다 
2. 원형 큐(Circular Queue)
    a. 원형 큐는 원 형태의 큐인 자료구조이다 
    b. 선형 큐와 동일하게 앞부분은 front, 뒤 부분을 rear로 지칭한다 
    c. 데이터가 비어 있을 때는 front와 rear위치가 동일하다 
    d. rear를 1만큼 증가시켰을 때 front와 동일하다면 데이터가 꽉 차있는 상태이다  (front%max_size == (rear+1)%max_size)
    e. 데이터 삽입 방법은 선형 큐와 동일하다 
    f. 데이터 삭제 방법은 선형 큐와 동일하다 
3. 연결 리스트 큐(Linked list Queue) 
    a. 연결 리스트를 활용한 큐 
    b. 큐의 길이를 유동적으로 조절할 수 있다 (오버 플로우 X) 
    c. 선형, 원형 둘다 구현 가능하다 
    d. 데이터를 저장하는 노드(node)와 다음의 노드를 가리키는 링크(link)로 구성된다 
    e. 가장 먼저 있는 데이터를 가리키는 리스트의 머리(head)를 만들어야 한다 
    f. 마지막 노드의 링크가 가리키는 꼬리(tail)을 만들어야 한다
4. 우선순위 큐 (Priority Queue) 
    a. 모든 데이터에 우선순위가 있다 
    b. 우선순위가 높은 데이터는 우선순위가 낮은 데이터 보다 먼저 큐에서 삭제된다 
    c. 두 개 이상의 데이터의 우선순위가 같으념 큐에 삽입 되어 있는 순서에 따라 삭제된다 
    d. 데이터를 삭제할 때 루트노드의 값을 추출하고 힙의 맨 끝의 데이터를 루트노드에 배치한다. 그 다음 맨 끝의 데이터를 삭제한다. 힙 속성에 따라 값의 위치를 변경해준다 (힙 속성이 유지될때까지 계속해서 값의 위치를 변경한다) 
    e. 데이터를 삽입할 때 힙끝에 삽입하고 부모노드와 비교하여 힙 속성에 따라 값의 위치를 변경해준다 (힙 속성이 유지될때까지 계속해서 값의 위치를 변경한다)
    f. 삽입과 삭제 모두 트리의 높이 만큼 타고 올라가기 때문에 시간복잡도는 log2n이 된다 
```


<br>


<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">1) 선형 큐의 장단점</span>

*장점* 

- 구현이 간단하다 

<br>

*단점*

- 삭제 연산을 반복하게 되면 front의 위치가 rear와 가까워지기 때문에 기존의 데이터 공간을 사용할 수 없는 문제가 발생한다 (실제로는 데이터 공간이 남아있다)
- 데이터 사용공간을 확보하기 위해서 삭제 연산시 원소를 삭제한 만큼 shift연산을 해야 하기 때문에 속도가 느리고 비효율적이다


<br>


<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">2) 원형 큐의 장단점</span>

*장점* 

- 메모리 공간 활용이 용이하다 (선형 큐의 단점을 해결)

<br>

*단점*

- 큐의 크기가 제한된다 

<br>

### 원형큐의 상태 

![queue_state](https://user-images.githubusercontent.com/33630505/152928278-2b756d31-4bc6-4adc-b68f-d36845b435cd.png)

<br>

### 원형큐의 삽입과 삭제 

![queue](https://user-images.githubusercontent.com/33630505/152928201-db1227f5-ec23-430f-af21-87597be20c8c.png)

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">3) 연결리스트 큐의 장단점</span>

*장점* 

- 큐의 크기 제한이 없다
- 삽입, 삭제가 편리하다

<br>

*단점*

- 데이터 탐색이 오래 걸린다 
- 구현이 비교적 어렵다 

<br>

### 연결리스트 큐의 삽입과 삭제 

![linked_enqueue](https://user-images.githubusercontent.com/33630505/152928462-8ec5256f-4087-47d2-9402-93295f8a8751.png)

![linked_dequeue](https://user-images.githubusercontent.com/33630505/152928482-186c94ea-8127-418e-b301-dcc9a5460a61.png)

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">4) 우선순위 큐의 장단점</span>

*장점* 

- 보통 힙으로 구현하기 때문에 힙의 장점인 데이터 삽입시 속도의 편차가 크지 않다는 장점이 있다 (log2n)
- 최대값 또는 최소값을 찾는데 최적화 되어 있다

<br>

*단점*

- 배열이나 연결 리스트에 비해 데이터 삭제 속도가 느리다는 단점이 있다 (log2n) 

<br>


<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">파이썬으로 구현하기</span>


### Queue

```python
class Queue: 
    def __init__(self):
        self.items = [] 

    def enqueue(self,val):
        self.items.append(val)
    
    def dequeue(self): 
        try: 
            return self.items.pop(0)
        except IndexError:
            print("Queue is empty") 
    
    def front(self): 
        try: 
            return self.items[0] 
        except IndexError: 
            print("Queue is empty") 
    
    def __len__(self): 
        return len(self.items) 
    
    def isEmpty(self): 
        return len(self)
```

<br>

### Circular Queue 

```python
size = 10 # 원형 큐의 크기 지정 

class CircularQueue: 
    def __init__(self):
        # 큐의 앞, 뒤, 데이터 공간을 지정한다 
        self.front = 0 
        self.rear = 0 
        self.datas = [None] * size 

    def enqueue(self, data):
        # Enqueue 작업 수행 (삽입)
        # 삽입 가능한 상태인지 체크후 
        # 데이터를 뒤에서부터 삽입한다. 이때 rear의 인덱스를 증가시키고 해당 인덱스에 삽입한다 
        # 만약 rear가 최대인덱스를 넘어가면 최대 크기를 나눈 나머지 인덱스를 지정한다 
        if not self.isFull():
            self.rear = (self.rear + 1) % size 
            self.datas[self.rear] = data

    def dequeue(self):
        # dequeue 작업 수행 (삭제)
        # 데이터가 비어있는지 체크후 
        # 데이터가 있다면 front를 옮김으로써 큐 안에 있는 데이터를 지운다  
        if not self.isEmpty():
            self.front = (self.front + 1) % size 
            self.datas[self.front]
        return self.datas[self.front]
    
    def peek(self):
        # 데이터가 비어있지 않으면 front 다음 인덱스 안에 있는 값을 반환한다 
        if not self.isEmpty():
            return self.datas[(self.front + 1) % size]

    def isEmpty(self):
        # front와 rear가 동일하다면 큐가 비어 있음으로 True를 반환한다 
        return self.front == self.rear 

    def isFull(self):
        # rear가 가리키는 인덱스의 다음 인덱스가 front의 인덱스와 동일하다면 가득참을 의미함으로
        # Truefmf qksghksgksek 
        return self.front%size == (self.rear+1)%size
 
    def clear(self):
        # 초기화 
        self.front = self.rear

cq = CircularQueue()

cq.enqueue(5)
cq.enqueue(2)
cq.enqueue(3)
cq.enqueue(1)
cq.enqueue(5)
cq.enqueue(2)
cq.enqueue(3)
cq.enqueue(1)
cq.enqueue(9)
cq.enqueue(7)

cq.isEmpty()
:False 
cq.isFull()
:True
cq.dequeue()
:5
cq.dequeue()
:2
cq.peek()
:3
```

<br>

### Linked list Queue 

```python
class Node:
    def __init__(self, data):
        # 노드에는 데이터와 그 다음 노드의 주소로 구성되어 있다 
        # Singly linked list 
        self.data = data 
        self.next = None

class LinkedListQueue:
    def __init__(self):
        # front, rear를 설정한다
        self.front = None
        self.rear = None

    def isEmpty(self):
        # front가 None이면 큐가 비어있기 때문에 None을 반환한다  
        if self.front is None:
            return True
        else:
            return False

    def enQueue(self, data):
        # 데이터를 새로 삽입 
        # 삽입할 노드를 선언
        new_node = Node(data)
        
			  # 비어 있으면 front와 rear에 새로운 노드를 추가한다 
        if self.isEmpty():
            self.front = new_node
            self.rear = new_node
        # 비어 있지않으면 기존 rear와 rear뒤에 새로운 노드를 추가한다 
        else: 
            self.rear.next = new_node
            self.rear = new_node
        

    def deQueue(self):
        # 비어 있으면 비어있다고 출력
        if self.isEmpty():
            return "Queue is Empty"
        # 비어 있지않으면 front를 dequeued에 담고 현재 front 뒤에 있는 것을 front로 지정
        else: 
            dequeued = self.front 
            self.front = self.front.next 
        # front가 None이면 큐가 비었기 때문에 rear도 None으로 설정
        if self.front is None: 
            self.rear = None 
            
        return dequeued.data

    def peek(self):
        if self.isEmpty():
            return "Queue is Empty"
        return self.front.data
    
    def __str__(self):
        print_queue = ''
        node = self.front
        while True:
            if(node == self.rear):
                print_queue += '<= [ '
                print_queue += str(node.data)
                print_queue += ' ] <='
                break
            try:
                print_queue += '<= [ '
                print_queue += str(node.data)
                print_queue += ' ] '
                node = node.next
            except:
                break

        return print_queue

lq = LinkedQueue()
lq.enQueue(4)
lq.enQueue(3)
lq.enQueue(2)
lq.enQueue(1)
print(lq)
:<= [ 4 ] <= [ 3 ] <= [ 2 ] <= [ 1 ] <=
lq.isEmpty()
:True 
lq.deQueue()
:4
print(lq)
:<= [ 3 ] <= [ 2 ] <= [ 1 ] <=
```

<br>


### Priority Queue 

```python
import heapq

# 힙이름, 추가할 데이터 
heap = []
heapq.heappush(heap,3)
heapq.heappush(heap,5)
heapq.heappush(heap,2)
heapq.heappush(heap,1)

[heapq.heappop(heap) for x in range(len(heap))]
: [1,2,3,5]
```

<br>

<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">큐의 활용</span>

```
- 프로세스 스케줄링
- 입출력 (파일 입출력)
- 프린터 대기열
- 네트워크 패킷 처리
- 게임 대기열
```


<hr>

### Reference 

```python
1. https://suyeon96.tistory.com/31
2. https://yoongrammer.tistory.com/81
3. https://cotak.tistory.com/69
4. https://kangworld.tistory.com/60
5. https://velog.io/@changyeonyoo/자료구조python-원형큐-덱-CircularQueue-CircularDeque-구현
```