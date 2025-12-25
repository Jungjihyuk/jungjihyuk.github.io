---
title: DFS/BFS (탐색 알고리즘)
date: 2022-02-09
draft: false
description: About search algorithm
categories:
- Data Structure
tags:
- BFS
- DFS
- Data Structure
- Python
slug: search_algorithm
---

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">DFS/BFS(탐색 알고리즘)</span> <br>

> DFS(Depth-First-Search)는 가장 깊은 부분을 우선적으로 탐색하는 알고리즘이고 BFS(Breadth-First-Search)는 가장 가까운 노드부터 탐색하는 알고리즘이다. 


<br>

<span style="font-size: 23px; color: rgb(194,147,67); padding: 2px;">[DFS]</span>

```python
‘가장 깊은 부분을 우선적으로 탐색하는 알고리즘’ 텍스트 자체만 읽으면 헷갈리거나 어려울 것 없다.
하지만, 데이터의 구조가 어떻길래 깊은 부분이 있지? 라고 생각할 수가 있다. 
DFS는 바로 그래프 즉, 노드와 노드를 연결하여 모아둔 비선형 자료구조를 탐색할 때 사용하는 알고리즘이다. 
```

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">그래프</span>

![graph](https://user-images.githubusercontent.com/33630505/153203015-0a8bffa5-1230-4897-b444-ca48f5efcc09.png)

<p>그래프는 위와 같은 형태를 하고 있으며 한 번쯤은 들어봤을 법한 데이터 구조인 트리구조 역시 그래프의 일종이다. 
이런 그래프는 연결된 노드 간의 관계를 표현할 수 있는 자료구조라고 볼 수 있다</p>

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">그래프 탐색</span>

> 그래프 탐색은 하나의 정점에서 시작해 차례대로 모든 정점들을 한 번씩 방문하는 것을 말한다 

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">DFS의 탐색과정</span>


![DFS1](https://user-images.githubusercontent.com/33630505/153203211-16ac1344-8232-4da3-ae3d-d92fd83ce8ff.png)

<p>DFS는 그래프 탐색을 하는 알고리즘으로써 가장 상위 노드에서 부터 시작해 가장 하위 노드가 있는 방향을 우선순위로 탐색을 진행한다. 
이때 DFS는 스택 자료구조(stack)를 통해 구현한다 </p>

![DFS2](https://user-images.githubusercontent.com/33630505/153203340-7991592a-9e84-4fc4-b14c-5d845928cb7d.png)

<p>트리의 가장 윗부분인 루트 노드를 기준으로 탐색을 시작한다. 

루트 노드와 연결된 노드 중에서 방문하지 않은 노드가 있는지 확인을 한다. 

방문하지 않은 노드는 스택에 넣는다

(동일한 깊이의 인접한 노드가 여러개가 있다면 이 중 노드의 번호가 작은 것 부터 차례대로 탐색하는 것이 관행이다)</p>

![DFS3](https://user-images.githubusercontent.com/33630505/153203535-25f19305-60d8-4a8f-a4c7-2e38b8682572.png)

<p>더 이상 방문하지 않은 노드가 없으면 탐색이 완료되었다고 판단하고 스택에서 현재 노드를 제외한다 </p>

![DFS4](https://user-images.githubusercontent.com/33630505/153203620-e9f3f27f-0d5d-4069-9c01-256034661906.png)

<p>계속해서 현재 노드에서 하위 노드에 방문하지 않은 노드가 있는지 없는지 확인하고 없으면 상위 노드로 다시 이동하고 있으면 하위 노드로 이동 후 현재 노드를 스택에서 제외한다</p>

![DFS5](https://user-images.githubusercontent.com/33630505/153203723-743283f7-c722-4ed0-9530-f267b179f32d.png)

<p>위와 같은 과정을 계속해서 반복하고 가장 바깥쪽 노드의 하위 노드까지 탐색을 진행한다 </p>

![DFS6](https://user-images.githubusercontent.com/33630505/153203821-c687269c-0929-4575-8f43-785dbd59e488.png)

<p>방문하지 않은 노드가 더 이상 존지해지 않으면 해당 노드를 스택에서 제거하고 탐색을 종료한다 </p>

<br>

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">[DFS 구현하기]</span> <br>

```python
# DFS - 깊이 우선 탐색 -> 스택을 이용
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]
# 방문노드
visited = [False] * len(graph)

def dfs(graph, v, visited):
    visited[v] = True
    print(v, end=' ')
    
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

# 0번 노드가 없으니 1번 노드부터 탐색
dfs(graph, 1, visited)
```

<br>

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">[DFS/BFS 탐색 순서 차이점]</span> 


<img width="606" alt="image" src="https://user-images.githubusercontent.com/33630505/156115826-dae78db4-a4c3-409b-b114-6fbabcde7ec0.png">


<br>

<hr>

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">[BFS]</span> 

```python
DFS와는 반대로 ‘넓이 우선 탐색’을 하는 알고리즘이다 
루트 노드에서 시작해서 정점으로터 가장 인접한 노드를 탐색하는 방법이다 
주로 두 노드 사이의 최단 경로 혹은 임의의 경로를 찾을 때 자주 사용하는 방법이다 
```

<br>

<span style="font-size: 23px; color: rgb(75,75,75); padding: 2px;">BFS의 탐색과정</span>

```python
BFS는 큐 자료구조를 이용한다 
1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다 
2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리한다 
3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다 
```

<br>

![BFS](https://user-images.githubusercontent.com/33630505/153204555-52c1bd6a-b0b4-427b-96cc-a8b45208d2a9.png)

<br>

![BFS1](https://user-images.githubusercontent.com/33630505/153204552-d9eafff0-3c5b-49cf-af51-45ab116dd4c2.png)

<br>

![BFS2](https://user-images.githubusercontent.com/33630505/153204543-c50c8024-9154-4779-a0fa-91f69a286a15.png)

<br>

![BFS3](https://user-images.githubusercontent.com/33630505/153204539-c8c1fe3f-0ec6-49ea-a390-2f2a4d3092c6.png)

<br>

![BFS4](https://user-images.githubusercontent.com/33630505/153204536-85537b1e-3c22-4da5-ad9a-8441771fbf90.png)

<br>

![BFS5](https://user-images.githubusercontent.com/33630505/153204530-c95f4dd5-19d3-4aae-b324-fe596cf53c2e.png)

<br>


<span style="font-size: 23px; color: rgb(103, 143, 133); padding: 2px;">BFS 장단점</span>

*장점* 

- 노드의 수가 적고 깊이가 얕은 경우 빠르게 탐색할 수 있다 
- 단순 검색 속도가 DFS보다 빠르다 
- 답이 되는 경로가 여러개인 경우에도 최단 경로임을 보장한다
- 최단 경로가 존재한다고 했을 때 어느 한 경로가 무한히 깊어진다고 해도 최단 경로를 반드시 찾을 수 있다

<br>

*단점*

- 재귀호출의 DFS와는 달리 다음에 탐색할 정점들을 저장해야 하기 때문에 저장공간이 많이 필요하다
- 노드의 수가 늘어나면 탐색해야 하는 노드 또한 많아지기 때문에 비현실적이다

<br>

<span style="font-size: 30px; color: rgb(75,163,123); padding: 2px;">[BFS 구현하기]</span> <br>

```python
from collections import deque

# BFS 함수 정의
def bfs(graph, start, visited):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    
    # 현재 노드를 방문 처리
    visited[start] = True
    
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end=' ')
        
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
```

<br>

### Reference 

```python
1. https://velog.io/@sohi_5/algorithmDFS
2. https://velog.io/@mgm-dev/간략-자료구조-정리
3. https://security-nanglam.tistory.com/413
4. https://coding-factory.tistory.com/612
5. https://spacebike.tistory.com/39
```