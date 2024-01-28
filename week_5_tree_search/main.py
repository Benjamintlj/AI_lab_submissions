# BFS algorithm in Python
import collections


# BFS algorithm
def bfs(graph, root):
    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


# DFS algorithm
def dfs(graph, root):
    visited = set()
    queue = collections.deque([root])
    visited.add(root)

    while queue:
        # Dequeue a vertex from queue
        vertex = queue.pop()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


if __name__ == '__main__':
    graph = {
        'A': ['C', 'B'],
        'B': ['A', 'F'],
        'C': ['A', 'D', 'F', 'G'],
        'D': ['C', 'E'],
        'E': ['D', 'G'],
        'F': ['B', 'C', 'G'],
        'G': ['F', 'C', 'E']
    }

    bfs(graph, 'A')
    print('')
    dfs(graph, 'A')
