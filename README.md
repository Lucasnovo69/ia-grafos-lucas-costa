# ia-grafos-lucas-costa
#Atividade Avaliativa
# ==============================================================
# PARTE A — DIJKSTRA (CAMINHO MÍNIMO EM GRAFOS PONDERADOS POSITIVOS)
# ==============================================================
import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    parent = {node: None for node in graph}
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > dist[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                parent[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    return dist, parent

def reconstruct_path(parent, target):
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path

graph = {
    'Minha Casa': [('A', 12), ('B', 2)],
    'A': [('C', 5), ('D', 14)],
    'B': [('A', 1), ('D', 7)],
    'C': [('D', 4), ('Faculdade', 10)],
    'D': [('Faculdade', 21)],
    'Faculdade': []
}

dist, parent = dijkstra(graph, 'Minha Casa')
path = reconstruct_path(parent, 'Faculdade')

print("Custo mínimo até a Faculdade:", dist['Faculdade'], "minutos")
print("Rota encontrada:", " → ".join(path))


# ==============================================================
# PARTE B — A-STAR (BUSCA INFORMADA COM HEURÍSTICA)
# ==============================================================
import numpy as np
import random

def generate_grid(size=20, obstacle_prob=0.15):
    grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i][j] = 1
    return grid

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(pos, grid):
    x, y = pos
    steps = [(1,0), (-1,0), (0,1), (0,-1)]
    result = []
    for dx, dy in steps:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx][ny] == 0:
            result.append((nx, ny))
    return result

def a_star(grid, start, goal, h):
    pq = [(h(start, goal), 0, start)]
    parent = {start: None}
    g_score = {start: 0}

    while pq:
        f, cost, current = heapq.heappop(pq)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        for neighbor in neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                parent[neighbor] = current
                heapq.heappush(pq, (f_score, tentative_g, neighbor))
    return None

grid = generate_grid()
start, goal = (0, 0), (19, 19)
path = a_star(grid, start, goal, heuristic)

if path:
    print("Caminho encontrado pelo robô autônomo:")
    print(path)
    display_grid = grid.copy()
    for x, y in path:
        display_grid[x][y] = 2
    print("\nGrid (0 = livre, 1 = obstáculo, 2 = caminho):")
    print(display_grid)
else:
    print("Nenhum caminho encontrado")


# ==============================================================
# PARTE C — ÁRVORES BINÁRIAS E PERCURSOS
# ==============================================================
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root

def in_order(root):
    if root:
        in_order(root.left)
        print(root.value, end=' ')
        in_order(root.right)

def pre_order(root):
    if root:
        print(root.value, end=' ')
        pre_order(root.left)
        pre_order(root.right)

def post_order(root):
    if root:
        post_order(root.left)
        post_order(root.right)
        print(root.value, end=' ')

values = [50, 30, 70, 20, 40, 60, 80, 35, 45]
root = None
for v in values:
    root = insert(root, v)

print("Percurso em-ordem:")
in_order(root)
print("\nPercurso pré-ordem:")
pre_order(root)
print("\nPercurso pós-ordem:")
post_order(root)
