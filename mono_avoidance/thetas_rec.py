import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def visualize_path(start, goals, obstacles, path=None, title='Theta* Path Planning'):
    fig, ax = plt.subplots(figsize=(10, 10))
    if isinstance(goals, list):
        goal = goals[0]
    else:
        goal = goals
    all_x = [start[0], goal[0]] + [p[0] for p in path] if path.shape[0] > 0 else []
    all_y = [start[1], goal[1]] + [p[1] for p in path] if path.shape[0] > 0 else []
    for rect in obstacles:
        all_x.extend([rect[0][0], rect[1][0]])
        all_y.extend([rect[0][1], rect[1][1]])
    padding = 1
    x_min, x_max = (min(all_x) - padding, max(all_x) + padding)
    y_min, y_max = (min(all_y) - padding, max(all_y) + padding)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(title)
    for rect in obstacles:
        (x1, y1), (x2, y2) = rect
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        bottom_left = (min(x1, x2), min(y1, y2))
        rect_patch = patches.Rectangle(bottom_left, width, height, linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
        ax.add_patch(rect_patch)
    ax.plot(start[0], start[1], 'go', markersize=12, label='Start')
    if isinstance(goals, list):
        for g in goals:
            ax.plot(g[0], g[1], 'yo', markersize=12, label='Goal')
    if path.shape[0] > 0:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        ax.plot(path_x, path_y, 'bo', markersize=4)
    ax.legend(loc='upper right')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(f'output_path_{time.time()}.png')
    plt.show()

class NodeT:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.parent = None

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def is_point_in_rect(point, rect):
    x, y = point
    (x1, y1), (x2, y2) = rect
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    return min_x <= x <= max_x and min_y <= y <= max_y

def is_collision_theta(pos, obstacles):
    for rect in obstacles:
        if is_point_in_rect((pos.x, pos.y), rect):
            return True
    return False

def line_rect_intersection(line_start, line_end, rect):
    (x1, y1), (x2, y2) = rect
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    if is_point_in_rect(line_start, rect) or is_point_in_rect(line_end, rect):
        return True
    if line_intersects_vertical(line_start, line_end, min_x, min_y, max_y):
        return True
    if line_intersects_vertical(line_start, line_end, max_x, min_y, max_y):
        return True
    if line_intersects_horizontal(line_start, line_end, min_y, min_x, max_x):
        return True
    if line_intersects_horizontal(line_start, line_end, max_y, min_x, max_x):
        return True
    return False

def line_intersects_vertical(p1, p2, x, y_min, y_max):
    if p1[0] < x and p2[0] < x or (p1[0] > x and p2[0] > x):
        return False
    t = (x - p1[0]) / (p2[0] - p1[0] + 1e-09)
    y = p1[1] + t * (p2[1] - p1[1])
    return y_min <= y <= y_max

def line_intersects_horizontal(p1, p2, y, x_min, x_max):
    if p1[1] < y and p2[1] < y or (p1[1] > y and p2[1] > y):
        return False
    t = (y - p1[1]) / (p2[1] - p1[1] + 1e-09)
    x = p1[0] + t * (p2[0] - p1[0])
    return x_min <= x <= x_max

def line_of_sight_f(node1, node2, obstacles):
    p1 = (node1.x, node1.y)
    p2 = (node2.x, node2.y)
    for rect in obstacles:
        if line_rect_intersection(p1, p2, rect):
            return False
    return True

def theta_star(start, goal, obstacles, step=1.0, threshold_multiplier=2.0):
    open_set = []
    closed_set = set()
    if not isinstance(start, NodeT):
        start_node = NodeT(start[0], start[1])
        goal_node = NodeT(goal[0], goal[1])
    else:
        start_node = start
        goal_node = goal
    start_node.g = 0
    start_node.h = euclidean_distance(start_node, goal_node)
    heapq.heappush(open_set, start_node)
    directions = [(0, step), (0, -step), (step, 0), (-step, 0), (step, step), (step, -step), (-step, step), (-step, -step)]
    threshold = step * threshold_multiplier
    while open_set:
        current = heapq.heappop(open_set)
        if euclidean_distance(current, goal_node) < threshold:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        closed_set.add((current.x, current.y))
        for dx, dy in directions:
            x = current.x + dx
            y = current.y + dy
            pos = (x, y)
            if (x, y) in closed_set:
                continue
            neighbor = NodeT(x, y)
            neighbor.h = euclidean_distance(neighbor, goal_node)
            if current.parent and line_of_sight_f(current.parent, neighbor, obstacles):
                tentative_g = current.parent.g + euclidean_distance(current.parent, neighbor)
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current.parent
                    heapq.heappush(open_set, neighbor)
            elif not is_collision_theta(neighbor, obstacles):
                tentative_g = current.g + euclidean_distance(current, neighbor)
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current
                    heapq.heappush(open_set, neighbor)
    return None
