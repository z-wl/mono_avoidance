import numpy as np
import casadi as cs
from mpl_toolkits.mplot3d import Axes3D
import random
from math import sqrt
import math
import matplotlib.pyplot as plt

def bspline_basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    if knots[i + k] == knots[i]:
        c1 = 0.0
    else:
        c1 = (t - knots[i]) / (knots[i + k] - knots[i]) * bspline_basis(i, k - 1, t, knots)
    if knots[i + k + 1] == knots[i + 1]:
        c2 = 0.0
    else:
        c2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * bspline_basis(i + 1, k - 1, t, knots)
    return c1 + c2

def bspline_curve(control_points, t, knots, k=3, mode='usual'):
    point = cs.MX.zeros(1, 3) if mode == 'usual' else np.zeros((1, 3))
    for i in range(control_points.shape[0]):
        if isinstance(control_points[i, :], np.ndarray):
            point += control_points[i, :].reshape(1, 3) * bspline_basis(i, k, t, knots)
        else:
            point += control_points[i, :] * bspline_basis(i, k, t, knots)
    return point

def min_distance_to_obstacles(point, obstacle_trajectories):
    min_distance = cs.inf
    for trajectory in obstacle_trajectories:
        trajectory_mx = cs.MX(np.vstack(trajectory).reshape(len(trajectory), 3))
        distances = cs.sqrt(cs.sum2((trajectory_mx - cs.repmat(point, len(trajectory), 1)) ** 2))
        min_distance = cs.fmin(min_distance, cs.mmin(distances))
    return min_distance

def plan_path_bspline(start, goal, obstacle_trajectories, n=1.0, num_control_points=10, num_samples=50):
    control_points = cs.MX.sym('P', (num_control_points, 3))
    knots = np.linspace(0, 1, num_control_points + 4)
    path_length = 0
    smoothness = 0
    for i in range(num_control_points - 1):
        delta = control_points[i + 1, :] - control_points[i, :]
        path_length += cs.sumsqr(delta)
    for i in range(1, num_control_points - 1):
        smoothness += cs.sumsqr(control_points[i + 1, :] - 2 * control_points[i, :] + control_points[i - 1, :])
    objective = path_length + 0.1 * smoothness
    constraints = []
    for i in range(3):
        constraints.append(control_points[0, i] == start[i])
        constraints.append(control_points[-1, i] == goal[i])
    t_samples = np.linspace(0, 1, num_samples)
    for t in t_samples:
        point = bspline_curve(control_points, t, knots)
        d = min_distance_to_obstacles(point, obstacle_trajectories)
        constraints.append(d >= n)
    nlp = {'x': cs.vec(control_points), 'f': objective, 'g': cs.vertcat(*constraints)}
    solver = cs.nlpsol('solver', 'ipopt', nlp)
    initial_guess = np.linspace(start, goal, num_control_points)
    result = solver(x0=initial_guess.flatten())
    optimized_control_points = np.array(result['x']).reshape((num_control_points, 3))
    trajectory = np.array([bspline_curve(optimized_control_points, t, knots, mode='final') for t in t_samples])
    return trajectory

class Node:

    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent

class OccupancyGrid:

    def __init__(self, resolution, bounds):
        self.resolution = resolution
        self.bounds = bounds
        self.grid = {}

    def position_to_index(self, position):
        return tuple((position - np.array([b[0] for b in self.bounds])) // self.resolution)

    def mark_collision_sphere(self, center, radius):
        min_corner = center - radius
        max_corner = center + radius
        min_index = self.position_to_index(min_corner)
        max_index = self.position_to_index(max_corner)
        min_index = [int(i) for i in min_index]
        max_index = [int(i) for i in max_index]
        for x in range(min_index[0], max_index[0] + 1):
            for y in range(min_index[1], max_index[1] + 1):
                for z in range(min_index[2], max_index[2] + 1):
                    self.grid[x, y, z] = True

    def is_collision(self, position):
        index = self.position_to_index(position)
        return (int(index[0]), int(index[1]), int(index[2])) in self.grid

def rrt(start, goal, obstacle_segments, a, step_size=0.5, max_iter=1000, goal_sample_rate=0.5, search_space=(-10, 10)):
    resolution = a
    bounds = [search_space, search_space, search_space]
    occupancy_grid = OccupancyGrid(resolution, bounds)
    for trail in obstacle_segments:
        for point in trail:
            occupancy_grid.mark_collision_sphere(np.array(point).reshape((3,)), a)
    start_node = Node(start)
    goal = np.array(goal)
    tree = [start_node]
    last_is_collision = 100
    init_goal_sample_rate = 1.0
    restart_goal_sample_rate = goal_sample_rate
    goal_sample_rate = init_goal_sample_rate
    for _ in range(max_iter):
        if random.random() < goal_sample_rate and last_is_collision != 0:
            sample = goal.astype(float)
        else:
            sample = np.array([random.uniform(*search_space) for _ in range(3)])
        nearest = min(tree, key=lambda node: np.linalg.norm(node.position - sample))
        direction = sample - nearest.position
        distance = np.linalg.norm(direction)
        if distance == 0:
            continue
        direction_unit = direction / distance
        new_position = nearest.position + direction_unit * min(step_size, distance)
        if not occupancy_grid.is_collision(new_position):
            last_is_collision += 1
            new_node = Node(new_position, parent=nearest)
            tree.append(new_node)
            if np.linalg.norm(new_node.position - goal) < step_size:
                final_node = Node(goal, parent=new_node)
                path = []
                current = final_node
                while current is not None:
                    path.append(current.position.tolist())
                    current = current.parent
                return path[::-1]
        else:
            last_is_collision = 0
        goal_sample_rate = restart_goal_sample_rate + (1 - restart_goal_sample_rate) * (1 - 1 / (last_is_collision + 1 + 1e-06))
        if _ == max_iter - 1:
            final_node = tree[-1]
            path = []
            current = final_node
            while current is not None:
                path.append(current.position.tolist())
                current = current.parent
            return path[::-1]
    return None

def show_trail(trail, ax=None, fig=None, name=None):
    import time
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='b', arrow_length_ratio=0.1)
    if isinstance(trail, list):
        for t in trail:
            ax.plot([p[0] for p in t][4:], [p[1] for p in t][4:], [p[2] for p in t][4:], color='green')
    else:
        for obs in trail.obstacles.values():
            if len(obs.tra_cache) < 10:
                continue
            try:
                ax.plot([p[0] for p in obs.tra_cache][4:], [p[1] for p in obs.tra_cache][4:], [p[2] for p in obs.tra_cache][4:], color='red')
                ax.plot([obs.tra_cache[-1][0]] + [p[0] for p in obs.next_trajectory()], [obs.tra_cache[-1][1]] + [p[1] for p in obs.next_trajectory()], [obs.tra_cache[-1][2]] + [p[2] for p in obs.next_trajectory()], color='green')
            except ValueError as e:
                print(e)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    name = name if name is not None else ''
    plt.savefig(f'./log/{name}_trail_{time.time()}.png')

def adjust_direction_to_avoid_obstacle(perpendicular_dir, obstacle_center, obstacle_size, current_point, max_tilt_angle=45):

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def calculate_angle(vector):
        x, y = vector
        angle = np.arctan2(y, x) * (180 / np.pi)
        return angle

    def angle_between_vectors(v1, v2):
        angle1 = calculate_angle(v1)
        angle2 = calculate_angle(v2)
        angle_diff = angle2 - angle1
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff <= -180:
            angle_diff += 360
        return angle_diff

    def rotate_vector_around_z_axis(vector, theta):
        x, y, z = vector
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy_vector = np.array([[x], [y]])
        rotated_xy_vector = np.dot(rotation_matrix, xy_vector)
        rotated_vector = np.array([rotated_xy_vector[0, 0], rotated_xy_vector[1, 0], z])
        return rotated_vector
    obstacle_center = np.array(obstacle_center)
    current_point = np.array(current_point)
    obstacle_vector = obstacle_center[:2] - current_point[:2]
    lr_limit = np.degrees(math.atan(obstacle_size / np.linalg.norm(obstacle_vector)))
    cone_axis = normalize(obstacle_vector)
    lr_bias = angle_between_vectors(obstacle_vector, perpendicular_dir[:2])
    l_diff = lr_limit - lr_bias
    r_diff = lr_limit + lr_bias
    if l_diff < 0 or r_diff < 0:
        return None
    if l_diff < r_diff:
        adjust_dir = rotate_vector_around_z_axis(perpendicular_dir, np.radians(l_diff))
    else:
        adjust_dir = rotate_vector_around_z_axis(perpendicular_dir, -np.radians(r_diff))
    return adjust_dir

def repulsion(ori, target, obstacle_trails, a=1, current_pos=None, static_obs=None, static_obs_size=3.5):

    def is_danger(trail_point):
        if np.linalg.norm(trail_point) < a:
            return True
        return False

    def get_avoid(ori, start, end):
        d = end - start
        ap = ori - end
        c = np.cross(d, ap)
        distance = np.linalg.norm(c) / (np.linalg.norm(d) + 1e-06)
        avoid_distance = a - distance
        t = np.dot(ap, d) / (np.dot(d, d) + 1e-06)
        p = end + t * d
        pos = ori - p
        return 0.7 * avoid_distance * pos / np.linalg.norm(pos)
    danger_trails = []
    collision_frames = {}
    for j, trail in enumerate(obstacle_trails):
        collision_frames[j] = 1000000000.0
        for i, point in enumerate(trail[1:]):
            if is_danger(point) and i < collision_frames[j]:
                collision_frames[j] = i
                danger_trails.append(j)
                break
    if len(danger_trails) == 0:
        return None

    def show(s, e, p, ax, fig):
        ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='b', arrow_length_ratio=0.1)
        more_end = s + 100 * (e - s)
        ax.plot([s[0], more_end[0]], [s[1], more_end[1]], [s[2], more_end[2]])
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color='red')
        fig.show()
    max_len = 0
    final = np.array([0.0, 0.0, 0.0])
    for key in danger_trails:
        danger_trail = obstacle_trails[key]
        ori_ = np.zeros((3,))
        start_ = danger_trail[collision_frames[key] - 1].reshape((3,))
        end_ = danger_trail[collision_frames[key]].reshape((3,))
        avoid_pos = get_avoid(ori_, start_, end_)
        if np.linalg.norm(avoid_pos) > max_len:
            max_len = np.linalg.norm(avoid_pos)
        final += avoid_pos
    final = np.array([final[2], -final[0], -final[1]])
    final = max_len * final / np.linalg.norm(final)
    if static_obs:
        for obs in static_obs:
            distance = np.linalg.norm(np.array(obs[:2]) - np.array(current_pos[:2]))
            if distance > 3 * static_obs_size:
                continue
            adjust_dir = adjust_direction_to_avoid_obstacle(final, obs, static_obs_size, current_pos)
            if adjust_dir is None:
                continue
            final = adjust_dir
    return [np.array([0, 0, 0]), final]

def vis_tra(ax, trails, predicts, avoid=None, target=None, all_obs=None):
    now = 0
    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='b', arrow_length_ratio=0.1)
    for key in trails.keys():
        trail = trails[key]
        if len(trail) < 30:
            continue
        predict = predicts[key]
        t_color = (0, 0, 1)
        p_color = (0, 1, 0)
        ax.plot([p[0] for p in trail][6:], [p[1] for p in trail][6:], [p[2] for p in trail][6:], color=t_color)
        ax.plot([trail[-1][0]] + [p[0] for p in predict], [trail[-1][1]] + [p[1] for p in predict], [trail[-1][2]] + [p[2] for p in predict], color=p_color, linewidth=2, alpha=0.5)
        now += 1
        plt.savefig(f'output{key}.png')
    plt.show()
    pass

def discretize_direction(vector_3d):
    x, y, _ = vector_3d
    directions = {'left': np.array([0.0, 1.0]), 'right': np.array([0.0, -1.0]), 'forward_left': np.array([1.0, 1.0]) / np.sqrt(2), 'forward_right': np.array([1.0, -1.0]) / np.sqrt(2), 'backward_left': np.array([-1.0, 1.0]) / np.sqrt(2), 'backward_right': np.array([-1.0, -1.0]) / np.sqrt(2)}
    if np.linalg.norm([x, y]) < 1e-06:
        return ('none', np.array([0, 0]))
    input_vec = np.array([x, y])
    input_vec = input_vec / np.linalg.norm(input_vec)
    best_match = None
    max_cosine = -np.inf
    for name, dir_vec in directions.items():
        cosine = np.dot(input_vec, dir_vec)
        if cosine > max_cosine:
            max_cosine = cosine
            best_match = (name, dir_vec)
    return best_match

def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def euler_to_quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return {'x': x, 'y': y, 'z': z, 'w': w}
