import numpy as np

class CameraParams:

    def __init__(self, cx, cy, fx, fy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def compute_intrinsic_matrix(fov, aspect, nearVal, width, height):
    import math
    vFovRadians = math.radians(fov)
    fy_world_units = nearVal / math.tan(vFovRadians / 2)
    fy = fy_world_units * height / (2 * nearVal)
    fx = fy * aspect
    cx = width / 2
    cy = height / 2
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return CameraParams(cx, cy, fx, fy)

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
import cv2

class ThreeDimensionalTracker:
    SPEED = 'speed'
    ACC = 'acc'

    def __init__(self, initial_position, initial_velocity=None, initial_acceleration=None, initial_timestamp=0, mode='speed'):
        if initial_velocity is None:
            initial_velocity = [0, 0, 0]
        if initial_acceleration is None:
            initial_acceleration = [0, 0, 0]
        self.mode = mode
        if mode == ThreeDimensionalTracker.ACC:
            self.state = np.array([[initial_position[0]], [initial_position[1]], [initial_position[2]], [initial_velocity[0]], [initial_velocity[1]], [initial_velocity[2]], [initial_acceleration[0]], [initial_acceleration[1]], [initial_acceleration[2]]], dtype=np.float32)
        elif mode == ThreeDimensionalTracker.SPEED:
            self.state = np.array([[initial_position[0]], [initial_position[1]], [initial_position[2]], [initial_velocity[0]], [initial_velocity[1]], [initial_velocity[2]]], dtype=np.float32)
        state_num = self.state.shape[0]
        self.kf = cv2.KalmanFilter(state_num, 3, 0)
        self.update_transition_matrix(dt=1.0)
        if mode == ThreeDimensionalTracker.ACC:
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        elif mode == ThreeDimensionalTracker.SPEED:
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(state_num, dtype=np.float32) * 0.0001
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.005
        self.kf.errorCovPost = np.eye(state_num, dtype=np.float32) * 0.001
        self.kf.statePost = self.state.copy()
        self.kf.predict()
        self.kf_copy = cv2.KalmanFilter(state_num, 3, 0)
        self.copy_self()
        self.last_timestamp = initial_timestamp

    def update_transition_matrix(self, dt):
        if self.mode == ThreeDimensionalTracker.ACC:
            self.kf.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0, 0.5 * dt ** 2, 0, 0], [0, 1, 0, 0, dt, 0, 0, 0.5 * dt ** 2, 0], [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt ** 2], [0, 0, 0, 1, 0, 0, dt, 0, 0], [0, 0, 0, 0, 1, 0, 0, dt, 0], [0, 0, 0, 0, 0, 1, 0, 0, dt], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        else:
            self.kf.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=np.float32)

    def update(self, observed_position, timestamp):
        self.last_timestamp = timestamp
        if not np.any(np.isnan(observed_position)):
            self.measurement = np.array([[observed_position[0]], [observed_position[1]], [observed_position[2]]], dtype=np.float32)
            self.kf.correct(self.measurement)
            pass
        else:
            print('Warning: Invalid measurement data, skipping correction.')

    def copy_self(self):
        self.kf_copy.transitionMatrix = self.kf.transitionMatrix.copy()
        self.kf_copy.measurementMatrix = self.kf.measurementMatrix.copy()
        self.kf_copy.measurementNoiseCov = self.kf.measurementNoiseCov.copy()
        self.kf_copy.processNoiseCov = self.kf.processNoiseCov.copy()
        self.kf_copy.errorCovPost = self.kf.errorCovPost.copy()
        self.kf_copy.errorCovPre = self.kf.errorCovPre.copy()
        self.kf_copy.statePost = self.kf.statePost.copy()
        self.kf_copy.statePre = self.kf.statePre.copy()

    def predict_trajectory(self, max_step=10):
        predict_tra = []
        while max_step > 0:
            next_state = self.kf_copy.predict()
            predict_tra.append(next_state[:3].copy())
            max_step -= 1
        return predict_tra

    def predict(self, dt=None):
        self.copy_self()
        if dt is not None:
            self.update_transition_matrix(dt)
        predicted_state = self.kf.predict()
        predicted_position = (predicted_state[0], predicted_state[1], predicted_state[2])
        return predicted_position
