import numpy as np
import cv2

class ObjectTrackerCV:

    def __init__(self, initial_position, initial_timestamp):
        self.state = np.array([[initial_position[0]], [initial_position[1]], [0], [0], [0], [0]], dtype=np.float32)
        self.measurement = np.array([[np.nan], [np.nan]], dtype=np.float32)
        self.kf = cv2.KalmanFilter(6, 2, 0)
        self.update_transition_matrix(dt=1.0)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.0001
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.005
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = self.state
        self.last_timestamp = initial_timestamp

    def update_transition_matrix(self, dt):
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0, 0.5 * dt ** 2, 0], [0, 1, 0, dt, 0, 0.5 * dt ** 2], [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=np.float32)

    def update(self, observed_position, timestamp):
        if hasattr(self, 'last_timestamp'):
            dt = timestamp - self.last_timestamp
            self.update_transition_matrix(dt)
        self.last_timestamp = timestamp
        self.measurement = np.array([[observed_position[0]], [observed_position[1]]], dtype=np.float32)
        self.kf.correct(self.measurement)

    def predict(self, dt=None):
        if dt is not None:
            self.update_transition_matrix(dt)
        predicted_state = self.kf.predict()
        predicted_position = (predicted_state[0].item(), predicted_state[1].item())
        return predicted_position
