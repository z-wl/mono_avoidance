import numpy as np
import collections
import cv2

class AreaTracker:

    def __init__(self, initial_area, initial_area_rate=0, initial_timestamp=0, window=24, mode='area', use_r=False, measure_noi=0.125, use_acc=False, processNoiseCov=0.0001, scale=1):
        self.window_size = window
        self.acc = use_acc
        self.scale = scale
        if use_acc:
            self.state = np.array([[initial_area], [initial_area_rate], [0.0]], dtype=np.float32)
        else:
            self.state = np.array([[initial_area], [initial_area_rate]], dtype=np.float32)
        state_num = self.state.shape[0]
        self.growth_rates = collections.deque([0])
        self.area_log = collections.deque([initial_area])
        self.timestamps = collections.deque([initial_timestamp])
        self.mode = mode
        self.scale = 1.2
        self.stable_step = 20
        self.kf = cv2.KalmanFilter(state_num, 1, 0)
        self.update_transition_matrix(dt=1.0)
        if use_acc:
            self.kf.measurementMatrix = np.array([[1, 0, 0]], dtype=np.float32)
        else:
            self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(state_num, dtype=np.float32) * processNoiseCov
        self.kf.measurementNoiseCov = np.array([[measure_noi]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(state_num, dtype=np.float32) * 1
        self.kf.statePost = self.state.copy()
        self.use_r = use_r
        self.last_timestamp = initial_timestamp
        self.kf.predict()
        self.k = 3000

    def rate(self):
        return self.growth_rates[-1]

    def update_transition_matrix(self, dt):
        if self.acc:
            self.kf.transitionMatrix = np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]], dtype=np.float32)
        else:
            self.kf.transitionMatrix = np.array([[1, dt], [0, 1]], dtype=np.float32)

    def update(self, observed_area, timestamp, weight=None):
        if hasattr(self, 'last_timestamp'):
            dt = timestamp - self.last_timestamp
            self.update_transition_matrix(dt)
        self.last_timestamp = timestamp
        self.timestamps.append(timestamp)
        if self.use_r and weight is not None:
            r = self.k / weight ** 0.5
            self.kf.measurementNoiseCov = np.array([[r]], dtype=np.float32)
        if not np.isnan(observed_area):
            self.measurement = np.array([[observed_area]], dtype=np.float32)
            self.kf.correct(self.measurement)
        else:
            print('Warning: Invalid measurement data, skipping correction.')

    def _smooth(self, var, mode='rate'):
        if mode in ['rate']:
            cur = var[-1]
            cur_time = self.timestamps[-1]
            if len(var) <= self.window_size:
                last = var[0]
                last_time = self.timestamps[0]
            else:
                last = var[-1 - self.window_size]
                last_time = self.timestamps[-1 - self.window_size]
            return (cur - last) / last / (cur_time - last_time)
        elif mode in ['area']:
            cur = var[-1]
            cur_time = self.timestamps[-1]
            if len(var) <= self.stable_step:
                last = var[-2]
                last_time = self.timestamps[-2]
            elif len(var) - self.stable_step < self.window_size:
                last = var[self.stable_step - 1]
                last_time = self.timestamps[self.stable_step - 1]
            else:
                last = var[-self.window_size]
                last_time = self.timestamps[-self.window_size]
            try:
                rs = (cur - last) / (cur_time - last_time)
            except ZeroDivisionError as e:
                print(e)
            return rs
        else:
            return 0

    def predict(self):
        predicted_state = self.kf.predict()
        predicted_area = predicted_state[0].item()
        self.area_log.append(predicted_area)
        self.growth_rates.append(self._smooth(self.area_log, self.mode))
        return predicted_area
