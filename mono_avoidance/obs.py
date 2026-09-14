import collections
import numpy as np
import time
from .adptive_area import AreaTracker
from .pos_predictor import ObjectTrackerCV
from .ttc_predictor import ThreeDimensionalTracker, CameraParams
from .path_plan import plan_path_bspline, rrt, repulsion, show_trail

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def get_3d_direction(camera, position):
    u, v = position
    x_n = (u - camera.cx) / camera.fx
    y_n = (v - camera.cy) / camera.fy
    direction = normalize(np.array([x_n, y_n, 1]))
    return direction

class Obstacle:

    def __init__(self, obs_id, camera_param=CameraParams(480, 360, 550, 550), window=10, width=960, height=720, alpha=80000, danger_threshold=0.05, danger_atte=0.85, _class=None, init_time=None):
        self.k = camera_param
        self.frames_value = collections.deque([])
        self.frames_interval = collections.deque([])
        self.now_frame = 0
        self.speed = 0.0
        self.danger_rate = 0.0
        self.last_pos = np.array([0.0, 0.0, 0.0])
        self.mov_vec = np.array([0.0, 0.0, 0.0])
        self.window = window
        self.width = width
        self.height = height
        self.screen_area = self.height * self.width
        self.danger_alpha = alpha
        self.timestamp = time.perf_counter() if not init_time else init_time
        self.obs_id = obs_id
        self.danger_corr = 1 / danger_threshold
        self.danger_atte = danger_atte
        self.epsilon = self.danger_alpha * self.danger_corr / self.screen_area ** 1.5
        self._class = _class

    def update(self, value, pos, height, timestamp=None):
        if pos is None:
            return
        if isinstance(pos, tuple):
            pos = get_3d_direction(self.k, pos)
        self.frames_value.append(value)
        while self.window < len(self.frames_value):
            self.frames_value.popleft()
        now_time = time.perf_counter() if not timestamp else timestamp
        self.frames_interval.append(now_time)
        while self.window < len(self.frames_interval):
            self.frames_interval.popleft()
        last_pos = self.last_pos
        self.last_pos = np.array(pos)
        if last_pos is not None:
            self.mov_vec = pos - last_pos
        else:
            self.mov_vec = pos - pos
        self.speed = 0.0
        last_frame = None
        last_time = None
        self.speed = (self.frames_value[-1] - self.frames_value[0]) / (1e-06 + self.frames_interval[-1] - self.frames_interval[0]) / (1e-06 + self.frames_value[0]) if len(self.frames_value) > 1 else 0.0
        self.danger_rate = self.speed if self.speed > 0 else self.danger_rate * self.danger_atte
        self.danger_rate = self.danger_rate if self.danger_rate <= 10 else 10
        self.timestamp = now_time

    def empty_update(self, timestamp=None):
        self.danger_rate = self.danger_rate * self.danger_atte
        self.last_pos = self.last_pos + self.mov_vec
        pass

    def get_timestamp(self):
        return self.timestamp

    def get_id(self):
        return self.obs_id

    def get_class(self):
        return self._class

    def __repr__(self):
        return repr(f'[obs_id: {self.obs_id}, window: {self.window}, frame_value: {self.frames_value}, danger_rate: {self.danger_rate}, mov_vec: {self.mov_vec}, timestamp: {self.timestamp}, speed: {self.speed}]')

class ObstacleKalman(Obstacle):

    def __init__(self, obs_id, camera_param, window=10, width=225, height=225, alpha=80000, danger_threshold=0.05, danger_atte=0.85, kalman=True, _class=None, predict_length=3, predict_interval=100, cache_size=500, predict_step=10, ttc_clamp=20, f=240):
        super(ObstacleKalman, self).__init__(obs_id, window=window, width=width, height=height, alpha=alpha, danger_threshold=danger_threshold, danger_atte=danger_atte, _class=_class)
        self.kalman = kalman
        self.k = camera_param
        self.area_predictor = None
        self.pos_predictor = None
        self.ttc_predictor = None
        self.tra_predictor = None
        self.f = f
        self.predict_interval = predict_interval
        self.predict_length = predict_length
        self.cache_size = cache_size
        self.area_cache = collections.deque([], maxlen=cache_size)
        self.pos_cache = collections.deque([], maxlen=cache_size)
        self.ttc_cache = collections.deque([], maxlen=cache_size)
        self.tra_cache = collections.deque([], maxlen=cache_size)
        self.next_tra_cache = collections.deque([], maxlen=30)
        self.predict_step = predict_step
        self.ttc_clamp = 100
        self.start_step = 5
        self.now_step = 0
        self.stable_step = 20
        self.first_tra = True
        self.raw_ttc = []
        self.raw_area = []
        self.timestamps = []
        self.area_scale = 0.83

    def predict_trajectory(self, max_step=100):
        if self.tra_predictor and (not self.first_tra):
            trajectory = self.tra_predictor.predict_trajectory(max_step)
        else:
            trajectory = []
        return trajectory

    def smooth(self, var):
        if len(var) <= 1:
            return 0
        cur = var[-1]
        cur_time = self.timestamps[-1]
        if len(var) <= self.stable_step:
            last = var[-2]
            last_time = self.timestamps[-2]
        elif len(var) - self.stable_step < self.window:
            last = var[self.stable_step - 1]
            last_time = self.timestamps[self.stable_step - 1]
        else:
            last = var[-self.window]
            last_time = self.timestamps[-self.window]
        try:
            rs = (cur - last) / (cur_time - last_time)
        except ZeroDivisionError as e:
            print(e)
        return rs

    def update(self, value, pos, height, timestamp=None):
        if pos is None:
            return
        now_time = timestamp
        self.timestamps.append(now_time)
        if now_time is None:
            now_time = time.perf_counter()
        pre_area = value
        self.raw_area.append(value)
        if self.kalman and (not self.area_predictor):
            self.area_predictor = AreaTracker(value, 0, now_time, self.window, use_r=True, processNoiseCov=0.0001, measure_noi=0.0001, use_acc=True, scale=self.area_scale)
        else:
            self.area_predictor.update(value, now_time)
            pre_area = self.area_predictor.predict()
        pre_pos = pos
        if self.kalman and (not self.pos_predictor):
            self.pos_predictor = ObjectTrackerCV(pos, now_time)
            self.pos_predictor.predict()
        else:
            self.pos_predictor.update(pos, now_time)
            pre_pos = self.pos_predictor.predict()
        real_dir = get_3d_direction(self.k, pre_pos)
        now_rate = self.area_predictor.rate()
        now_ttc = 3 * pre_area / (now_rate if now_rate > 0.001 else 0.001)
        self.raw_ttc.append(now_ttc)
        pre_ttc = now_ttc = now_ttc if now_ttc < self.ttc_clamp else self.ttc_clamp
        if self.kalman and (not self.ttc_predictor) and (len(self.area_predictor.growth_rates) > 1) and (pre_ttc < self.ttc_clamp):
            self.ttc_predictor = AreaTracker(now_ttc, initial_timestamp=now_time, mode='no', use_r=True, measure_noi=0.05)
            self.ttc_predictor.predict()
        elif self.ttc_predictor:
            self.ttc_predictor.update(now_ttc, now_time)
            pre_ttc = self.ttc_predictor.predict()
        ttc_pos = np.array([real_dir[0] * pre_ttc, real_dir[1] * pre_ttc, real_dir[2] * pre_ttc])
        pre_ttc_pos = ttc_pos
        if self.kalman and (not self.tra_predictor) and self.ttc_predictor:
            self.tra_predictor = ThreeDimensionalTracker(ttc_pos, initial_timestamp=now_time, mode='speed')
        elif self.tra_predictor:
            self.tra_predictor.update(ttc_pos, now_time)
            pre_ttc_pos = self.tra_predictor.predict()
        self.area_cache.append(np.array(pre_area).reshape(-1, 1))
        self.pos_cache.append(np.array(pre_pos).reshape(-1, 1))
        self.ttc_cache.append(np.array(pre_ttc).reshape(-1, 1))
        self.tra_cache.append(np.array(pre_ttc_pos).reshape(-1, 1))
        next_trajectory = []
        if self.tra_predictor and (not self.first_tra):
            next_trajectory = self.predict_trajectory(self.predict_step)
        if self.tra_predictor and self.first_tra:
            self.first_tra = False
        self.next_tra_cache.append(next_trajectory)
        self.timestamp = now_time
        self.now_step += 1
        return next_trajectory

    def empty_update(self, timestamp=None):
        if len(self.area_cache) < 2:
            return []
        now_time = time.perf_counter()
        timestamp = timestamp if timestamp else self.timestamp + 1 / self.f
        if self.tra_predictor and (not self.first_tra):
            pre_ttc_pos = self.tra_predictor.predict()
        next_trajectory = self.predict_trajectory(self.predict_step)
        self.next_tra_cache.append(next_trajectory)
        return next_trajectory

    def next_trajectory(self, index=None):
        if not index:
            return self.next_tra_cache[-1]
        else:
            raise NotImplementedError('换掉deque')

class Obstacles:
    FOV_CATERCORNER = 'catercorner'
    FOV_HORIZONTAL = 'horizontal'
    MODE_AUTO_IDENTIFICATION = 'auto'
    MODE_LABOUR = 'labour'

    def __init__(self, active_time=1, window=10, danger_threshold=0.5, danger_mov=10, height=640, width=640, near_val=0.1, fov=53.12, fov_mode=FOV_CATERCORNER, reverse_threshold=0.4, mode=None, same_threshold=5, smooth=False, avoid_variation=0.02, avoid_vec_attn=0.85):
        self.obstacles = {}
        self.window = window
        self.active_time = active_time
        self.danger_threshold = danger_threshold
        self.height = height
        self.width = width
        self.near_val = near_val
        self.fov = fov
        self.fov_mode = fov_mode
        self.pic_unit = self.get_pic_unit()
        self.danger_mov = danger_mov * self.pic_unit
        self.real_height = self.height * self.pic_unit
        self.real_width = self.width * self.pic_unit
        self.reverse_threshold = reverse_threshold
        self.re_height = self.reverse_threshold * self.height * 0.5 * self.pic_unit
        self.re_width = self.reverse_threshold * self.width * 0.5 * self.pic_unit
        self.max_danger_rate = 0
        self.last_avoid_vec = np.array([0.0, 0.0, 0.0])
        self.max_avoid_variation = avoid_variation
        self.mode = mode if mode else Obstacles.MODE_AUTO_IDENTIFICATION
        self.same_threshold = same_threshold
        self.smooth = smooth
        self.avoid_vec_attn = avoid_vec_attn
        self.zero_speed_threshold = 1
        self.no_detect_start = None
        self.zero_speed = np.array([0.0, 0.0, 0.0])
        if mode == 'auto':
            self.obs_brief = {}
        self.obs_ids = 0

    def get_pic_unit(self):
        real_vision_len = 2 * self.near_val * np.tan(self.fov * 0.5 * np.pi / 180)
        if self.fov_mode == Obstacles.FOV_CATERCORNER:
            catercorner_len = (self.width ** 2 + self.height ** 2) ** 0.5
            return real_vision_len / catercorner_len
        elif self.fov_mode == Obstacles.FOV_HORIZONTAL:
            return self.width / real_vision_len

    def add_obs(self, obs_id, _class=None, t=None):
        self.obstacles[obs_id] = Obstacle(obs_id, window=self.window, width=self.width, height=self.height, _class=_class, init_time=t)

    def update_obs(self, obs_id, value, pos, height=0, _class=None, timestamp=None):
        if obs_id not in self.obstacles.keys():
            self.add_obs(obs_id, _class, timestamp)
        self.obstacles[obs_id].update(value, pos, height, timestamp)

    def auto_update_obs(self, value, pos, _class, timestamp=None, tar_id=None):
        obs_id = tar_id
        if obs_id is None:
            new_id = self.gen_id()
            self.update_obs(new_id, value, pos, _class, timestamp)
            return (new_id, True)
        self.update_obs(obs_id, value, pos, _class, timestamp)
        return (obs_id, False)

    def search_obs(self, value, pos, _class):
        min_diff = np.inf
        obs_id = -1
        old = None
        for obs in self.obstacles.values():
            t_value = obs.frames_value[-1]
            t_pos = obs.last_pos
            scale = t_value ** 0.5
            v_diff = np.abs(value - t_value) / (value + 1e-12) + np.abs(value - t_value) / (t_value + 1e-12)
            p_diff = (((t_pos[0] - pos[0]) / scale) ** 2 + ((t_pos[1] - pos[1]) / scale) ** 2) ** 0.5
            if min_diff > v_diff + p_diff:
                min_diff = v_diff + p_diff
                obs_id = obs.get_id()
                old = (t_value, t_pos)
        if min_diff < self.same_threshold:
            return (obs_id, min_diff)
        return (None, 99999)

    def diff(self, query, target):
        value_diff = np.abs((query[0] - target[0]) / target[0])
        pos_diff = np.linalg.norm(query[1] - target[1])
        q_diff = (value_diff ** 2 + pos_diff ** 2) ** 0.5
        return q_diff

    def gen_id(self):
        new_id = self.obs_ids
        self.obs_ids += 1
        return new_id

    def check_active(self, timestamp=None):
        now_stamp = time.perf_counter() if not timestamp else timestamp
        for key in list(self.obstacles.keys()):
            if now_stamp - self.obstacles[key].get_timestamp() >= self.active_time:
                self.delete_obs(key, self.obstacles[key].get_class())

    def delete_obs(self, obs_id, _class=None):
        del self.obstacles[obs_id]
        if _class is not None:
            pass

    def get_axis_vec(self, pos, threshold, screen_len, m_vec, danger_rate):
        if abs(pos) > threshold:
            vec = pos * danger_rate * (1 - abs(pos) / screen_len)
            return -vec
        else:
            vec = (pos + (screen_len - pos) / 2) * danger_rate
            return vec if m_vec * pos <= 0 and abs(m_vec) > self.danger_mov else -vec

    def all_update(self, obs_info: list, absolute_timestamp=None):
        update_list = {}
        temp_next_tra = {}
        tar_obs_id = []
        post_info = {}
        new_obs = []
        for info in obs_info:
            value, pos, _class, timestamp = info
            id_, diff = self.search_obs(value, pos, _class)
            if id_ is None:
                new_obs.append(info)
                continue
            if id_ in post_info:
                if diff < post_info[id_][0]:
                    post_info[id_] = (diff, info)
            else:
                tar_obs_id.append(id_)
                post_info[id_] = (diff, info)

        def inner_add(info_, obs_id_):
            value, pos, _class, timestamp = info_
            obs_id, is_new_obs = self.auto_update_obs(value, pos, _class, timestamp, obs_id_)
            update_list[obs_id] = True
        for obs in new_obs:
            inner_add(obs, None)
        for tar_id in tar_obs_id:
            inner_add(post_info[tar_id][1], tar_id)
        timestamp = obs_info[0][-1] if len(obs_info) > 1 else None
        for key in self.obstacles.keys():
            if key not in update_list:
                self.obstacles[key].empty_update(timestamp)
        self.check_active(absolute_timestamp)

    def get_max_danger_rate(self):
        max_danger_rate = 0.0
        for obs in self.obstacles.values():
            max_danger_rate = max_danger_rate if max_danger_rate > obs.danger_rate else obs.danger_rate
        return max_danger_rate

class ObstaclesKalman:

    def __init__(self, camera_param, active_time=0.4, window=10, danger_threshold=3, height=640, width=640, mode=None, same_threshold=2, smooth=False, avoid_variation=0.02, avoid_vec_attn=0.85):
        self.obstacles = {}
        self.camera_param = camera_param
        self.window = window
        self.active_time = active_time
        self.danger_threshold = danger_threshold
        self.height = height
        self.width = width
        self.max_danger_rate = 0
        self.last_avoid_vec = np.array([0.0, 0.0, 0.0])
        self.max_avoid_variation = avoid_variation
        self.mode = mode if mode else Obstacles.MODE_AUTO_IDENTIFICATION
        self.same_threshold = same_threshold
        self.smooth = smooth
        self.avoid_vec_attn = avoid_vec_attn
        self.zero_speed_threshold = 1
        self.no_detect_start = None
        self.zero_speed = np.array([0.0, 0.0, 0.0])
        self.next_trajectory = {}
        if mode == 'auto':
            self.obs_brief = {}
        self.obs_ids = 0

    def gen_id(self):
        new_id = self.obs_ids
        self.obs_ids += 1
        return new_id

    def add_obs(self, obs_id, _class=None):
        self.obstacles[obs_id] = ObstacleKalman(obs_id, self.camera_param, window=self.window, width=self.width, height=self.height, _class=_class)

    def update_obs(self, obs_id, value, pos, height=0, _class=None, timestamp=None):
        if obs_id not in self.obstacles.keys():
            self.add_obs(obs_id, _class)
        return self.obstacles[obs_id].update(value, pos, height, timestamp)

    def auto_update_obs(self, value, pos, _class, timestamp=None, tar_id=None):
        obs_id = tar_id
        if obs_id is None:
            new_id = self.gen_id()
            tra = self.update_obs(new_id, value, pos, _class=_class, timestamp=timestamp)
            return (new_id, True, tra)
        tra = self.update_obs(obs_id, value, pos, _class=_class, timestamp=timestamp)
        return (obs_id, False, tra)

    def check_active(self, timestamp=None):
        now_time = time.perf_counter() if not timestamp else timestamp
        delete_key = []
        for obs in self.obstacles.values():
            if now_time - obs.get_timestamp() > self.active_time:
                delete_key.append(obs.get_id())
        for key in delete_key:
            del self.obstacles[key]
            del self.next_trajectory[key]

    def get_index_predict(self, index=None):
        if index is not None:
            raise NotImplementedError('还未实现下标访问')
        now_predict = {}
        for key in self.next_trajectory.keys():
            now_predict[key] = self.next_trajectory[key][-1]
        return now_predict

    def get_all_tracker_res(self):
        all_trail = {}
        for key in self.obstacles.keys():
            all_trail[key] = self.obstacles[key].tra_cache
        return all_trail

    def update_next_tra(self, tra_dict):
        for key in tra_dict.keys():
            if key not in self.next_trajectory:
                self.next_trajectory[key] = collections.deque([tra_dict[key]], maxlen=100)
            else:
                self.next_trajectory[key].append(tra_dict[key])

    def track_all_update(self, obs_info: list, absolute_timestamp=None):
        update_list = {}
        temp_next_tra = {}
        for info in obs_info:
            value, pos, _class, timestamp, obs_id = info
            if obs_id is None:
                continue
            tra = self.update_obs(obs_id, value, pos, _class=_class, timestamp=timestamp)
            update_list[obs_id] = True
            temp_next_tra[obs_id] = tra
        timestamp = obs_info[0][-1] if len(obs_info) > 1 else None
        for key in self.obstacles.keys():
            if key not in update_list:
                tra = self.obstacles[key].empty_update(timestamp)
                temp_next_tra[key] = tra
        self.update_next_tra(temp_next_tra)
        self.check_active(absolute_timestamp)
        pass

    def all_update(self, obs_info: list, absolute_timestamp=None):
        if len(obs_info) > 0 and len(obs_info[0]) == 5:
            self.track_all_update(obs_info, absolute_timestamp)
            return
        update_list = {}
        temp_next_tra = {}
        tar_obs_id = []
        post_info = {}
        new_obs = []
        for info in obs_info:
            if isinstance(info, dict):
                info = info.values()
            value, pos, _class, timestamp = info
            id_, diff = self.search_obs(value, pos, _class)
            if id_ is None:
                new_obs.append(info)
                continue
            if id_ in post_info:
                if diff < post_info[id_][0]:
                    post_info[id_] = (diff, info)
            else:
                tar_obs_id.append(id_)
                post_info[id_] = (diff, info)

        def inner_add(info_, obs_id_):
            value, pos, _class, timestamp = info_
            obs_id, is_new_obs, tra = self.auto_update_obs(value, pos, _class, timestamp, obs_id_)
            update_list[obs_id] = True
            temp_next_tra[obs_id] = tra
        for obs in new_obs:
            inner_add(obs, None)
        for tar_id in tar_obs_id:
            inner_add(post_info[tar_id][1], tar_id)
        timestamp = obs_info[0][-1] if len(obs_info) > 1 else None
        for key in self.obstacles.keys():
            if key not in update_list:
                tra = self.obstacles[key].empty_update(timestamp)
                temp_next_tra[key] = tra
        self.update_next_tra(temp_next_tra)
        self.check_active(absolute_timestamp)

    def search_obs(self, value, pos, _class):
        min_diff = np.inf
        obs_id = -1
        old = None
        for obs in self.obstacles.values():
            t_value = obs.area_cache[-1]
            t_pos = obs.pos_cache[-1]
            scale = t_value ** 0.5
            v_diff = np.abs(value - t_value) / (value + 1e-12) + np.abs(value - t_value) / (t_value + 1e-12)
            p_diff = (((t_pos[0] - pos[0]) / scale) ** 2 + ((t_pos[1] - pos[1]) / scale) ** 2) ** 0.5
            if min_diff > v_diff + p_diff:
                min_diff = v_diff + p_diff
                obs_id = obs.get_id()
                old = (t_value, t_pos)
        if min_diff < self.same_threshold:
            return (obs_id, min_diff)
        return (None, 99999)

    def plan_method(self, trails):
        a = 1.5
        speed = 2
        dg_thr = 4
        all_new_obs = []
        sep_new_obs = []
        valid_trails = []
        for key in trails:
            if len(trails[key]) == 0:
                continue
            min_id = -1
            min_dist = 10000000000.0
            for i, p in enumerate(trails[key]):
                dist = np.linalg.norm(p)
                if dist < min_dist:
                    min_id = i
                    min_dist = dist
            if np.linalg.norm(trails[key][min_id]) > dg_thr:
                continue
            valid_trails.append(trails[key])
            obs_len = np.linalg.norm(trails[key][min_id - 1] - trails[key][0]) * speed
            obs_num = int(obs_len / (2 * a))
            obs_num = obs_num if obs_num > 2 else 2
            new_obs_ = 0
            new_obs = []
            new_obs.append(((trails[key][0][0].item() * speed - a, trails[key][0][1].item() * speed - a), (trails[key][0][0].item() * speed + a, trails[key][0][1].item() * speed + a)))
            obs_num_ = obs_num - 1
            new_obs_ += 1
            while obs_num_ > 0:
                new_obs_ += 1
                if obs_num_ == 1:
                    new_obs.append(((trails[key][min_id][0].item() * speed + a, trails[key][min_id][1].item() * speed + a), (trails[key][min_id][0].item() * speed - a, trails[key][min_id][1].item() * speed - a)))
                else:
                    n_id = int(min_id * (new_obs_ / obs_num))
                    new_obs.append(((trails[key][n_id][0].item() * speed + a, trails[key][n_id][1].item() * speed + a), (trails[key][n_id][0].item() * speed - a, trails[key][n_id][1].item() * speed - a)))
                obs_num_ -= 1
            sep_new_obs.append(new_obs)
            all_new_obs += new_obs
        if len(valid_trails) > 0:
            show_trail(valid_trails, name='predict')
        return (all_new_obs, sep_new_obs)

    def avoidance(self, origin, target, method='b-spline', a=3, static_obs=None, current_pos=None, sta_obs_size=None):
        valid_trajectory = {}
        for key in self.next_trajectory:
            if len(self.obstacles[key].tra_cache) > 20 and self.obstacles[key].area_cache[-1] > 800:
                valid_trajectory[key] = self.next_trajectory[key][-1][:20]
        if method == 'b-spline':
            avoid_trail = plan_path_bspline(origin, target, valid_trajectory.values())
        elif method in ['rrt']:
            avoid_trail = rrt(origin, target, [valid_trajectory[k] for k in valid_trajectory.keys()], a=a)
        elif method in ['repulsion']:
            avoid_trail = repulsion(origin, target, [valid_trajectory[k] for k in valid_trajectory.keys()], a=a, static_obs=static_obs, current_pos=current_pos, static_obs_size=sta_obs_size)
        elif method in ['plan']:
            return self.plan_method(valid_trajectory)
            pass
        else:
            raise NotImplementedError(f'have not implement method: {method}')
        return avoid_trail
