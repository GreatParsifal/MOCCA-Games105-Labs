##############
# 姓名：马培杰
# 学号：2300012811
##############
from graph import *
from answer_task1 import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


class CharacterController():
    def __init__(self, controller) -> None:
        self.controller = controller

        self.graph = Graph('./nodes.npy')
        self.graph.load_from_file()

        self.walk_loop_ratio = 0.5
        self.walk_loop_half_life = 0.22
        self._preprocess_loop_motions()

        self.node_names = [nd.name for nd in self.graph.nodes]
        self.node_map = {nd.name: nd for nd in self.graph.nodes}

        self.fk_cache = {}
        for nd in self.graph.nodes:
            jt, jo = nd.motion.batch_forward_kinematics()
            self.fk_cache[nd.name] = (jt, jo)

        self.cur_root_pos = None
        self.cur_root_rot = None

        self.runtime = None
        self.cur_node = None
        self.cur_edge = None
        self.cur_frame = 0
        self.cur_end_frame = -1
        self.root_floor_y = None

        self.in_transition = False
        self.trans_prev = None
        self.trans_next = None
        self.trans_frame = 0
        self.trans_len = 10
        self.trans_alpha_pow = 1.0

        self.switch_cooldown = 9
        self.frames_since_switch = 999
        self.turn_lock_frames = 0
        self.turn_lock_max_frames = 50

        self.idle_speed_eps = 0.08
        self.idle_yaw_vel_eps = 0.12

        self.min_trans_len = 12
        self.max_trans_len = 20
        self.turn_to_walk_extra_len = 6
        self.turn_to_walk_max_len = 28
        self.walk_to_turn_extra_len = 7
        self.walk_to_turn_max_len = 30
        self.walk_to_turn_alpha_pow = 1.35
        self.turn_start_skip_frames = 2
        self.spin_start_skip_frames = 5
        self.match_root_weight = 1.2
        self.match_rot_weight = 0.35
        self.match_joint_weight = 1.0
        self.match_root_vel_weight = 0.85
        self.match_joint_vel_weight = 0.28

        self.lookahead_mode_idx = 2
        self.lookahead_steer_idx = 1
        self.lookahead_trans_idx = 2

        self.spin_enter_yaw_err = np.pi * 0.5
        self.spin_enter_yaw_vel = 1.2
        self.spin_reverse_yaw_err = 2.2
        self.spin_reverse_speed_max = 0.8
        self.turn_enter_yaw_err = 0.55
        self.turn_release_yaw_err = 0.30
        self.yaw_near_pi_eps = 0.22
        self.turn_cross_eps = 0.05

        self.target_yaw_filter_alpha = 0.2
        self.smoothed_target_yaw = None
        
        

        self.initialize()

    def _preprocess_loop_motions(self):
        for nd in self.graph.nodes:
            if nd.name.lower() == 'walk.bvh':
                nd.motion = build_loop_motion(
                    nd.motion,
                    ratio=self.walk_loop_ratio,
                    half_life=self.walk_loop_half_life
                )

    def _safe_quat(self, quat):
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8 or (not np.isfinite(n)):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    def _smootherstep(self, x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    def _slerp_quat(self, q1, q2, a):
        key_times = np.array([0.0, 1.0], dtype=np.float64)
        q1 = self._safe_quat(q1)
        q2 = self._safe_quat(q2)
        key_rots = R.from_quat(np.stack([q1, q2], axis=0))
        slerp = Slerp(key_times, key_rots)
        return slerp([a]).as_quat()[0]

    def _extract_yaw_rotation(self, quat):
        q = self._safe_quat(quat)
        fwd = R.from_quat(q).apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        fwd_xz = np.array([fwd[0], fwd[2]], dtype=np.float64)
        n = np.linalg.norm(fwd_xz)
        if n < 1e-8:
            return R.from_quat(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
        fwd_xz /= n
        yaw = np.arctan2(fwd_xz[0], fwd_xz[1])
        return R.from_rotvec(yaw * np.array([0.0, 1.0, 0.0], dtype=np.float64))

    def _signed_yaw_delta(self, cur_yaw: R, target_yaw: R):
        f_cur = cur_yaw.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        f_tar = target_yaw.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))

        a = np.array([f_cur[0], f_cur[2]], dtype=np.float64)
        b = np.array([f_tar[0], f_tar[2]], dtype=np.float64)

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0

        a /= na
        b /= nb
        cross = a[0] * b[1] - a[1] * b[0]
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.arctan2(cross, dot)

    def _update_smoothed_target_yaw(self, desired_rot_list):
        idx = int(np.clip(self.lookahead_mode_idx, 0, len(desired_rot_list) - 1))
        raw_target_yaw = self._extract_yaw_rotation(desired_rot_list[idx])

        if self.smoothed_target_yaw is None:
            self.smoothed_target_yaw = raw_target_yaw
            return self.smoothed_target_yaw

        q_prev = self.smoothed_target_yaw.as_quat()
        q_raw = raw_target_yaw.as_quat()
        self.smoothed_target_yaw = R.from_quat(
            self._slerp_quat(q_prev, q_raw, self.target_yaw_filter_alpha)
        )
        return self.smoothed_target_yaw

    def initialize(self):
        self.cur_node = self.graph.nodes[0]
        self.cur_edge = None
        self.cur_frame = 0
        self.cur_end_frame = self.cur_node.motion.motion_length

        jt0, jo0 = self.fk_cache[self.cur_node.name]
        root_pos0 = jt0[0, 0].copy()
        root_yaw0 = self._extract_yaw_rotation(jo0[0, 0])

        self.runtime = {
            'node': self.cur_node,
            'frame': 0,
            'anchor_frame': 0,
            'anchor_pos': root_pos0,
            'anchor_yaw': root_yaw0
        }

        _, jt, jo = self._sample_runtime_pose(self.runtime)
        self.cur_root_pos = jt[0].copy()
        self.cur_root_rot = jo[0].copy()
        self.root_floor_y = float(self.cur_root_pos[1])

    def _apply_floor_constraint(self, joint_translation):
        if self.root_floor_y is None:
            return joint_translation

        jt = joint_translation.copy()
        lift = float(self.root_floor_y - jt[0, 1])
        if abs(lift) > 1e-8:
            jt[:, 1] += lift
        return jt

    def _sample_runtime_pose(self, runtime):
        nd = runtime['node']
        frame = int(runtime['frame'])
        anchor_frame = int(runtime['anchor_frame'])
        anchor_pos = runtime['anchor_pos']
        anchor_yaw = runtime['anchor_yaw']

        jt_all, jo_all = self.fk_cache[nd.name]
        n = jt_all.shape[0]
        frame = frame % n
        anchor_frame = anchor_frame % n

        jt = jt_all[frame].copy()
        jo = jo_all[frame].copy()

        jt_anchor = jt_all[anchor_frame, 0]
        delta = jt - jt_anchor.reshape(1, 3)
        jt_world = anchor_yaw.apply(delta) + anchor_pos.reshape(1, 3)
        jo_world = (anchor_yaw * R.from_quat(jo)).as_quat()

        joint_name = nd.motion.joint_name
        return joint_name, jt_world, jo_world

    def _step_runtime(self, runtime, root_pos, root_rot):
        nd = runtime['node']
        n = nd.motion.motion_length
        frame = int(runtime['frame'])

        if frame >= n - 1:
            runtime['anchor_pos'] = root_pos.copy()
            runtime['anchor_yaw'] = self._extract_yaw_rotation(root_rot)
            runtime['anchor_frame'] = 0
            runtime['frame'] = 0
        else:
            runtime['frame'] = frame + 1

    def _pose_match_cost(self, jt_a, jo_a, jt_b, jo_b):
        root_pos_cost = np.linalg.norm(jt_a[0, [0, 2]] - jt_b[0, [0, 2]])
        root_yaw_a = self._extract_yaw_rotation(jo_a[0])
        root_yaw_b = self._extract_yaw_rotation(jo_b[0])
        root_rot_cost = abs(self._signed_yaw_delta(root_yaw_a, root_yaw_b))

        joint_pos_cost = np.mean(np.linalg.norm(jt_a - jt_b, axis=1))

        return (
            self.match_root_weight * root_pos_cost +
            self.match_rot_weight * root_rot_cost +
            self.match_joint_weight * joint_pos_cost
        )

    def _pose_match_cost_with_velocity(self, jt_a, jo_a, vel_a, jt_b, jo_b, vel_b):
        base_cost = self._pose_match_cost(jt_a, jo_a, jt_b, jo_b)
        root_vel_cost = np.linalg.norm(vel_a[0, [0, 2]] - vel_b[0, [0, 2]])
        joint_vel_cost = np.mean(np.linalg.norm(vel_a - vel_b, axis=1))

        return (
            base_cost +
            self.match_root_vel_weight * root_vel_cost +
            self.match_joint_vel_weight * joint_vel_cost
        )

    def _runtime_pose_velocity(self, runtime, frame):
        nd = runtime['node']
        n = nd.motion.motion_length
        f0 = int(frame) % n
        f1 = (f0 + 1) % n

        rt0 = {
            'node': nd,
            'frame': f0,
            'anchor_frame': int(runtime['anchor_frame']),
            'anchor_pos': runtime['anchor_pos'],
            'anchor_yaw': runtime['anchor_yaw']
        }
        rt1 = {
            'node': nd,
            'frame': f1,
            'anchor_frame': int(runtime['anchor_frame']),
            'anchor_pos': runtime['anchor_pos'],
            'anchor_yaw': runtime['anchor_yaw']
        }

        _, jt0, _ = self._sample_runtime_pose(rt0)
        _, jt1, _ = self._sample_runtime_pose(rt1)
        return jt1 - jt0

    def _find_best_start_frame(self, target_node, cur_jt, cur_jo, cur_runtime, min_frame=0):
        jt_all, _ = self.fk_cache[target_node.name]
        n = jt_all.shape[0]
        if n <= 1:
            return 0

        min_frame = int(np.clip(min_frame, 0, n - 1))

        anchor_pos = cur_jt[0].copy()
        anchor_yaw = self._extract_yaw_rotation(cur_jo[0])

        best_f = min_frame
        best_cost = 1e18
        cur_vel = self._runtime_pose_velocity(cur_runtime, int(cur_runtime['frame']))
        for f in range(min_frame, n):
            probe_rt = {
                'node': target_node,
                'frame': f,
                'anchor_frame': f,
                'anchor_pos': anchor_pos,
                'anchor_yaw': anchor_yaw
            }
            _, jt_probe, jo_probe = self._sample_runtime_pose(probe_rt)
            probe_vel = self._runtime_pose_velocity(probe_rt, f)
            c = self._pose_match_cost_with_velocity(cur_jt, cur_jo, cur_vel, jt_probe, jo_probe, probe_vel)
            if c < best_cost:
                best_cost = c
                best_f = f

        return best_f

    def _steer_runtime_yaw(self, desired_rot_list, desired_vel_list):
        if self.runtime is None:
            return
        if self.runtime['node'].name != 'walk.bvh':
            return

        idx = int(np.clip(self.lookahead_steer_idx, 0, len(desired_vel_list) - 1))
        speed = np.linalg.norm(desired_vel_list[idx][[0, 2]])
        if speed < 0.12:
            return

        cur_q = self.runtime['anchor_yaw'].as_quat()
        tar_q = self._extract_yaw_rotation(desired_rot_list[idx]).as_quat()
        alpha = float(np.clip(0.06 + 0.12 * speed, 0.06, 0.24))
        self.runtime['anchor_yaw'] = R.from_quat(self._slerp_quat(cur_q, tar_q, alpha))

    def _choose_spin_mode(self, yaw_err, yaw_vel):
        return 'spin_counter_clockwise'

    def _choose_turn_mode(self, yaw_err, yaw_vel):
        sign = self._resolve_turn_sign(yaw_err, yaw_vel)
        return 'turn_left' if sign < 0.0 else 'turn_right'

    def _resolve_turn_sign(self, yaw_err, yaw_vel):
        sign = yaw_err
        if abs(sign) < 1e-4 or abs(abs(sign) - np.pi) < self.yaw_near_pi_eps:
            sign = yaw_vel
        if abs(sign) < 1e-4:
            sign = yaw_err
        return sign

    def _turn_mode_from_node_name(self, node_name):
        name = node_name.lower()
        if 'turn_left' in name or 'spin_counter_clockwise' in name:
            return 'turn_left'
        if 'turn_right' in name or 'spin_clockwise' in name:
            return 'turn_right'
        return None

    def _turn_mode_sign(self, turn_mode):
        if turn_mode == 'turn_left':
            return -1.0
        if turn_mode == 'turn_right':
            return 1.0
        return 0.0

    def _should_hold_turn(self, turn_mode, yaw_err):
        turn_sign = self._turn_mode_sign(turn_mode)
        if abs(turn_sign) < 0.5:
            return False

        aligned_err = float(yaw_err) * turn_sign
        if aligned_err <= self.turn_cross_eps:
            return False

        return abs(float(yaw_err)) > self.turn_release_yaw_err

    def _is_idle_input(self, desired_vel_list, desired_avel_list):
        speed = np.linalg.norm(desired_vel_list[0][[0, 2]])
        yaw_vel = abs(float(desired_avel_list[0, 1]))
        return speed < self.idle_speed_eps and yaw_vel < self.idle_yaw_vel_eps

    def _desired_mode(self, desired_rot_list, desired_vel_list, desired_avel_list):
        idx = int(np.clip(self.lookahead_mode_idx, 0, len(desired_vel_list) - 1))
        v_now = desired_vel_list[idx]
        avel_now = desired_avel_list[idx]

        speed = np.linalg.norm(v_now[[0, 2]])
        yaw_vel = float(avel_now[1])

        cur_yaw = self._extract_yaw_rotation(self.cur_root_rot)
        raw_target_yaw = self._extract_yaw_rotation(desired_rot_list[idx])
        target_yaw = self.smoothed_target_yaw
        if target_yaw is None:
            target_yaw = self._update_smoothed_target_yaw(desired_rot_list)
        yaw_err = self._signed_yaw_delta(cur_yaw, target_yaw)
        raw_yaw_err = self._signed_yaw_delta(cur_yaw, raw_target_yaw)

        abs_yaw_vel = abs(yaw_vel)
        abs_yaw_err = abs(yaw_err)
        abs_raw_yaw_err = abs(raw_yaw_err)

        if speed < self.idle_speed_eps and abs_yaw_vel < self.idle_yaw_vel_eps:
            return 'stay'

        current_turn_mode = self._turn_mode_from_node_name(self.cur_node.name)
        if current_turn_mode is not None and self._should_hold_turn(current_turn_mode, yaw_err):
            return current_turn_mode

        if abs_raw_yaw_err > self.spin_reverse_yaw_err and (
            speed < self.spin_reverse_speed_max or abs_yaw_vel > 0.6
        ):
            return self._choose_spin_mode(yaw_err, yaw_vel)

        if speed < 0.18 and (abs_yaw_vel > self.spin_enter_yaw_vel or abs_raw_yaw_err > self.spin_enter_yaw_err):
            return self._choose_spin_mode(yaw_err, yaw_vel)

        if abs_yaw_vel > 0.40 or abs_yaw_err > self.turn_enter_yaw_err:
            return self._choose_turn_mode(yaw_err, yaw_vel)

        if speed > 0.10:
            return 'walk'

        return 'stay'

    def _edge_score(self, edge, mode):
        name = edge.destination.name.lower()
        score = 0.0

        if mode == 'stay':
            if edge.destination == self.cur_node:
                score += 50.0
            return score

        if mode == 'walk':
            if 'walk' in name:
                score += 100.0
            elif 'turn' in name or 'spin' in name:
                score += 10.0
            else:
                score += 0.0

        elif mode == 'turn_left':
            if 'turn_left' in name:
                score += 100.0
            elif 'spin_counter_clockwise' in name:
                score += 85.0
            elif 'turn' in name:
                score += 40.0

        elif mode == 'turn_right':
            if 'turn_right' in name:
                score += 100.0
            elif 'spin_clockwise' in name:
                score += 85.0
            elif 'turn' in name:
                score += 40.0

        elif mode == 'spin_clockwise':
            if 'spin_clockwise' in name:
                score += 100.0
            elif 'turn_right' in name:
                score += 75.0
            elif 'spin' in name:
                score += 50.0

        elif mode == 'spin_counter_clockwise':
            if 'spin_counter_clockwise' in name:
                score += 100.0
            elif 'turn_left' in name:
                score += 75.0
            elif 'spin' in name:
                score += 50.0

        if edge.destination == self.cur_node and mode != 'stay':
            score -= 10.0

        return score

    def _pick_target_edge(self, desired_rot_list, desired_vel_list, desired_avel_list):
        if self.cur_node is None or len(self.cur_node.edges) == 0:
            return None

        mode = self._desired_mode(desired_rot_list, desired_vel_list, desired_avel_list)

        best_edge = None
        best_score = -1e18
        for edge in self.cur_node.edges:
            score = self._edge_score(edge, mode)
            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def _start_transition(self, target_node, desired_rot_list):
        _, cur_jt, cur_jo = self._sample_runtime_pose(self.runtime)
        cur_root_pos = cur_jt[0].copy()
        cur_root_rot = cur_jo[0].copy()
        self.cur_root_pos = cur_root_pos
        self.cur_root_rot = cur_root_rot

        idx = int(np.clip(self.lookahead_trans_idx, 0, len(desired_rot_list) - 1))
        target_yaw = self._extract_yaw_rotation(desired_rot_list[idx])
        cur_yaw = self._extract_yaw_rotation(cur_root_rot)
        yaw_gap = abs(self._signed_yaw_delta(cur_yaw, target_yaw))

        prev_name = self.runtime['node'].name.lower()
        next_name = target_node.name.lower()
        turn_to_walk = ('walk' in next_name) and (('turn' in prev_name) or ('spin' in prev_name))
        walk_to_turn = ('walk' in prev_name) and (('turn' in next_name) or ('spin' in next_name))

        start_skip = 0
        if 'spin' in next_name:
            start_skip = self.spin_start_skip_frames
        elif 'turn' in next_name:
            start_skip = self.turn_start_skip_frames
        start_frame = self._find_best_start_frame(target_node, cur_jt, cur_jo, self.runtime, min_frame=start_skip)

        self.in_transition = True
        self.trans_frame = 0
        self.trans_alpha_pow = self.walk_to_turn_alpha_pow if walk_to_turn else 1.0
        base_len = int(np.clip(
            self.min_trans_len + (self.max_trans_len - self.min_trans_len) * (yaw_gap / np.pi),
            self.min_trans_len,
            self.max_trans_len
        ))
        if turn_to_walk:
            self.trans_len = int(np.clip(
                base_len + self.turn_to_walk_extra_len,
                self.min_trans_len,
                self.turn_to_walk_max_len
            ))
        elif walk_to_turn:
            self.trans_len = int(np.clip(
                base_len + self.walk_to_turn_extra_len,
                self.min_trans_len,
                self.walk_to_turn_max_len
            ))
        else:
            self.trans_len = base_len

        self.trans_prev = {
            'node': self.runtime['node'],
            'frame': int(self.runtime['frame']),
            'anchor_frame': int(self.runtime['anchor_frame']),
            'anchor_pos': self.runtime['anchor_pos'].copy(),
            'anchor_yaw': self.runtime['anchor_yaw']
        }

        self.trans_next = {
            'node': target_node,
            'frame': start_frame,
            'anchor_frame': start_frame,
            'anchor_pos': cur_root_pos,
            'anchor_yaw': self._extract_yaw_rotation(cur_root_rot)
        }

    def update_state(self,
                     desired_pos_list,
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list):
        self._update_smoothed_target_yaw(desired_rot_list)

        if (not self.in_transition) and self._is_idle_input(desired_vel_list, desired_avel_list):
            joint_name, joint_translation, joint_orientation = self._sample_runtime_pose(self.runtime)
            joint_translation = self._apply_floor_constraint(joint_translation)
            self.cur_root_pos = joint_translation[0].copy()
            self.cur_root_rot = joint_orientation[0].copy()
            self.cur_frame = int(self.runtime['frame']) if self.runtime is not None else 0
            self.cur_end_frame = self.cur_node.motion.motion_length
            return joint_name, joint_translation, joint_orientation

        self._steer_runtime_yaw(desired_rot_list, desired_vel_list)

        target_edge = self._pick_target_edge(desired_rot_list, desired_vel_list, desired_avel_list)
        target_node = target_edge.destination if target_edge is not None else self.cur_node
        self.cur_edge = target_edge

        can_switch = self.frames_since_switch >= self.switch_cooldown

        cur_yaw = self._extract_yaw_rotation(self.cur_root_rot)
        target_yaw = self.smoothed_target_yaw if self.smoothed_target_yaw is not None else cur_yaw
        yaw_err = self._signed_yaw_delta(cur_yaw, target_yaw)
        current_turn_mode = self._turn_mode_from_node_name(self.cur_node.name)
        turning_now = current_turn_mode is not None

        if (not self.in_transition) and turning_now:
            self.turn_lock_frames += 1
        else:
            self.turn_lock_frames = 0

        if (turning_now and
            self._should_hold_turn(current_turn_mode, yaw_err) and
            self.turn_lock_frames < self.turn_lock_max_frames):
            can_switch = False
        if (not self.in_transition and
            target_node.name != self.cur_node.name and
            can_switch):
            self._start_transition(target_node, desired_rot_list)
            self.frames_since_switch = 0

        if self.in_transition:
            next_frame = int(self.trans_next['frame'])
            joint_name_a, jt_a, jo_a = self._sample_runtime_pose(self.trans_prev)
            joint_name_b, jt_b, jo_b = self._sample_runtime_pose(self.trans_next)

            M = jt_a.shape[0]
            t = self.trans_frame / max(self.trans_len - 1, 1)
            if self.trans_alpha_pow != 1.0:
                t = float(np.clip(t, 0.0, 1.0)) ** self.trans_alpha_pow
            a = self._smootherstep(t)

            joint_name = joint_name_a
            joint_translation = (1.0 - a) * jt_a + a * jt_b
            joint_translation = self._apply_floor_constraint(joint_translation)

            joint_orientation = np.zeros_like(jo_a)
            for j in range(M):
                joint_orientation[j] = self._slerp_quat(jo_a[j], jo_b[j], a)

            will_finish = (self.trans_frame >= self.trans_len - 1)
            if will_finish:
                self.runtime = {
                    'node': self.trans_next['node'],
                    'frame': next_frame,
                    'anchor_frame': next_frame,
                    'anchor_pos': joint_translation[0].copy(),
                    'anchor_yaw': self.trans_next['anchor_yaw']
                }
                self.cur_node = self.runtime['node']
                self.in_transition = False
                self.trans_prev = None
                self.trans_next = None
                self.cur_edge = None
                self.turn_lock_frames = 0
                self.trans_alpha_pow = 1.0
            else:
                self._step_runtime(self.trans_prev, jt_a[0], jo_a[0])
                self._step_runtime(self.trans_next, jt_b[0], jo_b[0])
                self.trans_frame += 1
        else:
            joint_name, joint_translation, joint_orientation = self._sample_runtime_pose(self.runtime)
            joint_translation = self._apply_floor_constraint(joint_translation)

            root_pos = joint_translation[0].copy()
            root_rot = joint_orientation[0].copy()
            self._step_runtime(self.runtime, root_pos, root_rot)

            self.cur_node = self.runtime['node']
            self.frames_since_switch += 1

        self.cur_root_pos = joint_translation[0].copy()
        self.cur_root_rot = joint_orientation[0].copy()

        self.cur_frame = int(self.runtime['frame']) if self.runtime is not None else 0
        self.cur_end_frame = self.cur_node.motion.motion_length

        return joint_name, joint_translation, joint_orientation