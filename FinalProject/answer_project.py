from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from bvh_loader import BVHMotion
from graph import Graph
from physics_warpper import PhysicsInfo
from smooth_utils import build_loop_motion


class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None

    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass

    def apply_static_torque(self):
        pass


class CharacterController:
    def __init__(self, viewer, controller, pd_controller) -> None:
        self.viewer = viewer
        self.controller = controller
        self.pd_controller = pd_controller
        self.dt = float(getattr(self.controller, "dt", 1.0 / 60.0))

        self.graph = Graph.build_demo_motion_graph()
        self.node_alias = {
            "idle": "idle.bvh",
            "walk": "walk.bvh",
            "run": "run.bvh",
            "jump": "jump.bvh",
            "turn_left": "turn_left.bvh",
            "turn_right": "turn_right.bvh",
            "dance": "excep_motion/dance.bvh",
            "kick": "excep_motion/kick.bvh",
            "backflip": "excep_motion/backflip.bvh",
        }
        self.loopable_nodes = {
            self.node_alias["idle"],
            self.node_alias["walk"],
            self.node_alias["run"],
        }
        self.jump_trim_range = (90, 158)
        self.root_locked_action_nodes = {
            self.node_alias["dance"],
            self.node_alias["kick"],
            self.node_alias["backflip"],
        }
        self.upper_body_action_nodes = {
            "upper_dance": self.node_alias["dance"],
        }
        self.animated_root_orientation_nodes = {
            self.node_alias["backflip"],
        }
        self.special_action_order = (
            "dance",
            "kick",
            "backflip",
        )
        self.upper_body_action_order = (
            "upper_dance",
        )
        idle_node = self.graph.get_node(self.node_alias["idle"])
        self.skeleton_template_motion = idle_node.motion.raw_copy()

        self.motions = []
        self.motion_cache = {}
        for node in self.graph.nodes:
            motion = self._preprocess_motion(node.name, node.motion)
            node.motion = motion
            motion.adjust_joint_name(self.viewer.joint_name)
            self.motions.append(motion)
            self.motion_cache[node.name] = {
                "motion": motion,
                "joint_name": motion.joint_name,
            }

        self.match_joint_indices = [idx for idx, name in enumerate(self.viewer.joint_name) if name != "RootJoint"]
        self._build_motion_features()
        self.upper_body_mask = self._build_upper_body_mask()

        self.cur_root_pos = np.zeros(3, dtype=np.float64)
        self.cur_root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        self.current_node_name = self.node_alias["idle"]
        self.current_frame = 0
        self.pending_locked_action = None
        self.transition = None
        self.active_motion_name = self.current_node_name
        self.active_motion_instance = self.motion_cache[self.current_node_name]["motion"]
        self.active_motion_start_frame = 0
        self.last_output_root_pos = self.active_motion_instance.joint_position[0, 0].copy()
        self.last_desired_root_rot = self.cur_root_rot.copy()
        self.last_output_root_rot = self.active_motion_instance.joint_rotation[0, 0].copy()
        self.root_pos = self.last_output_root_pos.copy()
        self.root_yaw = self._quat_yaw(self.last_output_root_rot)
        self.move_speed = 0.0
        self.target_yaw = self.root_yaw
        self.turn_visual_weight = 0.0
        self.turn_visual_active = None
        self.turn_visual_frame = {
            "left": 0,
            "right": 0,
        }
        self.root_initialized = False
        self.prev_sampled_root_pos = None
        self.prev_sampled_root_rot = None
        self.frame_counter = 0
        self.frames_since_motion_switch = 10 ** 6
        self.jump_world_xz_anchor = None
        self.jump_air_velocity = np.zeros(2, dtype=np.float64)
        self.special_action_interrupt_armed = True
        self.upper_body_action_name = None
        self.upper_body_motion_name = None
        self.upper_body_motion_instance = None
        self.upper_body_frame = 0

        self.idle_speed_threshold = 0.15
        self.run_speed_threshold = 1.2
        self.transition_blend_frames = 8
        self.gait_transition_blend_frames = 14
        self.jump_landing_blend_frames = 16
        self.turn_entry_search_frames = 24
        self.move_speed_sharpness = 10.0
        self.turn_yaw_sharpness = 12.0
        self.jump_air_control_sharpness = 2.5
        self.jump_landing_control_sharpness = 6.0
        self.jump_takeoff_speed_scale = 1.0
        self.turn_visual_enter_angle = np.deg2rad(15.0)
        self.turn_visual_full_angle = np.deg2rad(45.0)
        self.turn_visual_activation_weight = 0.05
        self.min_locomotion_residency_frames = 10
        self.phase_match_search_radius = 6
        self.upper_body_overlay_fade_frames = 10

    def _preprocess_motion(self, motion_name: str, motion: BVHMotion) -> BVHMotion:
        processed_motion = motion.raw_copy()
        if motion_name in (self.node_alias["walk"], self.node_alias["run"]):
            processed_motion = build_loop_motion(processed_motion)
        elif motion_name == self.node_alias["jump"]:
            start_frame, end_frame = self.jump_trim_range
            processed_motion = processed_motion.sub_sequence(start_frame, end_frame)
        elif motion_name in self.root_locked_action_nodes:
            processed_motion = self._retarget_special_motion(processed_motion)
        processed_motion.num_frames = processed_motion.joint_position.shape[0]
        return processed_motion

    def _retarget_special_motion(self, source_motion: BVHMotion) -> BVHMotion:
        target_motion = self.skeleton_template_motion.raw_copy()
        num_frames = source_motion.joint_position.shape[0]

        target_motion.joint_position = np.repeat(
            target_motion.joint_position[:1].copy(),
            num_frames,
            axis=0,
        )
        target_motion.joint_rotation = np.repeat(
            target_motion.joint_rotation[:1].copy(),
            num_frames,
            axis=0,
        )

        joint_map = {
            "RootJoint": "Hips",
            "lHip": "LeftUpLeg",
            "lKnee": "LeftLeg",
            "lAnkle": "LeftFoot",
            "lToeJoint": "LeftToeBase",
            "rHip": "RightUpLeg",
            "rKnee": "RightLeg",
            "rAnkle": "RightFoot",
            "rToeJoint": "RightToeBase",
            "pelvis_lowerback": "Spine",
            "lowerback_torso": "Spine1",
            "torso_head": "Head",
            "lTorso_Clavicle": "LeftShoulder",
            "lShoulder": "LeftArm",
            "lElbow": "LeftForeArm",
            "lWrist": "LeftHand",
            "rTorso_Clavicle": "RightShoulder",
            "rShoulder": "RightArm",
            "rElbow": "RightForeArm",
            "rWrist": "RightHand",
        }

        target_names = target_motion.joint_name
        source_names = source_motion.joint_name
        source_is_z_up = self._special_motion_is_z_up(source_motion)
        coord_rot = (
            R.from_euler("X", -90.0, degrees=True)
            if source_is_z_up
            else R.identity()
        )

        target_lhip = target_motion.joint_position[0, target_names.index("lHip"), 0]
        source_lhip = source_motion.joint_position[0, source_names.index("LeftUpLeg"), 0]
        scale = 1.0 if abs(source_lhip) < 1e-8 else float(target_lhip / source_lhip)

        for target_name, source_name in joint_map.items():
            target_idx = target_names.index(target_name)
            source_idx = source_names.index(source_name)
            source_rot = R.from_quat(source_motion.joint_rotation[:, source_idx, :])
            target_motion.joint_rotation[:, target_idx, :] = (
                coord_rot * source_rot * coord_rot.inv()
            ).as_quat()
            source_pos = source_motion.joint_position[:, source_idx, :]
            target_motion.joint_position[:, target_idx, :] = coord_rot.apply(source_pos) * scale

        target_motion.num_frames = num_frames
        return target_motion

    def _special_motion_is_z_up(self, source_motion: BVHMotion) -> bool:
        if "LeftLeg" not in source_motion.joint_name:
            return False
        left_leg_idx = source_motion.joint_name.index("LeftLeg")
        left_leg_offset = source_motion.joint_position[0, left_leg_idx]
        return abs(left_leg_offset[2]) > abs(left_leg_offset[1]) * 2.0

    def _build_motion_features(self):
        for node_name, cache in self.motion_cache.items():
            motion = cache["motion"]
            joint_translation, joint_orientation = motion.batch_forward_kinematics()
            relative_translation = joint_translation - joint_translation[:, :1, :]
            forward = R.from_quat(joint_orientation[:, 0, :]).apply(
                np.tile(np.array([[0.0, 0.0, 1.0]]), (motion.num_frames, 1))
            )
            root_velocity = np.zeros((motion.num_frames, 2), dtype=np.float64)
            root_velocity[1:] = motion.joint_position[1:, 0, [0, 2]] - motion.joint_position[:-1, 0, [0, 2]]
            cache["relative_translation"] = relative_translation
            cache["forward"] = forward[:, [0, 2]]
            cache["root_velocity"] = root_velocity
            cache["mean_speed"] = float(np.mean(np.linalg.norm(root_velocity, axis=1)) / max(self.dt, 1e-8))

    def _build_upper_body_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.viewer.joint_name), dtype=np.float64)
        for idx, joint_name in enumerate(self.viewer.joint_name):
            name = joint_name.lower()
            if "root" in name:
                weight = 0.0
            elif "pelvis" in name or "hip" in name:
                weight = 0.1
            elif "spine" in name:
                weight = 0.45
            elif "chest" in name or "upperchest" in name:
                weight = 0.8
            elif "neck" in name or "head" in name:
                weight = 1.0
            elif "shoulder" in name or "collar" in name or "clavicle" in name:
                weight = 0.7
            elif "arm" in name and "forearm" not in name:
                weight = 0.35
            elif "forearm" in name or "elbow" in name:
                weight = 0.2
            elif "hand" in name or "finger" in name or "thumb" in name:
                weight = 0.1
            elif any(token in name for token in ("upleg", "leg", "knee", "ankle", "foot", "toe")):
                weight = 0.0
            else:
                weight = 0.0
            mask[idx] = weight
        mask[0] = 0.0
        return mask

    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _shortest_angle_diff(self, target: float, current: float) -> float:
        return self._wrap_angle(target - current)

    def _damp_scalar(self, current: float, target: float, sharpness: float) -> float:
        alpha = 1.0 - np.exp(-sharpness * self.dt)
        return float(current + alpha * (target - current))

    def _damp_angle(self, current: float, target: float, sharpness: float) -> float:
        alpha = 1.0 - np.exp(-sharpness * self.dt)
        return self._wrap_angle(current + alpha * self._shortest_angle_diff(target, current))

    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        return R.from_rotvec(np.array([0.0, yaw, 0.0], dtype=np.float64)).as_quat()

    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        if edge1 <= edge0:
            return float(x >= edge1)
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))

    def _facing_direction_xz(self, root_rot: np.ndarray) -> np.ndarray:
        facing = R.from_quat(root_rot).apply(np.array([0.0, 0.0, 1.0]))
        facing_xz = facing[[0, 2]]
        norm = np.linalg.norm(facing_xz)
        if norm < 1e-8:
            return np.array([0.0, 1.0], dtype=np.float64)
        return facing_xz / norm

    def _align_motion_to_world(self, motion: BVHMotion, frame_id: int, root_pos: np.ndarray, root_rot: np.ndarray) -> BVHMotion:
        aligned_motion = motion.raw_copy()
        frame_id = frame_id % motion.num_frames

        target_translation_xz = root_pos[[0, 2]]
        current_translation_xz = aligned_motion.joint_position[frame_id, 0, [0, 2]].copy()
        aligned_motion.joint_position[:, 0, [0, 2]] += target_translation_xz - current_translation_xz

        target_facing_xz = self._facing_direction_xz(root_rot)
        current_root_rot = aligned_motion.joint_rotation[frame_id, 0]
        current_facing = R.from_quat(current_root_rot).apply(np.array([0.0, 0.0, 1.0]))
        current_facing_xz = current_facing[[0, 2]]
        current_norm = np.linalg.norm(current_facing_xz)
        if current_norm < 1e-8:
            current_facing_xz = np.array([0.0, 1.0], dtype=np.float64)
        else:
            current_facing_xz = current_facing_xz / current_norm

        target_yaw = float(np.arctan2(target_facing_xz[0], target_facing_xz[1]))
        current_yaw = float(np.arctan2(current_facing_xz[0], current_facing_xz[1]))
        yaw_delta = self._shortest_angle_diff(target_yaw, current_yaw)
        yaw_rot = R.from_rotvec(np.array([0.0, yaw_delta, 0.0], dtype=np.float64))

        root_rotations = R.from_quat(aligned_motion.joint_rotation[:, 0, :])
        aligned_motion.joint_rotation[:, 0, :] = (yaw_rot * root_rotations).as_quat()

        root_positions = aligned_motion.joint_position[:, 0, :].copy()
        pivot = root_positions[frame_id].copy()
        rel = root_positions - pivot
        aligned_motion.joint_position[:, 0, :] = yaw_rot.apply(rel) + pivot
        return aligned_motion

    def _make_aligned_motion(self, motion_name: str, frame_id: int, root_pos: np.ndarray, root_rot: np.ndarray) -> BVHMotion:
        motion = self.motion_cache[motion_name]["motion"]
        frame_id = frame_id % motion.num_frames
        return self._align_motion_to_world(motion, frame_id, root_pos, root_rot)

    def _start_motion_instance(self, motion_name: str, start_frame: int, root_pos: np.ndarray, root_rot: np.ndarray):
        self.active_motion_name = motion_name
        self.active_motion_start_frame = start_frame % self.motion_cache[motion_name]["motion"].num_frames
        self.active_motion_instance = self._make_aligned_motion(
            motion_name,
            self.active_motion_start_frame,
            root_pos,
            root_rot,
        )
        self.prev_sampled_root_pos = None
        self.prev_sampled_root_rot = None

    def _sample_motion_pose(self, motion_name: str, motion_instance: BVHMotion, frame_id: int):
        motion = self.motion_cache[motion_name]["motion"]
        frame_id = frame_id % motion.num_frames
        joint_translation, joint_orientation = motion_instance.batch_forward_kinematics(frame_id_list=[frame_id])
        return joint_translation[0], joint_orientation[0], frame_id

    def _quat_slerp(self, quat_a: np.ndarray, quat_b: np.ndarray, t: float) -> np.ndarray:
        quat_a = quat_a / np.linalg.norm(quat_a, axis=1, keepdims=True)
        quat_b = quat_b / np.linalg.norm(quat_b, axis=1, keepdims=True)
        dots = np.sum(quat_a * quat_b, axis=1, keepdims=True)
        flip_mask = dots < 0.0
        quat_b = quat_b.copy()
        quat_b[flip_mask[:, 0]] *= -1.0
        dots = np.sum(quat_a * quat_b, axis=1, keepdims=True)
        dots = np.clip(dots, -1.0, 1.0)

        result = np.empty_like(quat_a)
        linear_mask = dots[:, 0] > 0.9995
        if np.any(linear_mask):
            lerped = quat_a[linear_mask] + t * (quat_b[linear_mask] - quat_a[linear_mask])
            result[linear_mask] = lerped / np.linalg.norm(lerped, axis=1, keepdims=True)

        spherical_mask = ~linear_mask
        if np.any(spherical_mask):
            theta_0 = np.arccos(dots[spherical_mask])
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            s0 = np.cos(theta) - dots[spherical_mask] * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            result[spherical_mask] = s0 * quat_a[spherical_mask] + s1 * quat_b[spherical_mask]
            result[spherical_mask] /= np.linalg.norm(result[spherical_mask], axis=1, keepdims=True)
        return result

    def _find_best_transition_frame(self, src_node_name: str, src_frame: int, dst_node_name: str) -> int:
        src_cache = self.motion_cache[src_node_name]
        dst_cache = self.motion_cache[dst_node_name]
        src_motion = src_cache["motion"]
        dst_motion = dst_cache["motion"]
        src_frame = src_frame % src_motion.num_frames

        src_rel = src_cache["relative_translation"][src_frame, self.match_joint_indices, :]
        src_forward = src_cache["forward"][src_frame]
        src_velocity = src_cache["root_velocity"][src_frame]

        dst_rel = dst_cache["relative_translation"][:, self.match_joint_indices, :]
        pose_cost = np.mean((dst_rel - src_rel[None, :, :]) ** 2, axis=(1, 2))
        facing_cost = 1.0 - np.sum(dst_cache["forward"] * src_forward[None, :], axis=1)
        vel_cost = np.linalg.norm(dst_cache["root_velocity"] - src_velocity[None, :], axis=1)
        total_cost = pose_cost + 0.25 * facing_cost + 0.1 * vel_cost

        if dst_node_name in (self.node_alias["turn_left"], self.node_alias["turn_right"]):
            max_search = min(self.turn_entry_search_frames, dst_motion.num_frames)
            return int(np.argmin(total_cost[:max_search]))

        if dst_node_name == self.node_alias["jump"]:
            return 0
        if dst_node_name in self.root_locked_action_nodes:
            return 0

        walk_run_pair = {
            self.node_alias["walk"],
            self.node_alias["run"],
        }
        if src_node_name in walk_run_pair and dst_node_name in walk_run_pair:
            phase = float(src_frame % src_motion.num_frames) / float(max(src_motion.num_frames, 1))
            mapped = int(round(phase * dst_motion.num_frames)) % dst_motion.num_frames
            radius = min(self.phase_match_search_radius, max(dst_motion.num_frames // 2, 1))
            candidate_ids = [(mapped + offset) % dst_motion.num_frames for offset in range(-radius, radius + 1)]
            local_cost = total_cost[candidate_ids]
            return int(candidate_ids[int(np.argmin(local_cost))])

        return int(np.argmin(total_cost))

    def _begin_transition(
        self,
        src_node_name: str,
        src_frame: int,
        src_instance: BVHMotion,
        dst_node_name: str,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        duration: int = None,
    ):
        dst_frame = self._find_best_transition_frame(src_node_name, src_frame, dst_node_name)
        dst_instance = self._make_aligned_motion(dst_node_name, dst_frame, root_pos, root_rot)
        transition_duration = self.transition_blend_frames if duration is None else int(duration)
        self.transition = {
            "src_node_name": src_node_name,
            "src_frame": src_frame,
            "src_motion_instance": src_instance,
            "dst_node_name": dst_node_name,
            "dst_frame": dst_frame,
            "dst_motion_instance": dst_instance,
            "progress": 0,
            "duration": max(1, transition_duration),
        }
        self.current_node_name = dst_node_name
        self.current_frame = dst_frame
        self.active_motion_name = dst_node_name
        self.active_motion_start_frame = dst_frame
        self.active_motion_instance = dst_instance
        self.prev_sampled_root_pos = None
        self.prev_sampled_root_rot = None
        self.frames_since_motion_switch = 0

    def _quat_yaw(self, quat: np.ndarray) -> float:
        forward = R.from_quat(quat).apply(np.array([0.0, 0.0, 1.0]))
        return float(np.arctan2(forward[0], forward[2]))

    def _update_continuous_root(self, desired_vel: np.ndarray, desired_rot: np.ndarray):
        target_speed = float(np.linalg.norm(desired_vel[[0, 2]]))
        self.move_speed = self._damp_scalar(self.move_speed, target_speed, self.move_speed_sharpness)

        if self._is_root_locked_action_state():
            self.cur_root_pos = self.root_pos.copy()
            self.cur_root_rot = self._yaw_to_quat(self.root_yaw)
            self.last_desired_root_rot = self.cur_root_rot.copy()
            return

        if target_speed >= 1e-4:
            self.target_yaw = self._quat_yaw(desired_rot)
            self.root_yaw = self._damp_angle(self.root_yaw, self.target_yaw, self.turn_yaw_sharpness)

        self.cur_root_pos = self.root_pos.copy()
        self.cur_root_rot = self._yaw_to_quat(self.root_yaw)
        self.last_desired_root_rot = self.cur_root_rot.copy()

    def _advance_root_from_animation(self, sampled_root_pos: np.ndarray, sampled_root_rot: np.ndarray):
        if self.prev_sampled_root_pos is None or self.prev_sampled_root_rot is None:
            self.prev_sampled_root_pos = sampled_root_pos.copy()
            self.prev_sampled_root_rot = sampled_root_rot.copy()
            self.root_pos[1] = sampled_root_pos[1]
            self.cur_root_pos = self.root_pos.copy()
            return

        if self._is_jump_related_state():
            if self.jump_world_xz_anchor is None:
                self.jump_world_xz_anchor = self.root_pos[[0, 2]].copy()
            target_air_velocity = np.array([np.sin(self.root_yaw), np.cos(self.root_yaw)], dtype=np.float64) * self.move_speed
            control_sharpness = (
                self.jump_landing_control_sharpness
                if self.transition is not None and self.transition["src_node_name"] == self.node_alias["jump"]
                else self.jump_air_control_sharpness
            )
            alpha = 1.0 - np.exp(-control_sharpness * self.dt)
            self.jump_air_velocity = self.jump_air_velocity + alpha * (target_air_velocity - self.jump_air_velocity)
            self.jump_world_xz_anchor = self.jump_world_xz_anchor + self.jump_air_velocity * self.dt
            self.root_pos[[0, 2]] = self.jump_world_xz_anchor.copy()
            self.root_pos[1] = sampled_root_pos[1]
            self.cur_root_pos = self.root_pos.copy()
            self.prev_sampled_root_pos = sampled_root_pos.copy()
            self.prev_sampled_root_rot = sampled_root_rot.copy()
            return
        elif self._is_root_locked_action_state():
            self.root_pos[1] = sampled_root_pos[1]
            self.cur_root_pos = self.root_pos.copy()
            self.prev_sampled_root_pos = sampled_root_pos.copy()
            self.prev_sampled_root_rot = sampled_root_rot.copy()
            return
        elif self.jump_world_xz_anchor is not None:
            self.root_pos[[0, 2]] = self.jump_world_xz_anchor.copy()
            self.jump_world_xz_anchor = None
            self.jump_air_velocity[:] = 0.0
            self.root_pos[1] = sampled_root_pos[1]
            self.cur_root_pos = self.root_pos.copy()
            self.prev_sampled_root_pos = sampled_root_pos.copy()
            self.prev_sampled_root_rot = sampled_root_rot.copy()
            return

        if self._is_gait_transition_active():
            forward = np.array([np.sin(self.root_yaw), 0.0, np.cos(self.root_yaw)], dtype=np.float64)
            self.root_pos[[0, 2]] = self.root_pos[[0, 2]] + forward[[0, 2]] * self.move_speed * self.dt
            self.root_pos[1] = sampled_root_pos[1]
            self.cur_root_pos = self.root_pos.copy()
            self.prev_sampled_root_pos = sampled_root_pos.copy()
            self.prev_sampled_root_rot = sampled_root_rot.copy()
            return

        raw_delta = sampled_root_pos - self.prev_sampled_root_pos
        raw_delta[1] = 0.0

        prev_sampled_yaw = self._quat_yaw(self.prev_sampled_root_rot)
        local_delta = R.from_quat(self._yaw_to_quat(prev_sampled_yaw)).inv().apply(raw_delta)

        clip_speed = self.motion_cache[self.current_node_name].get("mean_speed", 0.0)
        if self.current_node_name == self.node_alias["idle"] or self.move_speed < self.idle_speed_threshold or clip_speed < 1e-6:
            speed_scale = 0.0
        else:
            speed_scale = float(np.clip(self.move_speed / clip_speed, 0.0, 1.5))

        scaled_local_delta = local_delta * speed_scale
        world_delta = R.from_quat(self.cur_root_rot).apply(scaled_local_delta)
        self.root_pos[[0, 2]] = self.root_pos[[0, 2]] + world_delta[[0, 2]]
        self.root_pos[1] = sampled_root_pos[1]
        self.cur_root_pos = self.root_pos.copy()
        self.prev_sampled_root_pos = sampled_root_pos.copy()
        self.prev_sampled_root_rot = sampled_root_rot.copy()

    def _select_locomotion_motion(self, speed: float) -> str:
        if speed < self.idle_speed_threshold:
            return self.node_alias["idle"]
        if self.controller.gait:
            return self.node_alias["run"]
        return self.node_alias["walk"]

    def _is_locomotion_node(self, node_name: str) -> bool:
        return node_name in (
            self.node_alias["idle"],
            self.node_alias["walk"],
            self.node_alias["run"],
        )

    def _transition_duration_for(self, src_node_name: str, dst_node_name: str) -> int:
        walk_run_pair = {
            self.node_alias["walk"],
            self.node_alias["run"],
        }
        if src_node_name == self.node_alias["jump"] and dst_node_name in self.loopable_nodes:
            return self.jump_landing_blend_frames
        if src_node_name in walk_run_pair and dst_node_name in walk_run_pair and src_node_name != dst_node_name:
            return self.gait_transition_blend_frames
        return self.transition_blend_frames

    def _is_gait_transition_active(self) -> bool:
        if self.transition is None:
            return False
        walk_run_pair = {
            self.node_alias["walk"],
            self.node_alias["run"],
        }
        return (
            self.transition["src_node_name"] in walk_run_pair
            and self.transition["dst_node_name"] in walk_run_pair
            and self.transition["src_node_name"] != self.transition["dst_node_name"]
        )

    def _is_jump_related_state(self) -> bool:
        if self.current_node_name == self.node_alias["jump"]:
            return True
        if self.transition is None:
            return False
        return (
            self.transition["src_node_name"] == self.node_alias["jump"]
            or self.transition["dst_node_name"] == self.node_alias["jump"]
        )

    def _is_root_locked_action_state(self) -> bool:
        if self.current_node_name in self.root_locked_action_nodes:
            return True
        if self.transition is None:
            return False
        return (
            self.transition["src_node_name"] in self.root_locked_action_nodes
            or self.transition["dst_node_name"] in self.root_locked_action_nodes
        )

    def _begin_jump(self, desired_vel: np.ndarray):
        move_dir = desired_vel[[0, 2]].astype(np.float64)
        move_speed = float(np.linalg.norm(move_dir))
        if move_speed > 1e-6:
            launch_velocity = move_dir * self.jump_takeoff_speed_scale
        else:
            launch_velocity = np.zeros(2, dtype=np.float64)
        self.jump_air_velocity = launch_velocity
        self.jump_world_xz_anchor = self.root_pos[[0, 2]].copy()
        self.pending_locked_action = self.node_alias["jump"]

    def _begin_special_action(self, action_name: str):
        self.pending_locked_action = self.node_alias[action_name]
        input_speed = float(np.linalg.norm(self.controller.input_vel[[0, 2]]))
        self.special_action_interrupt_armed = input_speed < self.idle_speed_threshold

    def _begin_upper_body_action(self, action_name: str):
        motion_name = self.upper_body_action_nodes[action_name]
        self.upper_body_action_name = action_name
        self.upper_body_motion_name = motion_name
        self.upper_body_frame = 0
        self.upper_body_motion_instance = self._make_aligned_motion(
            motion_name,
            0,
            self.root_pos.copy(),
            self.cur_root_rot.copy(),
        )

    def _uses_animated_root_orientation_state(self) -> bool:
        if self.current_node_name in self.animated_root_orientation_nodes:
            return True
        if self.transition is None:
            return False
        return (
            self.transition["src_node_name"] in self.animated_root_orientation_nodes
            or self.transition["dst_node_name"] in self.animated_root_orientation_nodes
        )

    def _select_target_motion(
        self,
        desired_vel_list: np.ndarray,
    ) -> str:
        speed = float(np.linalg.norm(desired_vel_list[0][[0, 2]]))
        input_speed = float(np.linalg.norm(self.controller.input_vel[[0, 2]]))

        if self.pending_locked_action in self.root_locked_action_nodes:
            if input_speed < self.idle_speed_threshold:
                self.special_action_interrupt_armed = True
            elif self.special_action_interrupt_armed:
                self.pending_locked_action = None
                self.special_action_interrupt_armed = False
                return self.node_alias["walk"]

        if self.pending_locked_action is not None:
            return self.pending_locked_action

        for action_name in self.special_action_order:
            if self.controller.consume_action(action_name):
                self._begin_special_action(action_name)
                return self.pending_locked_action

        for action_name in self.upper_body_action_order:
            if self.controller.consume_action(action_name):
                self._begin_upper_body_action(action_name)
                break

        if self.controller.consume_jump():
            self._begin_jump(desired_vel_list[0])
            return self.pending_locked_action

        desired_node = self._select_locomotion_motion(speed)
        if (
            self._is_locomotion_node(self.current_node_name)
            and self._is_locomotion_node(desired_node)
            and desired_node != self.current_node_name
            and self.frames_since_motion_switch < self.min_locomotion_residency_frames
        ):
            return self.current_node_name
        return desired_node

    def _maybe_switch_motion(self, target_node_name: str):
        if target_node_name == self.current_node_name:
            return

        valid_next = self.graph.get_successor_names(self.current_node_name)
        if target_node_name not in valid_next:
            return

        self._begin_transition(
            self.current_node_name,
            self.current_frame,
            self.active_motion_instance,
            target_node_name,
            self.root_pos.copy(),
            self.cur_root_rot.copy(),
            duration=self._transition_duration_for(self.current_node_name, target_node_name),
        )

    def _sample_current_motion(self):
        motion = self.motion_cache[self.current_node_name]["motion"]
        joint_name = self.motion_cache[self.current_node_name]["joint_name"]
        wrapped = False

        if self.transition is not None:
            progress = self.transition["progress"]
            src_translation, src_orientation, _ = self._sample_motion_pose(
                self.transition["src_node_name"],
                self.transition["src_motion_instance"],
                self.transition["src_frame"] + progress,
            )
            dst_translation, dst_orientation, frame_id = self._sample_motion_pose(
                self.transition["dst_node_name"],
                self.transition["dst_motion_instance"],
                self.transition["dst_frame"] + progress,
            )
            blend_t = float(progress + 1) / float(self.transition["duration"])
            joint_translation = (1.0 - blend_t) * src_translation + blend_t * dst_translation
            joint_orientation = self._quat_slerp(src_orientation, dst_orientation, blend_t)
            self.transition["progress"] += 1
            next_frame = frame_id + 1
            wrapped = next_frame >= motion.num_frames
            self.current_frame = next_frame % motion.num_frames
            if self.transition["progress"] >= self.transition["duration"]:
                self.transition = None
        else:
            joint_translation, joint_orientation, frame_id = self._sample_motion_pose(
                self.current_node_name,
                self.active_motion_instance,
                self.current_frame,
            )
            next_frame = frame_id + 1
            wrapped = next_frame >= motion.num_frames
            self.current_frame = next_frame % motion.num_frames

        if wrapped and self.current_node_name == self.node_alias["jump"]:
            self.pending_locked_action = None
            speed = float(np.linalg.norm(self.controller.desired_vel[[0, 2]])) if hasattr(self.controller, "desired_vel") else 0.0
            if self.controller.gait and speed >= self.idle_speed_threshold:
                next_motion = self.node_alias["run"]
            elif speed >= self.idle_speed_threshold:
                next_motion = self.node_alias["walk"]
            else:
                next_motion = self.node_alias["idle"]
            jump_motion = self.motion_cache[self.node_alias["jump"]]["motion"]
            src_frame = max(0, jump_motion.num_frames - self.transition_blend_frames)
            self._begin_transition(
                self.node_alias["jump"],
                src_frame,
                self.active_motion_instance,
                next_motion,
                self.root_pos.copy(),
                self.cur_root_rot.copy(),
            )
        elif wrapped and self.current_node_name in self.root_locked_action_nodes:
            self.pending_locked_action = None
            self.special_action_interrupt_armed = True
            speed = float(np.linalg.norm(self.controller.desired_vel[[0, 2]])) if hasattr(self.controller, "desired_vel") else 0.0
            if self.controller.gait and speed >= self.idle_speed_threshold:
                next_motion = self.node_alias["run"]
            elif speed >= self.idle_speed_threshold:
                next_motion = self.node_alias["walk"]
            else:
                next_motion = self.node_alias["idle"]
            action_motion = self.motion_cache[self.current_node_name]["motion"]
            src_frame = max(0, action_motion.num_frames - self.transition_blend_frames)
            self._begin_transition(
                self.current_node_name,
                src_frame,
                self.active_motion_instance,
                next_motion,
                self.root_pos.copy(),
                self.cur_root_rot.copy(),
            )
        elif wrapped and self.current_node_name in self.loopable_nodes:
            self._start_motion_instance(
                self.current_node_name,
                0,
                self.root_pos.copy(),
                self.cur_root_rot.copy(),
            )

        return joint_name, joint_translation, joint_orientation

    def _retarget_pose_to_controlled_root(
        self,
        joint_translation: np.ndarray,
        joint_orientation: np.ndarray,
        advance_root: bool = True,
    ):
        source_root_pos = joint_translation[0].copy()
        source_root_rot = joint_orientation[0].copy()
        if self._uses_animated_root_orientation_state():
            target_root_rot = source_root_rot.copy()
        else:
            target_root_rot = self.cur_root_rot.copy()
        if advance_root:
            self._advance_root_from_animation(source_root_pos, source_root_rot)
        delta_rot = R.from_quat(target_root_rot) * R.from_quat(source_root_rot).inv()
        rel = joint_translation - source_root_pos[None, :]
        target_root_pos = source_root_pos.copy()
        target_root_pos[[0, 2]] = self.root_pos[[0, 2]]
        self.root_pos[1] = source_root_pos[1]
        self.cur_root_pos = self.root_pos.copy()
        retargeted_translation = delta_rot.apply(rel) + target_root_pos[None, :]
        retargeted_orientation = (delta_rot * R.from_quat(joint_orientation)).as_quat()
        retargeted_translation[0] = target_root_pos
        retargeted_orientation[0] = target_root_rot
        return retargeted_translation, retargeted_orientation

    def _sample_turn_visual_pose(self, yaw_error: float):
        abs_error = abs(yaw_error)
        target_weight = self._smoothstep(
            self.turn_visual_enter_angle,
            self.turn_visual_full_angle,
            abs_error,
        )
        self.turn_visual_weight = self._damp_scalar(
            self.turn_visual_weight,
            target_weight,
            self.move_speed_sharpness,
        )

        if self.turn_visual_weight < self.turn_visual_activation_weight:
            self.turn_visual_active = None
            return None, self.turn_visual_weight

        active_dir = "left" if yaw_error >= 0.0 else "right"
        if active_dir != self.turn_visual_active:
            self.turn_visual_frame[active_dir] = 0
            self.turn_visual_active = active_dir

        motion_name = self.node_alias["turn_left"] if active_dir == "left" else self.node_alias["turn_right"]
        motion = self.motion_cache[motion_name]["motion"]
        frame_id = self.turn_visual_frame[active_dir] % motion.num_frames
        turn_translation, turn_orientation = self._sample_motion_pose(motion_name, motion, frame_id)[:2]
        self.turn_visual_frame[active_dir] = (frame_id + 1) % motion.num_frames
        return self._retarget_pose_to_controlled_root(turn_translation, turn_orientation), self.turn_visual_weight

    def _apply_turn_visual(
        self,
        joint_translation: np.ndarray,
        joint_orientation: np.ndarray,
        yaw_error: float,
    ):
        turn_pose, weight = self._sample_turn_visual_pose(yaw_error)
        if turn_pose is None:
            return joint_translation, joint_orientation

        turn_translation, turn_orientation = turn_pose
        blended_translation = joint_translation.copy()
        blended_orientation = joint_orientation.copy()
        for joint_idx, mask_weight in enumerate(self.upper_body_mask):
            blend_weight = float(mask_weight * weight)
            if blend_weight <= 1e-6:
                continue
            blended_translation[joint_idx] = (
                (1.0 - blend_weight) * joint_translation[joint_idx]
                + blend_weight * turn_translation[joint_idx]
            )
            blended_orientation[joint_idx] = self._quat_slerp(
                joint_orientation[joint_idx : joint_idx + 1],
                turn_orientation[joint_idx : joint_idx + 1],
                blend_weight,
            )[0]

        blended_translation[0] = self.root_pos.copy()
        blended_orientation[0] = self.cur_root_rot.copy()
        return blended_translation, blended_orientation

    def _sample_upper_body_overlay_pose(self):
        if self.upper_body_motion_name is None or self.upper_body_motion_instance is None:
            return None

        motion = self.motion_cache[self.upper_body_motion_name]["motion"]
        if self.upper_body_frame >= motion.num_frames:
            self.upper_body_action_name = None
            self.upper_body_motion_name = None
            self.upper_body_motion_instance = None
            self.upper_body_frame = 0
            return None

        overlay_translation, overlay_orientation, frame_id = self._sample_motion_pose(
            self.upper_body_motion_name,
            self.upper_body_motion_instance,
            self.upper_body_frame,
        )
        self.upper_body_frame = frame_id + 1
        return self._retarget_pose_to_controlled_root(
            overlay_translation,
            overlay_orientation,
            advance_root=False,
        )

    def _upper_body_overlay_weight(self) -> float:
        if self.upper_body_motion_name is None:
            return 0.0

        motion = self.motion_cache[self.upper_body_motion_name]["motion"]
        fade = max(1, min(self.upper_body_overlay_fade_frames, motion.num_frames // 2))
        fade_in = min(1.0, float(self.upper_body_frame) / float(fade))
        fade_out = min(1.0, float(motion.num_frames - self.upper_body_frame) / float(fade))
        return float(np.clip(min(fade_in, fade_out), 0.0, 1.0))

    def _apply_upper_body_overlay(
        self,
        joint_translation: np.ndarray,
        joint_orientation: np.ndarray,
    ):
        overlay_pose = self._sample_upper_body_overlay_pose()
        if overlay_pose is None:
            return joint_translation, joint_orientation

        weight = self._upper_body_overlay_weight()
        if weight <= 1e-6:
            return joint_translation, joint_orientation

        overlay_translation, overlay_orientation = overlay_pose
        blended_translation = joint_translation.copy()
        blended_orientation = joint_orientation.copy()
        for joint_idx, mask_weight in enumerate(self.upper_body_mask):
            blend_weight = float(mask_weight * weight)
            if blend_weight <= 1e-6:
                continue
            blended_translation[joint_idx] = (
                (1.0 - blend_weight) * joint_translation[joint_idx]
                + blend_weight * overlay_translation[joint_idx]
            )
            blended_orientation[joint_idx] = self._quat_slerp(
                joint_orientation[joint_idx : joint_idx + 1],
                overlay_orientation[joint_idx : joint_idx + 1],
                blend_weight,
            )[0]

        blended_translation[0] = joint_translation[0].copy()
        blended_orientation[0] = joint_orientation[0].copy()
        return blended_translation, blended_orientation

    def update_state(
        self,
        desired_pos_list,
        desired_rot_list,
        desired_vel_list,
        desired_avel_list,
    ):
        desired_pos_list = np.asarray(desired_pos_list, dtype=np.float64)
        desired_rot_list = np.asarray(desired_rot_list, dtype=np.float64)
        desired_vel_list = np.asarray(desired_vel_list, dtype=np.float64)
        desired_avel_list = np.asarray(desired_avel_list, dtype=np.float64)
        self.frame_counter += 1
        self.frames_since_motion_switch += 1

        if not self.root_initialized:
            self.root_pos = desired_pos_list[0].copy()
            self.root_initialized = True
        self._update_continuous_root(desired_vel_list[0], desired_rot_list[0])

        if self.current_frame == 0 and self.transition is None and self.active_motion_name != self.current_node_name:
            self._start_motion_instance(self.current_node_name, self.current_frame, self.root_pos.copy(), self.cur_root_rot)

        if self.active_motion_name == self.current_node_name and self.current_frame == 0 and self.transition is None:
            self._start_motion_instance(self.current_node_name, self.current_frame, self.root_pos.copy(), self.cur_root_rot)

        target_node_name = self._select_target_motion(desired_vel_list)
        self._maybe_switch_motion(target_node_name)
        joint_name, joint_translation, joint_orientation = self._sample_current_motion()
        joint_translation, joint_orientation = self._retarget_pose_to_controlled_root(
            joint_translation,
            joint_orientation,
        )
        yaw_error = self._shortest_angle_diff(self.target_yaw, self.root_yaw)
        if not self._is_root_locked_action_state():
            joint_translation, joint_orientation = self._apply_turn_visual(
                joint_translation,
                joint_orientation,
                yaw_error,
            )
            joint_translation, joint_orientation = self._apply_upper_body_overlay(
                joint_translation,
                joint_orientation,
            )
        self.last_output_root_pos = joint_translation[0].copy()
        self.last_output_root_rot = joint_orientation[0].copy()
        return joint_name, joint_translation, joint_orientation

    def sync_controller_and_character(self, character_state):
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
