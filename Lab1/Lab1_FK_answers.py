import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    stack = []
    pending_type = None
    pending_name = None
    current_idx = None

    with open(bvh_file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = text.replace("{", "\n{\n").replace("}", "\n}\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("MOTION"):
            break

        if line.startswith("HIERARCHY"):
            continue

        if line.startswith("ROOT"):
            pending_type = "ROOT"
            pending_name = line.split(None, 1)[1].strip()
            continue

        if line.startswith("JOINT"):
            pending_type = "JOINT"
            pending_name = line.split(None, 1)[1].strip()
            continue

        if line.startswith("End Site"):
            pending_type = "END"
            pending_name = "end"
            continue

        if line == "{":
            if pending_type is not None:
                parent_idx = stack[-1] if stack else -1
                if pending_type == "END" and parent_idx != -1:
                    name = joint_name[parent_idx] + "_end"
                else:
                    name = pending_name

                current_idx = len(joint_name)
                joint_name.append(name)
                joint_parent.append(parent_idx)
                joint_offset.append([0.0, 0.0, 0.0])
                stack.append(current_idx)

                pending_type = None
                pending_name = None
            continue

        if line == "}":
            if stack:
                stack.pop()
                current_idx = stack[-1] if stack else None
            continue

        if line.startswith("OFFSET"):
            parts = line.split()
            offset = list(map(float, parts[1:4]))
            if current_idx is not None:
                joint_offset[current_idx] = offset
            continue

        if line.startswith("CHANNELS"):
            continue

    joint_offset = np.array(joint_offset, dtype=np.float32)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    M = len(joint_name)
    joint_positions = np.zeros((M, 3), dtype=np.float32)
    joint_orientations = np.zeros((M, 4), dtype=np.float32)

    frame = motion_data[frame_id]
    ch_idx = 0

    for i in range(M):
        p = joint_parent[i]
        is_end = joint_name[i].endswith("_end")

        if p == -1:
            t = frame[ch_idx:ch_idx + 3]
            ch_idx += 3
            euler = frame[ch_idx:ch_idx + 3]
            ch_idx += 3

            rot = R.from_euler("XYZ", euler, degrees=True)
            joint_positions[i] = t + joint_offset[i]
            joint_orientations[i] = rot.as_quat()
            continue

        parent_rot = R.from_quat(joint_orientations[p])
        parent_pos = joint_positions[p]

        if is_end:
            rot = parent_rot
        else:
            euler = frame[ch_idx:ch_idx + 3]
            ch_idx += 3
            local_rot = R.from_euler("XYZ", euler, degrees=True)
            rot = parent_rot * local_rot

        joint_positions[i] = parent_pos + parent_rot.apply(joint_offset[i])
        joint_orientations[i] = rot.as_quat()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)
    a_motion = load_motion_data(A_pose_bvh_path)

    t_index = {name: i for i, name in enumerate(t_name)}
    a_index = {name: i for i, name in enumerate(a_name)}
    common_names = [name for name in t_name if name in a_index]

    def build_children(parent_list):
        children = [[] for _ in range(len(parent_list))]
        for i, p in enumerate(parent_list):
            if p >= 0:
                children[p].append(i)
        return children

    t_children = build_children(t_parent)

    def align_one_vector(src, dst):
        """Return a stable shortest-arc rotation mapping src -> dst."""
        src_n = np.linalg.norm(src)
        dst_n = np.linalg.norm(dst)
        if src_n < 1e-8 or dst_n < 1e-8:
            return R.identity()

        a = src / src_n
        b = dst / dst_n
        cross = np.cross(a, b)
        cross_n = np.linalg.norm(cross)
        dot = np.clip(np.dot(a, b), -1.0, 1.0)

        if cross_n < 1e-8:
            if dot > 0:
                return R.identity()
            # Opposite direction: rotate 180 deg around an axis orthogonal to a.
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if np.abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = axis - np.dot(axis, a) * a
            axis /= np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis)

        axis = cross / cross_n
        angle = np.arccos(dot)
        return R.from_rotvec(angle * axis)

    q_map = {}
    for name in common_names:
        ti = t_index[name]
        ai = a_index[name]

        vec_a = []
        vec_t = []
        for child_t in t_children[ti]:
            child_name = t_name[child_t]
            if child_name not in a_index:
                continue
            child_a = a_index[child_name]

            oa = a_offset[child_a]
            ot = t_offset[child_t]

            vec_a.append(oa)
            vec_t.append(ot)

        if len(vec_a) == 0:
            q_map[name] = R.identity()
        elif len(vec_a) == 1:
            q_map[name] = align_one_vector(np.array(vec_a[0], dtype=np.float64), np.array(vec_t[0], dtype=np.float64))
        else:
            rot, _ = R.align_vectors(np.array(vec_t), np.array(vec_a))
            q_map[name] = rot

    def build_channel_map(names, parents):
        ch = 0
        channel_map = {}
        root_name = None
        for i, name in enumerate(names):
            is_end = name.endswith("_end")
            if parents[i] == -1:
                root_name = name
                channel_map[name] = {
                    "pos": slice(ch, ch + 3),
                    "rot": slice(ch + 3, ch + 6)
                }
                ch += 6
            elif not is_end:
                channel_map[name] = {"rot": slice(ch, ch + 3)}
                ch += 3
        return channel_map, root_name, ch

    t_ch_map, t_root, t_ch_num = build_channel_map(t_name, t_parent)
    a_ch_map, a_root, _ = build_channel_map(a_name, a_parent)

    n_frame = a_motion.shape[0]
    motion_data = np.zeros((n_frame, t_ch_num), dtype=np.float32)

    if t_root in a_ch_map and t_root in t_ch_map:
        motion_data[:, t_ch_map[t_root]["pos"]] = a_motion[:, a_ch_map[t_root]["pos"]]

    for f in range(n_frame):
        frame_a = a_motion[f]

        for name in t_name:
            if name.endswith("_end") or name not in t_ch_map or name not in a_ch_map:
                continue

            rot_slice_t = t_ch_map[name]["rot"]
            rot_slice_a = a_ch_map[name]["rot"]
            euler_a = frame_a[rot_slice_a]
            r_a = R.from_euler("XYZ", euler_a, degrees=True)

            ti = t_index[name]
            p = t_parent[ti]
            if p == -1:
                q_parent = R.identity()
            else:
                parent_name = t_name[p]
                q_parent = q_map.get(parent_name, R.identity())

            q_i = q_map.get(name, R.identity())
            r_t = q_parent * r_a * q_i.inv()
            motion_data[f, rot_slice_t] = r_t.as_euler("XYZ", degrees=True)

    return motion_data