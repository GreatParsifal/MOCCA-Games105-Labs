import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, input_joint_positions, input_joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """    
    joint_positions = input_joint_positions.copy()
    joint_orientations = input_joint_orientations.copy()

    path, _, _, _ = meta_data.get_path_from_root_to_end()

    root_idx = path[0]
    end_idx = path[-1]
    fixed_root_pos = joint_positions[root_idx].copy()

    joint_parent = meta_data.joint_parent
    joint_num = len(joint_parent)

    children = [[] for _ in range(joint_num)]
    for i, p in enumerate(joint_parent):
        if p >= 0:
            children[p].append(i)

    descendants = [[] for _ in range(joint_num)]
    for j in range(joint_num):
        stack = [j]
        chain = []
        while stack:
            cur = stack.pop()
            chain.append(cur)
            stack.extend(children[cur])
        descendants[j] = chain

    controllable_joints = path[:-1]
    max_iter = 80
    threshold = 1e-2
    damping = 1e-2
    step_scale = 0.6
    max_angle = 0.2

    for _ in range(max_iter):
        end_pos = joint_positions[end_idx]
        error = target_pose - end_pos
        if np.linalg.norm(error) < threshold:
            break

        dof = 3 * len(controllable_joints)
        jacobian = np.zeros((3, dof), dtype=np.float64)

        for i, j_idx in enumerate(controllable_joints):
            joint_pos = joint_positions[j_idx]
            orient = R.from_quat(joint_orientations[j_idx])
            axes_world = orient.apply(np.eye(3))
            arm = end_pos - joint_pos
            for k in range(3):
                jacobian[:, 3 * i + k] = np.cross(axes_world[k], arm)

        jj_t = jacobian @ jacobian.T
        delta_theta = jacobian.T @ np.linalg.solve(
            jj_t + (damping ** 2) * np.eye(3),
            error,
        )
        delta_theta *= step_scale
        delta_theta = np.clip(delta_theta, -max_angle, max_angle)

        for i, j_idx in enumerate(controllable_joints):
            dtheta = delta_theta[3 * i: 3 * i + 3]
            angle = np.linalg.norm(dtheta)
            if angle < 1e-10:
                continue

            orient = R.from_quat(joint_orientations[j_idx])
            axis_world = orient.apply(dtheta / angle)
            delta_rot = R.from_rotvec(axis_world * angle)

            pivot = joint_positions[j_idx].copy()
            for node in descendants[j_idx]:
                rel = joint_positions[node] - pivot
                joint_positions[node] = pivot + delta_rot.apply(rel)
                joint_orientations[node] = (delta_rot * R.from_quat(joint_orientations[node])).as_quat()

        # Keep the chosen root joint fixed in world space.
        offset = fixed_root_pos - joint_positions[root_idx]
        joint_positions += offset

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    joint_positions = joint_positions.copy()
    joint_orientations = joint_orientations.copy()

    path, _, _, _ = meta_data.get_path_from_root_to_end()
    end_idx = path[-1]
    wrist_idx = meta_data.joint_parent[end_idx]

    # Stage-1 uses shoulder/elbow chain to place wrist; wrist itself does not move wrist position.
    pos_joints = path[:-2] if len(path) >= 3 else path[:-1]

    joint_parent = meta_data.joint_parent
    joint_num = len(joint_parent)
    children = [[] for _ in range(joint_num)]
    for i, p in enumerate(joint_parent):
        if p >= 0:
            children[p].append(i)

    descendants = [[] for _ in range(joint_num)]
    for j in range(joint_num):
        stack = [j]
        chain = []
        while stack:
            cur = stack.pop()
            chain.append(cur)
            stack.extend(children[cur])
        descendants[j] = chain

    if 'RootJoint' in meta_data.joint_name:
        root_idx = meta_data.joint_name.index('RootJoint')
    else:
        root_idx = path[0]

    # Position target: keep wrist horizontally fixed relative to root, with absolute world height.
    root_rot = R.from_quat(joint_orientations[root_idx])
    root_pos = joint_positions[root_idx]
    root_right = root_rot.apply(np.array([1.0, 0.0, 0.0]))
    root_forward = root_rot.apply(np.array([0.0, 0.0, 1.0]))
    root_right[1] = 0.0
    root_forward[1] = 0.0
    nr = np.linalg.norm(root_right)
    nf = np.linalg.norm(root_forward)
    root_right = root_right / nr if nr > 1e-8 else np.array([1.0, 0.0, 0.0])
    root_forward = root_forward / nf if nf > 1e-8 else np.array([0.0, 0.0, 1.0])

    target_wrist = root_pos + relative_x * root_right + relative_z * root_forward
    target_wrist[1] = target_height

    # -------- Stage 1: Wrist position IK (shoulder-elbow only) --------
    max_iter = 60
    pos_threshold = 1e-2
    damping = 5e-2
    step_scale = 0.35
    max_angle = 0.15

    for _ in range(max_iter):
        wrist_pos = joint_positions[wrist_idx]
        pos_error = target_wrist - wrist_pos
        if np.linalg.norm(pos_error) < pos_threshold or len(pos_joints) == 0:
            break

        dof = 3 * len(pos_joints)
        jac_pos = np.zeros((3, dof), dtype=np.float64)

        for i, j_idx in enumerate(pos_joints):
            joint_pos = joint_positions[j_idx]
            orient = R.from_quat(joint_orientations[j_idx])
            axes_world = orient.apply(np.eye(3))
            arm = wrist_pos - joint_pos
            for k in range(3):
                jac_pos[:, 3 * i + k] = np.cross(axes_world[k], arm)

        jjt = jac_pos @ jac_pos.T
        delta_theta = jac_pos.T @ np.linalg.solve(
            jjt + (damping ** 2) * np.eye(3),
            pos_error,
        )
        delta_theta *= step_scale
        delta_theta = np.clip(delta_theta, -max_angle, max_angle)

        for i, j_idx in enumerate(pos_joints):
            dtheta = delta_theta[3 * i: 3 * i + 3]
            angle = np.linalg.norm(dtheta)
            if angle < 1e-10:
                continue

            orient = R.from_quat(joint_orientations[j_idx])
            axis_world = orient.apply(dtheta / angle)
            delta_rot = R.from_rotvec(axis_world * angle)

            pivot = joint_positions[j_idx].copy()
            for node in descendants[j_idx]:
                rel = joint_positions[node] - pivot
                joint_positions[node] = pivot + delta_rot.apply(rel)
                joint_orientations[node] = (delta_rot * R.from_quat(joint_orientations[node])).as_quat()

    # -------- Stage 2: Wrist orientation hard constraint in world frame --------
    # Use the actual world-space wrist->end direction and align it to global +Y.
    world_up = np.array([0.0, 1.0, 0.0])
    wrist_pos = joint_positions[wrist_idx]
    hand_dir_world = joint_positions[end_idx] - wrist_pos
    hand_dir_norm = np.linalg.norm(hand_dir_world)
    if hand_dir_norm > 1e-10:
        hand_dir_world /= hand_dir_norm
        align_axis = np.cross(hand_dir_world, world_up)
        align_axis_norm = np.linalg.norm(align_axis)
        align_dot = np.clip(np.dot(hand_dir_world, world_up), -1.0, 1.0)
        if align_axis_norm > 1e-10:
            align_axis /= align_axis_norm
            align_angle = np.arctan2(align_axis_norm, align_dot)
            corr_rot = R.from_rotvec(align_axis * align_angle)

            pivot = joint_positions[wrist_idx].copy()
            for node in descendants[wrist_idx]:
                rel = joint_positions[node] - pivot
                joint_positions[node] = pivot + corr_rot.apply(rel)
                joint_orientations[node] = (corr_rot * R.from_quat(joint_orientations[node])).as_quat()
   
    return joint_positions, joint_orientations
    
