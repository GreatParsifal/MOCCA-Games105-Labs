###################################
# 学号： 2300012811
# 姓名： 马培杰
###################################

import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo

DEBUG_LOG_PATH = "part3_debug.log"


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose: (20, 4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs: 指定参数，可能包含kp,kd
    输出： global_torque: (20, 3)的numpy数组，表示每个关节的全局坐标下的目标力矩，因为不需要控制方向，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    """kp = kargs.get('kp', 500) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    kd = kargs.get('kd', 20) 
    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_body_orientation()
    # print(physics_info.get_root_pos_and_vel())
    parent_index = physics_info.parent_index
    joint_avel = physics_info.get_body_angular_velocity()
    target_rotation = pose

    global_torque = np.zeros((20,3)) """
    # 你的代码
    kp = kargs.get('kp',4000)
    kd = kargs.get('kd',50)
    parent_index = physics_info.parent_index
    joint_orientation = physics_info.get_body_orientation()
    joint_avel = physics_info.get_body_angular_velocity()
    target_rotation = pose

    global_torque = np.zeros((20,3))

    for i in range(len(parent_index)):
        p = parent_index[i]

        cur_R = R.from_quat(joint_orientation[i])
        tar_R_local = R.from_quat(target_rotation[i])

        if p == -1:
            parent_R = R.identity()
            parent_w = np.zeros(3)
            cur_R_local = cur_R
        else:
            parent_R = R.from_quat(joint_orientation[p])
            parent_w = joint_avel[p]
            cur_R_local = parent_R.inv() * cur_R

        err_R_local = tar_R_local * cur_R_local.inv()
        err_rotvec_local = err_R_local.as_rotvec()

        rel_w_world = joint_avel[i] - parent_w
        rel_w_local = parent_R.inv().apply(rel_w_world)

        torque_local = kp * err_rotvec_local - kd * rel_w_local

        torque_local = np.clip(torque_local, -1000, 1000)

        global_torque[i] = parent_R.apply(torque_local)
    
    # 抹去根节点力矩
    #global_torque[0] = np.zeros_like(global_torque[0])
    return global_torque

def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力，在后续仿真中只会保留y方向的力
           global_root_torque: (3,)的numpy数组，表示根节点的全局坐标下的辅助力矩，用来控制角色的朝向，实际上就是global_torque的第0项
           global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
        3. 可以适当将根节点目标位置上提以产生更大的辅助力，使角色走得更自然
    '''
    # ------一些提示代码，你可以随意修改------------#
    """ global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_torque = global_torque[0] """
    # 你的代码
    global_torque = part1_cal_torque(pose, physics_info, kp=kargs.get('kp', 500), kd=kargs.get('kd', 20))

    kp = kargs.get('root_kp', 4000)
    kd = kargs.get('root_kd', 50)
    
    root_position, root_velocity = physics_info.get_root_pos_and_vel()

    target_position = target_position.copy()
    #target_position[1] += 0.05

    pos_err = target_position - root_position
    vel_err = -root_velocity

    global_root_force = kp * pos_err + kd * vel_err
    global_root_force = np.clip(global_root_force, -5000, 5000)

    global_root_torque = global_torque[0].copy()
    global_root_torque = np.clip(global_root_torque, -1000, 1000)
    
    ########
    return global_root_force, global_root_torque, global_torque



frame_cnt = 0
def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (20, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃，如果效果好可以加分（0-20），可以考虑让角色站稳后再摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    joint_positions = physics_info.get_joint_translation()
    joint_orientation = physics_info.get_body_orientation()
    joint_avel = physics_info.get_body_angular_velocity()
    parent_index = physics_info.parent_index
    root_position, root_velocity = physics_info.get_root_pos_and_vel()

    global frame_cnt
    frame_cnt += 1

    right_foot_idx, left_foot_idx = 9, 10
    right_foot_pos = joint_positions[right_foot_idx]
    left_foot_pos = joint_positions[left_foot_idx]
    right_support_pos = right_foot_pos + R.from_quat(
        joint_orientation[right_foot_idx]
    ).apply([0.010, 0.002, 0.060])
    left_support_pos = left_foot_pos + R.from_quat(
        joint_orientation[left_foot_idx]
    ).apply([-0.010, 0.002, 0.060])

    target_position = (
        0.1 * right_foot_pos
        + 0.1 * left_foot_pos
        + 0.4 * right_support_pos
        + 0.4 * left_support_pos
    )
    target_position[1] = bvh.joint_position[0][0][1]

    global_torque = np.zeros((20, 3))
    child_rotation = R.from_quat(joint_orientation[1:])
    parent_rotation = R.from_quat(joint_orientation[parent_index[1:]])
    current_local_rotation = child_rotation.inv() * parent_rotation
    target_local_rotation = R.from_quat(pose[1:])
    rotation_error = (target_local_rotation * current_local_rotation).as_euler('xyz', degrees=True)
    global_torque[1:] = parent_rotation.apply(200 * rotation_error) - 10 * joint_avel[1:]

    torque_norm = np.linalg.norm(global_torque, axis=1, keepdims=True)
    scale = np.ones_like(torque_norm)
    overflow_mask = torque_norm > 1000
    scale[overflow_mask] = 0.02
    global_torque *= scale

    virtual_force = 4000 * (target_position - root_position) - 20 * root_velocity
    controlled_joints = {
        'RootJoint', 'rHip', 'lHip', 'rKnee', 'lKnee',
        'rAnkle', 'lAnkle', 'rToeJoint', 'lToeJoint'
    }
    torque = global_torque.copy()
    for i, name in enumerate(joint_name):
        if name not in controlled_joints:
            continue
        lever_arm = root_position - joint_positions[i]
        torque[i] -= np.cross(lever_arm, virtual_force)

    torque = np.clip(torque, -1000, 1000)

    """ if frame_cnt == 1:
        with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as log_file:
            log_file.write("part3 debug log\n")

    if frame_cnt % 200 == 0:
        leg_torque_norm = np.linalg.norm(torque[1:], axis=1)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(
                "[part3] "
                f"frame={frame_cnt} "
                f"root_pos={np.round(root_position, 3)} "
                f"root_vel={np.round(root_velocity, 3)} "
                f"target_pos={np.round(target_position, 3)} "
                f"offset={np.round(target_position - root_position, 3)} "
                f"virtual_force={np.round(virtual_force, 3)} "
                f"max_leg_torque={leg_torque_norm.max():.2f} "
                f"mean_leg_torque={leg_torque_norm.mean():.2f}\n"
            ) """

    torque[0] = np.zeros_like(torque[0])
    return torque
