"""
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bvh_motion import BVHMotion
from smooth_utils import *

# part1
def blend_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, v:float=None, input_alpha:np.ndarray=None) -> BVHMotion:
    '''
    输入: 两个将要blend的动作，类型为BVHMotion
          将要生成的BVH的速度v
          如果给出插值的系数alpha就不需要再计算了
          target_fps,将要生成BVH的fps
    输出: blend两个BVH动作后的动作，类型为BVHMotion
    假设两个动作的帧数分别为n1, n2
    首先需要制作blend 的权重适量 alpha
    插值系数alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    Tips:
        1. 计算给出两端动作的速度，两个BVH已经将Root Joint挪到(0.0, 0.0)的XOZ位置上了，为了便于你计算，我们假定提供的bvh都是沿着z轴正方向向前运动的
        2. 利用v计算插值系数alpha
        3. 线性插值以及Slerp
    '''
    
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros_like(res.joint_position)
    res.joint_rotation = np.zeros_like(res.joint_rotation)
    res.joint_rotation[...,3] = 1.0

    n1 = bvh_motion1.motion_length
    n2 = bvh_motion2.motion_length
    m = bvh_motion1.joint_position.shape[1]

    dt1 = bvh_motion1.frame_time
    dt2 = bvh_motion2.frame_time
    v1 = (bvh_motion1.joint_position[-1, 0, 2] - bvh_motion1.joint_position[0, 0, 2]) / max((n1-1)*dt1, 1e-8)
    v2 = (bvh_motion2.joint_position[-1, 0, 2] - bvh_motion2.joint_position[0, 0, 2]) / max((n2-1)*dt2, 1e-8)

    if input_alpha is not None:
        alpha = np.asarray(input_alpha, dtype=np.float64).reshape(-1)
        n3 = alpha.shape[0]
    else:
        w2 = (v - v1) / (v2 - v1)
        w2 = float(np.clip(w2, 0.0, 1.0))
        w1 = 1.0 - w2
        n3_float = (w1 * n1 * v1 + w2 * v2 * n2) / max(v, 1e-8)
        n3 = max(2, int(np.round(n3_float)))
        alpha = np.full((n3,), w2, dtype=np.float64)

    j_idx = np.rint(np.linspace(0, n1-1, n3)).astype(np.int32)
    k_idx = np.rint(np.linspace(0, n2-1, n3)).astype(np.int32)

    res.frame_time = bvh_motion1.frame_time
    res.joint_position = np.zeros((n3, m, 3), dtype=np.float64)
    res.joint_rotation = np.zeros((n3, m, 4), dtype=np.float64)

    w = alpha[:, None, None]
    p1 = bvh_motion1.joint_position[j_idx]
    p2 = bvh_motion2.joint_position[k_idx]
    res.joint_position = (1.0 - w) * p1 + w * p2

    t_key = np.array([0.0, 1.0])
    for i in range(n3):
        a = float(alpha[i])
        for j in range(m):
            q_pair = np.stack((bvh_motion1.joint_rotation[j_idx[i], j], bvh_motion2.joint_rotation[k_idx[i], j]), axis=0)
            slerp = Slerp(t_key, R.from_quat(q_pair))
            res.joint_rotation[i, j] = slerp([a]).as_quat()[0]
    
    return res

# part2
def build_loop_motion(bvh_motion:BVHMotion, ratio:float, half_life:float) -> BVHMotion:
    '''
    输入: 将要loop化的动作，类型为BVHMotion
          damping在前在后的比例ratio, ratio介于[0,1]
          弹簧振子damping效果的半衰期 half_life
          如果你使用的方法不含上面两个参数，就忽视就可以了，因接口统一保留
    输出: loop化后的动作，类型为BVHMotion
    
    Tips:
        1. 计算第一帧和最后一帧的旋转差、Root Joint位置差 (不用考虑X和Z的位置差)
        2. 如果使用"inertialization"，可以利用`smooth_utils.py`的
        `quat_to_avel`函数计算对应角速度的差距，对应速度的差距请自己填写
        3. 逐帧计算Rotations和Postions的变化
        4. 注意 BVH的fps需要考虑，因为需要算对应时间
        5. 可以参考`smooth_utils.py`的注释或者 https://theorangeduck.com/page/creating-looping-animations-motion-capture
    
    '''
    res = bvh_motion.raw_copy()

    n = res.motion_length
    
    ratio = float(np.clip(ratio, 0.0, 1.0))
    half_life = max(float(half_life), 1e-5)
    dt = res.frame_time

    q0 = res.joint_rotation[0]
    qT = res.joint_rotation[-1]
    r0 = R.from_quat(q0)
    rT = R.from_quat(qT)

    rot_off_front = (r0 * rT.inv()).as_rotvec()
    rot_off_back = (rT * r0.inv()).as_rotvec()

    y0 = res.joint_position[0, 0, 1]
    yT = res.joint_position[-1, 0, 1]
    y_off_front = y0 - yT
    y_off_back = -y_off_front

    avel = quat_to_avel(res.joint_rotation, dt)
    avel0 = avel[0]
    avelT = avel[-1]
    avel_off_front = avel0 - avelT
    avel_off_back = -avel_off_front

    root_vy = np.zeros(n, dtype = np.float64)
    root_vy[1:] = (res.joint_position[1:, 0, 1] - res.joint_position[:-1, 0, 1]) / dt
    root_vy[0] = root_vy[1]
    vy_off_front = root_vy[0] - root_vy[-1]
    vy_off_back = -vy_off_front

    for i in range(n):
        t_front = i * dt
        t_back = (n - 1 - i) * dt

        f_rot, _ = decay_spring_implicit_damping_rot(
            rot_off_front.copy(), avel_off_front.copy(), half_life, t_front
        )
        b_rot, _ = decay_spring_implicit_damping_rot(
            rot_off_back.copy(), avel_off_back.copy(), half_life, t_back
        )
        rot_blend = ratio * f_rot + (1.0 - ratio) * b_rot

        res.joint_rotation[i] = (
            R.from_rotvec(rot_blend) * R.from_quat(res.joint_rotation[i])
        ).as_quat()

        f_pos, _ = decay_spring_implicit_damping_pos(
            np.array([0.0, y_off_front, 0.0], dtype=np.float64),
            np.array([0.0, vy_off_front, 0.0], dtype=np.float64),
            half_life, t_front
        )
        b_pos, _ = decay_spring_implicit_damping_pos(
            np.array([0.0, y_off_back, 0.0], dtype=np.float64),
            np.array([0.0, vy_off_back, 0.0], dtype=np.float64),
            half_life, t_back
        )
        pos_blend = ratio * f_pos + (1.0 - ratio) * b_pos
        res.joint_position[i, 0, 1] += pos_blend[1]

    if n >= 2:
        w = np.linspace(0.0, 1.0, n, dtype=np.float64)
        blend = (0.5 - w).reshape(-1, 1)

        r_first = R.from_quat(res.joint_rotation[0])
        r_last = R.from_quat(res.joint_rotation[-1])
        rot_residual = (r_last * r_first.inv()).as_rotvec()
        rot_correction = blend[:, :, None] * rot_residual[None, :, :]

        for i in range(n):
            res.joint_rotation[i] = (
                R.from_rotvec(rot_correction[i]) * R.from_quat(res.joint_rotation[i])
            ).as_quat()

        y_residual = res.joint_position[-1, 0, 1] - res.joint_position[0, 0, 1]
        res.joint_position[:, 0, 1] += blend[:, 0] * y_residual

        res.joint_rotation = align_quat(res.joint_rotation, inplace=False)
    return res

# part3
def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''

    m1 = bvh_motion1.raw_copy()
    m2 = bvh_motion2.raw_copy()
    m2.adjust_joint_name(m1.joint_name)

    n1 = m1.motion_length
    n2 = m2.motion_length
    M = m1.joint_position.shape[1]

    mix_frame1 = int(np.clip(mix_frame1, 0, n1-1))
    max_mix = min(mix_time, n1 - mix_frame1, n2)
    max_mix = int(max(0, max_mix))

    anchor_pos_xz = m1.joint_position[mix_frame1, 0, [0, 2]]
    anchor_rot = m1.joint_rotation[mix_frame1, 0]
    anchor_facing_xz = R.from_quat(anchor_rot).apply(np.array([0.0, 0.0 ,1.0]))[[0, 2]]
    m2 = m2.translation_and_rotation(0, anchor_pos_xz, anchor_facing_xz)

    alpha = np.linspace(0.0, 1.0, max_mix, dtype=np.float64)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    p1 = m1.joint_position[mix_frame1:mix_frame1+max_mix]
    p2 = m2.joint_position[:max_mix]
    w = alpha[:, None, None]
    p_blend = (1.0 - w) * p1 + w * p2

    q_blend = np.zeros((max_mix, M, 4), dtype=np.float64)
    t_key = np.array([0.0, 1.0])
    for i in range(max_mix):
        a = float(alpha[i])
        for j in range(M):
            q_pair = np.stack((m1.joint_rotation[mix_frame1 + i, j], m2.joint_rotation[i, j]), axis=0)
            slerp = Slerp(t_key, R.from_quat(q_pair))
            q_blend[i, j] = slerp([a]).as_quat()[0]

    res = m1.raw_copy()
    res.joint_position = np.concatenate(
        [m1.joint_position[:mix_frame1], p_blend, m2.joint_position[max_mix:]],
        axis = 0
    )
    res.joint_rotation = np.concatenate(
        [m1.joint_rotation[:mix_frame1], q_blend, m2.joint_rotation[max_mix:]],
        axis = 0
    )

    return res

