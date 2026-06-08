import numpy as np

# part 0
def load_meta_data(bvh_file_path):
    """
    请把lab1-FK-part1的代码复制过来
    请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        channels: List[int]，整数列表，joint的自由度，根节点为6(三个平动三个转动)，其余节点为3(三个转动)
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量
    Tips:
        joint_name顺序应该和bvh一致
    """
    

    joints = []
    joint_parents = []
    channels = []
    joint_offsets = []

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
                    name = joints[parent_idx] + "_end"
                else:
                    name = pending_name

                current_idx = len(joints)
                joints.append(name)
                joint_parents.append(parent_idx)
                channels.append(0) 
                joint_offsets.append([0.0, 0.0, 0.0])
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
            vals = line.split()
            if current_idx is not None:
                joint_offsets[current_idx] = [float(vals[1]), float(vals[2]), float(vals[3])]
            continue

        if line.startswith("CHANNELS"):
            vals = line.split()
            if current_idx is not None:
                channels[current_idx] = int(vals[1])
            continue

    joint_offsets = np.array(joint_offsets, dtype=np.float32)
    return joints, joint_parents, channels, joint_offsets