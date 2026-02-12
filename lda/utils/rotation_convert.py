import numpy as np
import pytorch3d.transforms as pt
def batched_R_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    R: (T, 3, 3) batched rotation matrices
    return (T, 3) roll, pitch, yaw
    """
    # pitch = asin(-R[2,0])
    r20 = np.clip(R[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(-r20)

    # 判断奇异值（万向节锁）
    cos_pitch = np.cos(pitch)
    singular = np.abs(cos_pitch) < 1e-6   # True: pitch ≈ ±90°

    roll = np.zeros_like(pitch)
    yaw  = np.zeros_like(pitch)

    # ---- 非奇异情况 ----
    idx = ~singular
    if np.any(idx):
        roll[idx] = np.arctan2(R[idx, 2, 1], R[idx, 2, 2])
        yaw[idx]  = np.arctan2(R[idx, 1, 0], R[idx, 0, 0])

    # ---- 奇异情况 (pitch = +90° 或 -90°) ----
    # roll = atan2(-R[0,1], R[1,1])
    # yaw is set to zero (不可区分)
    idx = singular
    if np.any(idx):
        roll[idx] = np.arctan2(-R[idx, 0, 1], R[idx, 1, 1])
        yaw[idx]  = 0.0   # 常用处理方法：设置 yaw = 0

    return np.stack([roll, pitch, yaw], axis=1).astype(np.float64)

def batched_rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """
    rpy: (T, 3) roll, pitch, yaw
    return (T, 3, 3) batched rotation matrices
    """
    roll, pitch, yaw = rpy[:,0], rpy[:,1], rpy[:,2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(rpy), 3, 3))
    R[:,0,0] = cy*cp
    R[:,0,1] = cy*sp*sr - sy*cr
    R[:,0,2] = cy*sp*cr + sy*sr
    R[:,1,0] = sy*cp
    R[:,1,1] = sy*sp*sr + cy*cr
    R[:,1,2] = sy*sp*cr - cy*sr
    R[:,2,0] = -sp
    R[:,2,1] = cp*sr
    R[:,2,2] = cp*cr
    return R

def invert_homogeneous_batch(T):
    """对一批 4x4 齐次变换矩阵求逆（利用 SE(3) 结构）"""
    R = T[:, :3, :3]          # (B, 3, 3)
    t = T[:, :3, 3]           # (B, 3)
    R_inv = np.transpose(R, (0, 2, 1))      # (B, 3, 3)
    t_inv = -np.einsum('bij,bj->bi', R_inv, t)  # (B, 3)
    T_inv = np.eye(4)[np.newaxis, :, :].repeat(T.shape[0], axis=0)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv
    return T_inv

def calculate_delta_eef(eef_position: np.ndarray) -> np.ndarray:
    """
    先将所有 EE 位姿对齐到第 0 帧坐标系，
    再计算对齐后轨迹中相邻帧的相对位姿差（共 N-1 个）。
    输入: eef_position (N, 6) -> [x, y, z, roll, pitch, yaw]
    输出: delta_eef (N-1, 6) -> [dx, dy, dz, d_roll, d_pitch, d_yaw]
    """
    N = len(eef_position)
    if N < 2:
        return np.zeros((0, 6))

    # Step 1: 构建所有 4x4 pose 矩阵 T_i
    eef_matrix = np.tile(np.eye(4), (N, 1, 1))  # (N, 4, 4)
    eef_matrix[:, :3, :3] = batched_rpy_to_R(eef_position[:, 3:6])
    eef_matrix[:, :3, 3] = eef_position[:, :3]

    # Step 2: 计算 T_{0->i} = T_0^{-1} @ T_i
    T0_inv = invert_homogeneous_batch(eef_matrix[0:1])  # (1, 4, 4)
    T_0_to_i = T0_inv @ eef_matrix  # (1,4,4) @ (N,4,4) -> (N,4,4)

    # Step 3: 计算相邻帧在对齐后坐标系下的相对变换
    # ΔT_i = T_{0->i}^{-1} @ T_{0->i+1}
    T_0_to_i_inv = invert_homogeneous_batch(T_0_to_i[:-1])  # (N-1, 4, 4)
    delta_T = T_0_to_i_inv @ T_0_to_i[1:]                  # (N-1, 4, 4)

    # Step 4: 提取欧拉角和平移
    delta_R = delta_T[:, :3, :3]       # (N-1, 3, 3)
    delta_t = delta_T[:, :3, 3]        # (N-1, 3)
    delta_euler = batched_R_to_rpy(delta_R)  # (N-1, 3)

    delta_eef = np.concatenate([delta_t, delta_euler], axis=1)  # (N-1, 6)
    return delta_eef

def delta2abs(delta_eef: np.ndarray, initial_eef: np.ndarray) -> np.ndarray:
    """
    从相对位姿差（对齐到第0帧后的局部相对变换）和初始绝对位姿，
    恢复完整的绝对位姿序列。
    
    Args:
        delta_eef: (N-1, 6) -> [dx, dy, dz, d_roll, d_pitch, d_yaw]
                   注意：顺序是 [t, euler]，与 calculate_delta_eef 输出一致
        initial_eef: (6,) -> [x0, y0, z0, roll0, pitch0, yaw0]
    
    Returns:
        abs_eef: (N, 6) -> 绝对位姿序列
    """
    if len(delta_eef) == 0:
        return initial_eef[np.newaxis, :]  # (1, 6)

    N_minus_1 = len(delta_eef)
    N = N_minus_1 + 1

    # Step 1: 构造初始位姿 T0 (4x4)
    T0 = np.eye(4)
    T0[:3, :3] = batched_rpy_to_R(initial_eef[None, 3:6])[0]
    T0[:3, 3] = initial_eef[:3]

    # Step 2: 构造 delta_T 数组 (N-1, 4, 4)
    delta_T = np.tile(np.eye(4), (N_minus_1, 1, 1))
    delta_T[:, :3, :3] = batched_rpy_to_R(delta_eef[:, 3:6])
    delta_T[:, :3, 3] = delta_eef[:, :3]

    # Step 3: 累积得到 T_0_to_i: T_0_to_0 = I, T_0_to_1 = ΔT0, T_0_to_2 = ΔT0 @ ΔT1, ...
    T_0_to_i = np.tile(np.eye(4), (N, 1, 1))  # (N, 4, 4)
    for i in range(1, N):
        T_0_to_i[i] = T_0_to_i[i - 1] @ delta_T[i - 1]

    # Step 4: 转换回全局坐标系: T_i = T0 @ T_0_to_i
    T_abs = T0 @ T_0_to_i  # (4,4) @ (N,4,4) -> (N,4,4)

    # Step 5: 提取平移和欧拉角
    trans = T_abs[:, :3, 3]  # (N, 3)
    R_abs = T_abs[:, :3, :3]  # (N, 3, 3)
    euler = batched_R_to_rpy(R_abs)  # (N, 3)

    abs_eef = np.concatenate([trans, euler], axis=1)  # (N, 6)
    return abs_eef
if __name__ == "__main__":
    eef = np.array([
        [0.1, 0.1, 0.2, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.2, 0.1, 0.0, 0.1, 0.1, 0.1],
        [0.2, 0.1, 0.0, 0.1, 0.1, 0.1]
    ])
    delta = calculate_delta_eef(eef)
    import pdb; pdb.set_trace()
    eef_rec = delta2abs(delta, eef[0])
    print("Original:\n", eef)
    print("Recovered:\n", eef_rec)
    print("Max error:", np.max(np.abs(eef - eef_rec)))
    