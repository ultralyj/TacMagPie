import numpy as np
import config
from typing import Tuple, Optional, Union

def magnetic_dipole_field(m, r, mu_0=4e-7*np.pi):
    """
    计算磁偶极子产生的磁场
    
    参数:
    m: 形状为 (3,1) 的磁矩向量，如 np.array([[mx], [my], [mz]])
    r: 形状为 (3,1) 的位置向量，如 np.array([[rx], [ry], [rz]])
    mu_0: 真空磁导率，默认值 4π × 10^-7 H/m
    
    返回:
    B: 形状为 (3,1) 的磁感应强度向量
    """
    m = np.asarray(m, dtype=np.float64).reshape(3,1)
    r = np.asarray(r, dtype=np.float64).reshape(3,1)
    # 计算 |r| 的相关量
    r_norm = np.linalg.norm(r)  # |r|
    
    if r_norm < 1e-15:  # 使用一个小的阈值
        return np.zeros(3)
    
    r_norm_3 = r_norm**3        # |r|^3
    r_norm_5 = r_norm**5        # |r|^5
    
    # 计算点积 m·r
    # 对于形状为 (3,1) 的数组，使用转置后相乘
    m_dot_r = np.dot(m.T, r)[0, 0]  # 或者用 m.flatten().dot(r.flatten())
    
    # 计算公式的两项
    term1 = (3 * m_dot_r / r_norm_5) * r
    term2 = m / r_norm_3
    
    # 计算最终的磁场
    B = (mu_0 / (4 * np.pi)) * (term1 - term2)
    
    return B

def magnetic_flux_3axis(sensor_position: Union[list, np.ndarray], 
                       grid_vec: np.ndarray, 
                       grid_positions: np.ndarray) -> np.ndarray:
    """
    计算三轴磁通量
    
    参数:
    sensor_position: 传感器位置，形状为 (3,) 的数组或列表
    grid_vec: 网格向量场
    grid_positions: 网格位置
    
    返回:
    b_sensor: 形状为 (3,) 的传感器处磁感应强度
    """
    try:
        # 输入验证
        sensor_position = np.asarray(sensor_position, dtype=float).flatten()
        if sensor_position.shape != (3,):
            raise ValueError("传感器位置必须是长度为3的向量")
        
        # 获取弹性体位置
        elastomer_positions = get_elastomer_positions(grid_vec, grid_positions)
        
        # 磁矩方向 (z轴方向)
        m_dir = np.array([0, 0, 1], dtype=float)
        
        # 预计算磁矩
        m = config.B0 * m_dir
        
        # 初始化磁场累加器
        b_sensor = np.zeros(3, dtype=np.float64)
        
        # 使用向量化操作替代三重循环
        grid_shape = (config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1)
        
        # 遍历所有网格点
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    # 计算位置向量
                    r_vec = elastomer_positions[i, j, k] - sensor_position

                    # 计算该点产生的磁场
                    B = magnetic_dipole_field(m, r_vec)

                    # 累加到总磁场
                    b_sensor += B.flatten()
        
        return b_sensor
        
    except Exception as e:
        print(f"计算三轴磁通量时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(3)

def get_ref_positions(ref: Union[list, np.ndarray], grid_positions: np.ndarray) -> np.ndarray:
    """
    获取参考位置
    
    参数:
    ref: 参考偏移量，形状为 (3,)
    grid_positions: 网格位置
    
    返回:
    ref_positions: 参考位置
    """
    ref = np.asarray(ref, dtype=float).reshape(3)
    ref_positions = grid_positions + ref
    return ref_positions

def get_elastomer_positions(grid_vec: np.ndarray, grid_positions: np.ndarray) -> np.ndarray:
    """
    得到当前弹性体形变后的位置
    
    参数:
    grid_vec: 网格向量场
    grid_positions: 原始网格位置
    
    返回:
    形变后的弹性体位置
    """
    return grid_positions + grid_vec

def get_grid_positions() -> np.ndarray:
    """
    生成网格位置
    
    返回:
    grid_positions: 形状为 (GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2]-1, 3) 的网格位置数组
    """
    grid_shape = (config.GRID_SIZE[0], config.GRID_SIZE[1], config.GRID_SIZE[2]-1, 3)
    grid_positions = np.zeros(grid_shape, dtype=float)
    
    # 使用向量化操作生成网格
    i_coords = (np.arange(config.GRID_SIZE[0]) - (config.GRID_SIZE[0]-1)/2) * config.GRID_SPACING
    j_coords = (np.arange(config.GRID_SIZE[1]) - (config.GRID_SIZE[1]-1)/2) * config.GRID_SPACING
    k_coords = (np.arange(1, config.GRID_SIZE[2])) * config.GRID_SPACING  # k从1开始
    
    # 使用meshgrid生成坐标网格
    I, J, K = np.meshgrid(i_coords, j_coords, k_coords, indexing='ij')
    
    grid_positions[..., 0] = I
    grid_positions[..., 1] = J
    grid_positions[..., 2] = K
    
    return grid_positions

# 可选的性能优化版本（使用完全向量化）
def magnetic_flux_3axis_vectorized(sensor_position: Union[list, np.ndarray],
                                  grid_vec: np.ndarray,
                                  grid_positions: np.ndarray) -> np.ndarray:
    """
    向量化版本的三轴磁通量计算（性能更好）
    """
    try:
        sensor_position = np.asarray(sensor_position, dtype=float).reshape(3, 1, 1, 1)
        
        # 获取弹性体位置
        elastomer_positions = get_elastomer_positions(grid_vec, grid_positions)
        
        # 计算所有位置向量
        r_vectors = elastomer_positions - sensor_position
        
        # 计算所有距离
        r_norms = np.linalg.norm(r_vectors, axis=3, keepdims=True)
        
        # 避免除以零
        r_norms = np.where(r_norms < 1e-15, np.inf, r_norms)
        
        # 磁矩
        m = np.array([0, 0, config.B0], dtype=float).reshape(1, 1, 1, 3)
        
        # 计算点积
        m_dot_r = np.sum(m * r_vectors, axis=3, keepdims=True)
        
        # 计算磁场公式
        r_norm_3 = r_norms**3
        r_norm_5 = r_norms**5
        
        term1 = (3 * m_dot_r / r_norm_5) * r_vectors
        term2 = m / r_norm_3
        
        # 所有点的磁场
        mu_0_4pi = (4e-7*np.pi) / (4 * np.pi)  # 简化常数
        B_all = mu_0_4pi * (term1 - term2)
        
        # 对所有点求和
        b_sensor = np.sum(B_all, axis=(0, 1, 2))
        
        return b_sensor.flatten()
        
    except Exception as e:
        print(f"向量化计算三轴磁通量时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(3)