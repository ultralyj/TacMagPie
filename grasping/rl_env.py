import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw
from typing import Tuple, Dict, Any

class RobotIQ2F85Env:
    def __init__(self, xml_path=None, xml_string=None):
        """
        初始化RobotIQ 2F85夹爪仿真环境
        
        Args:
            xml_path: MJCF模型文件路径
            xml_string: MJCF模型字符串
        """
        # 加载模型
        if xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        elif xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            raise ValueError("必须提供xml_path或xml_string")
            
        self.data = mujoco.MjData(self.model)
        
        # 设置摄像头参数
        self.camera = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.opt)
        
        # 查找关键组件的ID
        self._init_ids()
        
        # 控制参数
        self.action_dim = 2  # 开合控制 + 升降控制
        self.observation_dim = self._get_observation_dim()
        
    def _init_ids(self):
        """初始化关键组件的ID"""
        # 查找执行器ID
        self.actuator_fingers = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        self.actuator_lift = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control"
        )
        
        # 查找绿色盒子的body ID
        self.green_box_body_id = None
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name == "box":
                self.green_box_body_id = i
                break
        
        # 查找flexcomp
        self.flex_comp_ids = {}
        for i in range(self.model.nflex):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_FLEX, i)
            if name in ['A', 'B']:
                self.flex_comp_ids[name] = i
        
    def _get_observation_dim(self):
        """计算观测空间的维度"""
        # 基础观测维度
        dim = 0
        
        # 1. 夹爪状态
        # 执行器位置
        dim += 2
        # 执行器速度
        dim += 2
        
        # 2. 绿色盒子位置和姿态
        if self.green_box_body_id is not None:
            dim += 7  # 位置(3) + 四元数(4)
            
        # 3. flexcomp节点位置
        for flex_id in self.flex_comp_ids.values():
            vertadr = self.model.flex_vertadr[flex_id]
            vertnum = self.model.flex_vertnum[flex_id]
            dim += vertnum * 3  # 每个节点3个坐标
            
        return dim
    
    def get_observation(self) -> np.ndarray:
        """
        获取环境观测
        
        Returns:
            np.ndarray: 观测向量
        """
        observations = []
        
        # 1. 获取夹爪状态
        # 执行器位置
        actuator_pos = np.array([
            self.data.actuator_length[self.actuator_fingers],
            self.data.actuator_length[self.actuator_lift]
        ])
        observations.append(actuator_pos)
        
        # 执行器速度
        actuator_vel = np.array([
            self.data.actuator_velocity[self.actuator_fingers],
            self.data.actuator_velocity[self.actuator_lift]
        ])
        observations.append(actuator_vel)
        
        # 2. 获取绿色盒子位置
        if self.green_box_body_id is not None:
            pos = self.data.xpos[self.green_box_body_id]
            quat = self.data.xquat[self.green_box_body_id]
            observations.extend([pos, quat])
        
        # 3. 获取flexcomp节点位置
        for name, flex_id in self.flex_comp_ids.items():
            # 获取flexcomp的顶点信息
            vertadr = self.model.flex_vertadr[flex_id]
            vertnum = self.model.flex_vertnum[flex_id]
            # 提取顶点位置
            for i in range(vertadr, vertadr + vertnum):
                x = self.data.flexvert_xpos[i].copy()
                observations.extend([x])
        
        return np.concatenate(observations)
    
    def get_flexcomp_positions(self) -> Dict[str, np.ndarray]:
        """
        获取两个flexcomp的所有节点位置
        
        Returns:
            Dict: 包含flexcomp A和B的节点位置
        """
        flex_positions = {}
        
        for name, flex_id in self.flex_comp_ids.items():
            vertadr = self.model.flex_vertadr[flex_id]
            vertnum = self.model.flex_vertnum[flex_id]
            
            positions = []
            for i in range(vertadr, vertadr + vertnum):
                pos = self.data.flexvert_xpos[i]
                positions.append(pos.copy())
            
            flex_positions[name] = np.array(positions)
            
        return flex_positions
    
    def get_green_box_info(self) -> Dict[str, np.ndarray]:
        """
        获取绿色盒子的位置和姿态信息
        
        Returns:
            Dict: 包含位置和姿态
        """
        if self.green_box_body_id is None:
            return {}
            
        return {
            'position': self.data.xpos[self.green_box_body_id].copy(),
            'orientation': self.data.xquat[self.green_box_body_id].copy(),
            'linear_velocity': self.data.cvel[self.green_box_body_id][3:6].copy(),
            'angular_velocity': self.data.cvel[self.green_box_body_id][:3].copy()
        }
    
    def get_gripper_state(self) -> Dict[str, float]:
        """
        获取夹爪当前状态
        
        Returns:
            Dict: 夹爪开合和抬升状态
        """
        return {
            'grip_position': self.data.actuator_length[self.actuator_fingers],
            'lift_position': self.data.actuator_length[self.actuator_lift],
            'grip_velocity': self.data.actuator_velocity[self.actuator_fingers],
            'lift_velocity': self.data.actuator_velocity[self.actuator_lift]
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步动作
        
        Args:
            action: [grip_control, lift_control]，范围[-1, 1]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # 转换动作到控制信号
        grip_ctrl = np.clip(action[0], -1, 1)
        lift_ctrl = np.clip(action[1], -1, 1)
        
        # 映射到实际控制范围
        # 开合控制: 0-255，映射到[0, 255]
        grip_target = 127.5 + grip_ctrl * 127.5
        
        # 抬升控制: 0-0.1，映射到[0, 0.1]
        lift_target = 0.05 + lift_ctrl * 0.05
        
        # 设置控制信号
        self.data.ctrl[self.actuator_fingers] = grip_target
        self.data.ctrl[self.actuator_lift] = lift_target
        
        # 前向仿真
        mujoco.mj_step(self.model, self.data)
        
        # 获取观测
        obs = self.get_observation()
        
        # 计算奖励（示例）
        reward = self._compute_reward()
        
        # 检查是否结束
        done = self._check_done()
        
        # 额外信息
        info = {
            'flex_positions': self.get_flexcomp_positions(),
            'green_box_info': self.get_green_box_info(),
            'gripper_state': self.get_gripper_state()
        }
        
        return obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 可选的：随机初始化盒子位置
        if self.green_box_body_id is not None:
            # 随机位置
            x = np.random.uniform(-0.05, 0.05)
            y = np.random.uniform(-0.05, 0.05)
            z = 0.03
            self.data.xpos[self.green_box_body_id] = [x, y, z]
            
            # 随机姿态
            quat = self._random_quaternion()
            self.data.xquat[self.green_box_body_id] = quat
        
        return self.get_observation()
    
    def _compute_reward(self) -> float:
        """计算奖励函数"""
        reward = 0.0
        
        # 1. 获取夹爪状态
        gripper_state = self.get_gripper_state()
        grip_pos = gripper_state['grip_position']  # 0=完全张开, 0.8=完全闭合
        lift_pos = gripper_state['lift_position']  # 0=最低, 0.1=最高
        # print(f"Grip Pos: {grip_pos}, Lift Pos: {lift_pos}")
        # 2. 获取盒子信息
        box_info = self.get_green_box_info()
        if not box_info:
            print("No green box found.")
            return reward  # 没有盒子，返回0
        
        box_pos = box_info['position']
        box_z = box_pos[2]  # 盒子高度
        
        # 3. 获取flexcomp形变信息
        flex_positions = self.get_flexcomp_positions()
        
        # 计算flexcomp形变量
        deformation = 0.0
        contact_detected = False
        if hasattr(self, 'flexcomp_initial_positions'):
            for name, current_pos in flex_positions.items():
                if name in self.flexcomp_initial_positions:
                    initial_pos = self.flexcomp_initial_positions[name]
                    # 计算每个节点的位移
                    node_deformation = np.linalg.norm(current_pos - initial_pos, axis=1)
                    deformation += node_deformation.mean()  # 平均形变量
        
        # 4. 检测接触
        # 通过检查flexcomp节点是否靠近盒子来检测接触
        contact_threshold = 0.01  # 10mm接触阈值
        for name, positions in flex_positions.items():
            for pos in positions:
                distance_to_box = np.linalg.norm(pos - box_pos)
                if distance_to_box < contact_threshold:
                    contact_detected = True
                    break
            if contact_detected:
                break
        
        # 5. 计算夹爪与盒子的水平距离
        # 假设夹爪中心在(0, 0, 高度)，计算与盒子的水平距离
        horizontal_distance = np.linalg.norm(box_pos[:2])
        
        # 6. 计算夹爪闭合程度（0-1，1表示完全闭合）
        closure_ratio = grip_pos  # 归一化到0-1
        
        # 7. 计算奖励（分阶段奖励函数）
        
        # 阶段1: 鼓励靠近盒子
        if horizontal_distance > 0.05:  # 距离较远
            print(f"Horizontal Distance: {horizontal_distance}, far from box.")
            # 距离奖励：越近奖励越高
          
            # 轻微鼓励闭合，为接触做准备
            closure_reward = closure_ratio * 0.1
            reward += closure_reward
            
        else:  # 已经比较靠近
            # 阶段2: 鼓励接触
            # print(f"Horizontal Distance: {horizontal_distance}, not far from box.")
            if not contact_detected:
                # print("No contact detected yet.")
                # 距离更近的奖励
                # 更强的闭合鼓励
                closure_reward = closure_ratio * 0.5
                reward += closure_reward
                
            else:  # 检测到接触
                print("Contact detected!")
                # 阶段3: 鼓励保持接触并形变
                contact_bonus = 2.0  # 接触基础奖励
                reward += contact_bonus
                
                # 形变奖励：形变越大，奖励越高
                if deformation > 0:
                    deformation_reward = deformation * 10.0
                    reward += deformation_reward
                    
                    # 额外的形变保持奖励
                    if deformation > 0.001:  # 微小形变
                        steady_deformation_reward = 0.5
                        reward += steady_deformation_reward
                
                # 闭合保持奖励：闭合状态越好，奖励越高
                if closure_ratio > 0.7:  # 70%以上闭合
                    closure_maintain_reward = closure_ratio * 1.0
                    reward += closure_maintain_reward
                    
                    if closure_ratio > 0.9:  # 接近完全闭合
                        strong_grip_bonus = 1.0
                        reward += strong_grip_bonus
        
        return reward
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        # 示例：检查盒子是否掉落到地面以下
        if self.green_box_body_id is not None:
            box_pos = self.data.xpos[self.green_box_body_id]
            if box_pos[2] < 0:  # 低于地面
                return True
        
        # 检查步数限制
        if self.data.time > 10.0:  # 10秒限制
            return True
            
        return False
    
    def _random_quaternion(self) -> np.ndarray:
        """生成随机四元数"""
        u, v, w = np.random.random(3)
        quat = np.array([
            np.sqrt(1 - u) * np.sin(2 * np.pi * v),
            np.sqrt(1 - u) * np.cos(2 * np.pi * v),
            np.sqrt(u) * np.sin(2 * np.pi * w),
            np.sqrt(u) * np.cos(2 * np.pi * w)
        ])
        return quat
    
    def render(self, viewer=None):
        """渲染环境"""
        if viewer is None:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return viewer

class RLAgent:
    """强化学习智能体基类"""
    def __init__(self, action_dim, observation_dim):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """获取动作"""
        raise NotImplementedError
        
    def update(self, obs, action, reward, next_obs, done):
        """更新策略"""
        raise NotImplementedError

class RandomAgent(RLAgent):
    """随机智能体（用于测试）"""
    def __init__(self, action_dim, observation_dim):
        super().__init__(action_dim, observation_dim)
        
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

def main():
    """主函数：演示如何使用环境"""
    # 创建环境
    env = RobotIQ2F85Env(xml_path="./model/2f85.xml")  # 使用提供的XML字符串
    
    # 创建随机智能体
    agent = RandomAgent(env.action_dim, env.observation_dim)
    
    # 创建查看器
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    # 重置环境
    obs = env.reset()
    
    # 仿真循环
    for episode in range(10):  # 10个episode
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 获取动作
            action = agent.get_action(obs)
            
            # 执行一步
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 输出信息
            if env.data.time % 1.0 < 0.01:  # 每秒输出一次
                print(f"\n时间: {env.data.time:.2f}s")
                print(f"夹爪状态: {info['gripper_state']}")
                print(f"盒子位置: {info['green_box_info'].get('position', [0,0,0])}")
                
                flex_pos = info['flex_positions']
                if 'A' in flex_pos:
                    print(f"Flexcomp A 第一个节点位置: {flex_pos['A'][0]}")
                if 'B' in flex_pos:
                    print(f"Flexcomp B 第一个节点位置: {flex_pos['B'][0]}")
            
            # 更新查看器
            viewer.sync()
            time.sleep(0.001)  # 控制仿真速度
        
        print(f"\nEpisode {episode} 完成，总奖励: {episode_reward:.2f}")
    
    viewer.close()

if __name__ == "__main__":
    # 从剪切板读取XML
    main()