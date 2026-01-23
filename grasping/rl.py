import numpy as np
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import multiprocessing as mp

# ==================== 改进的超参数配置 ====================
class Config:
    MODEL_PATH = "./model/2f85.xml"
    MAX_STEPS = 500  # 减少步数，加快训练
    
    # DQN参数
    LEARNING_RATE = 3e-4
    GAMMA = 0.98
    EPSILON_START = 1.0
    EPSILON_END = 0.1  # 保持一定探索
    EPSILON_DECAY = 0.998
    
    # 训练参数
    BATCH_SIZE = 128
    MEMORY_SIZE = 50000
    TARGET_UPDATE = 5
    NUM_EPISODES = 2000
    WARMUP_EPISODES = 50  # 预热阶段
    
    # 改进的动作空间 - 更精细的控制
    GRIPPER_ACTIONS = [ 150, 170, 190]  # 5个夹爪位置
    SLIDER_ACTIONS = [0.0,  0.1]  # 5个高度
    
    # 课程学习
    USE_CURRICULUM = True
    CURRICULUM_STAGES = [
        {"max_steps": 300, "box_range": 0.005, "episodes": 200},
        {"max_steps": 300, "box_range": 0.015, "episodes": 200},
        {"max_steps": 300, "box_range": 0.025, "episodes": 200},
    ]
    
    # HER (Hindsight Experience Replay)
    USE_HER = True
    HER_RATIO = 0.8
    
    MODEL_SAVE_PATH = "./saved_models"
    RENDER = True
    RENDER_EVERY = 1

# ==================== 改进的DQN网络（加入Dueling架构）====================
class DuelingDQN(nn.Module):
    """Dueling DQN网络 - 分离状态价值和动作优势"""
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ==================== 优先经验回放 ====================
class PrioritizedReplayBuffer:
    """优先经验回放 - 优先采样重要经验"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# ==================== 改进的环境 ====================
class ImprovedGraspingEnv:
    """改进的抓取环境 - 更密集的奖励"""
    def __init__(self, model_path, render=False, curriculum_stage=0):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render
        if self.render_mode:
            self.viewer = None
        else:
            self.viewer = None
        self.curriculum_stage = curriculum_stage
        
        # 获取关键索引
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        self.slider_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "position_control"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )
        
        self.flex_comp_ids = {}
        for i in range(self.model.nflex):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_FLEX, i)
            if name in ['Elas_A', 'Elas_B']:
                self.flex_comp_ids[name] = i
                
        self.gripper_actions = Config.GRIPPER_ACTIONS
        self.slider_actions = Config.SLIDER_ACTIONS
        self.num_actions = len(self.gripper_actions) * len(self.slider_actions)
        
        self.initial_box_pos = np.array([0.0, 0.0, 0.03])
        
        # 记录历史信息用于奖励塑形
        self.prev_distance = None
        self.prev_box_height = None
        self.contact_steps = 0
        self.max_box_height = 0.03
        
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        
        # 随机化盒子位置
        self.data.qpos[self._get_box_qpos_idx():self._get_box_qpos_idx()+3] = \
            self.initial_box_pos
        
        # 初始化夹爪为打开状态
        self.data.ctrl[self.gripper_actuator_id] = 0
        self.data.ctrl[self.slider_actuator_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        # 重置历史信息
        box_pos = self._get_box_pos()
        gripper_pos = self._get_gripper_pos()
        self.prev_distance = np.linalg.norm(gripper_pos - box_pos)
        self.prev_box_height = box_pos[2]
        self.contact_steps = 0
        self.max_box_height = box_pos[2]
        
        return self._get_state()
    
    def _get_box_qpos_idx(self):
        return self.model.body_jntadr[self.box_body_id]
    
    def _get_box_pos(self):
        box_qpos_idx = self._get_box_qpos_idx()
        return self.data.qpos[box_qpos_idx:box_qpos_idx+3].copy()
    
    def _get_gripper_pos(self):
        """获取夹爪中心位置"""
        slider_pos = self.data.ctrl[self.slider_actuator_id]
        gripper_z = 0.18 - slider_pos
        return np.array([0.0, 0.0, gripper_z])
    
    def _get_state(self):
        """提取状态向量 - 简化版本"""
        box_pos = self._get_box_pos()
        box_qpos_idx = self._get_box_qpos_idx()
        
        box_vel = self.data.qvel[box_qpos_idx:box_qpos_idx+3].copy()
        
        gripper_pos = self.data.ctrl[self.gripper_actuator_id]
        slider_pos = self.data.ctrl[self.slider_actuator_id]
        
        # 相对位置更重要
        gripper_3d_pos = self._get_gripper_pos()
        relative_pos = box_pos - gripper_3d_pos
        
        # 检测接触
        has_contact = float(self._check_contact())
        
        state = np.concatenate([
            [box_pos[2]],           # 3
            [box_vel[2]],           # 3
            [gripper_pos],     # 1
            [slider_pos],      # 1
        ])
        
        return state
    
    def _check_contact(self):
        """检测夹爪与盒子的接触"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom1_name and geom2_name:
                if ('pad' in geom1_name and 'box' in geom2_name) or \
                   ('pad' in geom2_name and 'box' in geom1_name):
                    return True
        return False
    
    def step(self, action_idx):
        """执行动作"""
        gripper_idx = action_idx // len(self.slider_actions)
        slider_idx = action_idx % len(self.slider_actions)
        
        gripper_action = self.gripper_actions[gripper_idx]
        slider_action = self.slider_actions[slider_idx]
        
        self.data.ctrl[self.gripper_actuator_id] = gripper_action
        self.data.ctrl[self.slider_actuator_id] = slider_action
        
        # 模拟
        for _ in range(25):  # 减少仿真步数加快训练
            mujoco.mj_step(self.model, self.data)
            if self.render_mode and self.viewer is not None:
                self.viewer.sync()
        
        next_state = self._get_state()
        reward, done, info = self._compute_reward()
        
        return next_state, reward, done, info
    
    def _compute_reward(self):
        """改进的奖励函数 - 更密集的奖励"""
        box_pos = self._get_box_pos()
        box_z = box_pos[2]
        gripper_pos = self._get_gripper_pos()
        
        has_contact = self._check_contact()
        
        reward = 0.0
        done = False
        info = {}
        flex = []
        
        flex_deformation_a = self.data.qpos[5:5+96].copy().reshape(-1, 3)
        flex_deformation_b = self.data.qpos[105:105+96].copy().reshape(-1, 3)
        x_bias = np.sum(flex_deformation_a[:, 0]) + np.sum(flex_deformation_b[:, 0])
        y_bias = np.sum(flex_deformation_a[:, 1]) + np.sum(flex_deformation_b[:, 1])
        z_bias = np.sum(flex_deformation_a[:, 2]) + np.sum(flex_deformation_b[:, 2])
        
        reward += 40 * (y_bias) + 20 * z_bias
        gripper_ctrl = self.data.ctrl[self.gripper_actuator_id]
        if gripper_ctrl > 100:  # 接触时应该闭合
            reward += 0.003
            
        if box_z > 0.03:
            lift_reward = (box_z - 0.03) * 100
            reward += lift_reward
            self.max_box_height = max(self.max_box_height, box_z)
        
        self.prev_box_height = box_z
        
        if box_z > 0.04:
            reward += 500.0
            done = True
            info['success'] = True
        
        if box_z < 0.005:  # 盒子掉落
            reward -= 20.0
            done = True
            info['failure'] = True
        
        reward -= 1
        
        
        info['distance'] = 0
        info['contact'] = has_contact
        info['box_height'] = box_z
        
        return reward, done, info
    
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# ==================== 改进的DQN智能体 ====================
class ImprovedDQNAgent:
    """改进的DQN智能体"""
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 使用Dueling DQN
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # 使用优先经验回放
        self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE)
        
        self.epsilon = config.EPSILON_START
        self.steps = 0
        self.beta = 0.4  # 重要性采样权重
        
    def select_action(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """训练一步"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return None
        
        # 增加beta（重要性采样权重）
        self.beta = min(1.0, self.beta + 0.001)
        
        # 采样
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.config.BATCH_SIZE, self.beta)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # 用policy网络选择动作
            next_actions = self.policy_net(next_states).argmax(1)
            # 用target网络评估价值
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.config.GAMMA * next_q_values
        
        # TD误差
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # 加权损失
        loss = (weights * nn.MSELoss(reduction='none')(current_q_values.squeeze(), target_q_values)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_END, 
                          self.epsilon * self.config.EPSILON_DECAY)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

# ==================== 改进的训练循环 ====================
import multiprocessing as mp
from functools import partial
import time

def train(process_id=0, base_save_path="models"):
    """改进的训练循环 - 支持多进程"""
    config = Config()
    
    # 为每个进程创建独立的保存路径
    config.MODEL_SAVE_PATH = os.path.join(base_save_path, f"process_{process_id}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # 设置随机种子，确保每个进程有不同的初始化
    import random
    import numpy as np
    import torch
    seed = int(time.time() * 1000) % 10000 + process_id * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"[进程 {process_id}] 开始训练，种子: {seed}")
    
    # 初始化环境 - 禁用渲染以提高速度
    env = ImprovedGraspingEnv(config.MODEL_PATH, render=True if config.RENDER else False)
    
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_actions
    
    print(f"[进程 {process_id}] 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    agent = ImprovedDQNAgent(state_dim, action_dim, config)
    
    episode_rewards = []
    success_count = 0
    recent_successes = deque(maxlen=100)
    
    # 课程学习阶段
    current_stage = 0
    stage_episodes = 0
    
    print(f"[进程 {process_id}] 🚀 开始训练...")
    
    for episode in range(config.NUM_EPISODES):
        # 更新课程学习阶段
        if config.USE_CURRICULUM and current_stage < len(config.CURRICULUM_STAGES):
            stage_info = config.CURRICULUM_STAGES[current_stage]
            if stage_episodes >= stage_info["episodes"]:
                current_stage += 1
                stage_episodes = 0
                if current_stage < len(config.CURRICULUM_STAGES):
                    print(f"[进程 {process_id}] 📚 进入课程学习阶段 {current_stage + 1}")
            env.curriculum_stage = current_stage
            stage_episodes += 1
        
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        max_steps = config.MAX_STEPS
        if config.USE_CURRICULUM and current_stage < len(config.CURRICULUM_STAGES):
            max_steps = config.CURRICULUM_STAGES[current_stage]["max_steps"]
        
        episode_info = {'max_distance': 0, 'contact_count': 0, 'max_height': 0}
        
        while not done and step < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # 记录信息
            if 'distance' in info:
                episode_info['max_distance'] = max(episode_info['max_distance'], info['distance'])
            if info.get('contact', False):
                episode_info['contact_count'] += 1
            if 'box_height' in info:
                episode_info['max_height'] = max(episode_info['max_height'], info['box_height'])
            
            agent.memory.push(state, action, reward, next_state, float(done))
            
            # 预热后开始训练
            if episode >= config.WARMUP_EPISODES:
                loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        success = info.get('success', False)
        if success:
            success_count += 1
        recent_successes.append(1 if success else 0)
        
        if episode % config.TARGET_UPDATE == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        # 详细的进度输出
        if episode % 10 == 0:  # 减少输出频率避免混乱
            avg_reward = np.mean(episode_rewards[-10:])
            success_rate = sum(recent_successes) / len(recent_successes) * 100
            print(f"[进程 {process_id}] Ep {episode:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Success: {success_rate:5.1f}%")
            
            # 写入日志
            log_path = os.path.join(config.MODEL_SAVE_PATH, "training_log.txt")
            with open(log_path, "a") as f:
                f.write(f"{episode},{avg_reward:.2f},{agent.epsilon:.3f},{success_rate:.1f},"
                        f"{episode_info['contact_count']},{episode_info['max_height']:.3f}\n")
    
    # 保存最终模型
    final_path = os.path.join(config.MODEL_SAVE_PATH, "model_final.pth")
    agent.save(final_path)
    print(f"[进程 {process_id}] ✅ 训练完成！最终模型: {final_path}")
    
    env.close()
    
    # 返回关键指标
    return {
        'process_id': process_id,
        'final_success_rate': sum(recent_successes) / len(recent_successes) * 100,
        'avg_reward': np.mean(episode_rewards[-100:]),
        'total_successes': success_count
    }

def parallel_train(num_processes=10, base_save_path="models"):
    """并行训练多个实例"""
    print(f"🚀 启动 {num_processes} 个并行训练进程...")
    print("=" * 80)
    
    # 创建进程池
    with mp.Pool(processes=num_processes) as pool:
        # 使用partial固定base_save_path参数
        train_func = partial(train, base_save_path=base_save_path)
        
        # 并行执行训练
        results = pool.map(train_func, range(num_processes))
    
    print("\n" + "=" * 80)
    print("📊 所有训练完成！结果汇总：")
    print("=" * 80)
    
    # 汇总结果
    for result in results:
        print(f"进程 {result['process_id']}: "
              f"成功率 {result['final_success_rate']:.2f}%, "
              f"平均奖励 {result['avg_reward']:.2f}, "
              f"总成功次数 {result['total_successes']}")
    
    # 找出最佳模型
    best_result = max(results, key=lambda x: x['final_success_rate'])
    print(f"\n🏆 最佳模型来自进程 {best_result['process_id']}")
    print(f"   成功率: {best_result['final_success_rate']:.2f}%")
    
    return results


    
# ==================== 测试函数 ====================
def test(model_path, num_episodes=10):
    """测试模型"""
    config = Config()
    env = ImprovedGraspingEnv(config.MODEL_PATH, render=True)
    
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_actions
    
    agent = ImprovedDQNAgent(state_dim, action_dim, config)
    agent.load(model_path)
    agent.epsilon = 0.0
    
    print(f"\n🧪 测试模型: {model_path}")
    print("=" * 60)
    
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        env.render()
        
        while not done and step < config.MAX_STEPS:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        if info.get('success', False):
            success_count += 1
            print(f"✅ Episode {episode + 1}: 成功! 奖励: {episode_reward:.2f}")
        else:
            print(f"❌ Episode {episode + 1}: 失败. 奖励: {episode_reward:.2f}")
    
    print(f"\n📊 测试完成! 成功率: {success_count / num_episodes * 100:.1f}%")
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="改进的MuJoCo强化学习抓取任务")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "train_para"],)
    parser.add_argument("--model", type=str, default="./saved_models/model_final.pth")
    
    
    args = parser.parse_args()
    
    if args.mode == "train_para":
        Config.RENDER = False  # 多进程训练时禁用渲染
        # 设置多进程启动方式（Windows需要）
        mp.set_start_method('spawn', force=True)
    
        # 运行并行训练
        results = parallel_train(num_processes=20, base_save_path="parallel_models")
    elif args.mode == "train":
        agent, rewards = train()
    else:
        test(args.model)