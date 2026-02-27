# main.py
"""
MPM 点云按压仿真 — 入口脚本

用法
----
python main.py                          # 使用 config.yaml + 自动生成演示球
python main.py my_config.yaml           # 指定配置文件
python main.py config.yaml ball.npy    # 指定配置 + 点云文件
python main.py ball.npy                 # 默认配置 + 指定点云文件
"""

import taichi as ti

from config_loader import SimConfig, load_config
from simulator     import MPMSimulator
from utils         import generate_demo_pointcloud, parse_args


def main():
    # ── 1. 解析命令行 ────────────────────────────────────────
    config_path, pc_file_arg = parse_args()

    # ── 2. 加载配置 ──────────────────────────────────────────
    print(f"[main] 加载配置: {config_path}")
    cfg = SimConfig(load_config(config_path))
    print(f"[main] {cfg}")

    # ── 3. 初始化 Taichi（必须在任何 ti.field 创建之前）──────
    ti.init(arch=ti.gpu)

    # ── 4. 准备点云文件 ──────────────────────────────────────
    if pc_file_arg:
        pc_file = pc_file_arg
        print(f"[main] 使用外部点云文件: {pc_file}")
    else:
        pc_cfg  = cfg.pointcloud_cfg
        pc_file = pc_cfg.get("file") or None

        if pc_file is None:
            demo    = pc_cfg["demo_ball"]
            pc_file = generate_demo_pointcloud(
                path   = demo.get("save_path", "./tmp/demo_ball.npy"),
                radius = float(demo.get("radius",   0.015)),
                n      = int(demo.get("n_points",   2000)),
            )

    # ── 5. 创建仿真并运行 ────────────────────────────────────
    sim = MPMSimulator(pc_file=pc_file, cfg=cfg)
    sim.run()


if __name__ == "__main__":
    main()
