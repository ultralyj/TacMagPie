import os
import numpy as np
import config

def update_logger(fp, data, runner, sensor_data, magdata):
    perf = runner.get_performance_info()
    line = f"{perf['elapsed_time']:.2f},{perf['simulation_time']:.2f}," \
           f"{data.qpos[runner.joint_id]:.3f}," \
           f"{sensor_data['force']:.2f},{sensor_data['velocity']:.4f},"
    for i in range(config.SENSOR_NUMBER):
        line += f"{magdata[i][0]:.6f},{magdata[i][1]:.6f},{magdata[i][2]:.6f},"
    fp.write(line + "\n")
    fp.flush()

def save_grid_vec(runner, screenshot_dir, timestamp):
    filename = f'grid_{timestamp}_{int(runner.data.time*1000):06d}.npy'
    np.save(os.path.join(screenshot_dir, filename), runner.grid_vec)
