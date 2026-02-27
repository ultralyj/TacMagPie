import numpy as np
import matplotlib.pyplot as plt

def live_plot_process(data_queue, sensor_num):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    fig.canvas.manager.set_window_title('Real-time Magnetic Flux Data')

    lines_x, lines_y, lines_z = [], [], []
    labels = ['Bx', 'By', 'Bz']
    colors = plt.cm.jet(np.linspace(0, 1, sensor_num))

    for i in range(sensor_num):
        lines_x.append(axes[0].plot([], [], color=colors[i], label=f'S{i}')[0])
        lines_y.append(axes[1].plot([], [], color=colors[i], label=f'S{i}')[0])
        lines_z.append(axes[2].plot([], [], color=colors[i], label=f'S{i}')[0])

    for i, ax in enumerate(axes):
        ax.set_ylabel(f'{labels[i]}')
        ax.legend(fontsize='small', ncol=2)
        ax.grid(True)

    axes[2].set_xlabel('Time (s)')

    time_data = []
    mag_history = [[] for _ in range(sensor_num)]

    print("绘图进程启动")

    while True:
        latest = []
        while not data_queue.empty():
            latest.append(data_queue.get())

        if any(item is None for item in latest):
            break

        if not latest:
            plt.pause(0.05)
            continue

        for t, data in latest:
            time_data.append(t)
            for i in range(sensor_num):
                mag_history[i].append(data[i])

        max_points = 2000
        time_data = time_data[-max_points:]
        mag_history = [h[-max_points:] for h in mag_history]

        for i in range(sensor_num):
            if mag_history[i]:
                arr = np.array(mag_history[i])
                lines_x[i].set_data(time_data, arr[:, 0])
                lines_y[i].set_data(time_data, arr[:, 1])
                lines_z[i].set_data(time_data, arr[:, 2])

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.001)

    plt.close()
