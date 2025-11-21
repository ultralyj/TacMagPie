import numpy as np
import config
def magnetic_flux_3axis(runner, sensors_positions):    
    runner.grid_vec #[x=21,y=21,z=3,3axis=3]
    grid_positions = np.zeros(runner.grid_vec.shape)
    for i in range(config.GRID_SIZE[0]):
        for j in range(config.GRID_SIZE[1]):
            for k in range(config.GRID_SIZE[2]):
                grid_positions[i][j][k][0] = (i-(config.GRID_SIZE[0]-1)/2)
                grid_positions[i][j][k][1] = (i-(config.GRID_SIZE[1]-1)/2)
                grid_positions[i][j][k][2] = (i-(config.GRID_SIZE[2]-1)/2)
                
def get_grid_positions():
    grid_positions = np.zeros((config.GRID_SIZE[0],config.GRID_SIZE[1],config.GRID_SIZE[2]-1,3))
    for i in range(config.GRID_SIZE[0]):
        for j in range(config.GRID_SIZE[1]):
            for k in range(config.GRID_SIZE[2]-1):
                grid_positions[i][j][k][0] = (i-(config.GRID_SIZE[0]-1)/2) * config.GRID_SPACING
                grid_positions[i][j][k][1] = (i-(config.GRID_SIZE[1]-1)/2) * config.GRID_SPACING
                grid_positions[i][j][k][2] = (i-(config.GRID_SIZE[2]-1)/2) * config.GRID_SPACING
    return grid_positions