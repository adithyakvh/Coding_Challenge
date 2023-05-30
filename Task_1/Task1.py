import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import copy
import time

def thief_and_cops(grid, orientations, fov):
        tick = time.time()
        indices = np.argwhere(grid > 0)
        policeman_positions = indices[np.argsort(grid[indices[:, 0], indices[:, 1]])]
        thief_position = np.where(grid<0)
        

        grid_height, grid_width = grid.shape
        x = np.arange(grid_width)
        y = np.arange(grid_height)
        xx, yy = np.meshgrid(x, y)

        # Adding 4 channels for each corner of every grid
        xx = np.repeat(xx[:, :, np.newaxis],  4, axis=2)
        yy = np.repeat(yy[:, :, np.newaxis],  4, axis=2)

        min_max_fov = np.array([orientations - fov/2, orientations+fov/2]).T
        # range_fov

        # Adding 4th axis for number of policemen
        xx = np.repeat(xx[:, :, :, np.newaxis], len(policeman_positions), axis=3).astype(np.float32)
        yy = np.repeat(yy[:, :, :, np.newaxis], len(policeman_positions), axis=3).astype(np.float32)

        # 1st channel, top left point (x - 0.5, y)
        # 2nd channel, top right point (x + 0.5, y)
        # 3rd channel, bottom left point(x - 0.5, y + 0.5)
        # 4th channel, bottom right point(x + 0.5, y + 0.5)
        # Repeat for all cops, last dimension in array is for number of cops

        xx[:, :, 0:3:2, :] = xx[:, :, 0:3:2, :] - 0.5
        yy[:, :, 0:2, :] =   yy[:, :, 0:2, :] - 0.5
        xx[:, :, 1:4:2, :] =   xx[:, :, 1:4:2, :] + 0.5
        yy[:, :, 2:, :] =   yy[:, :, 2:, :] + 0.5

        policeman_row = np.tile(policeman_positions[:, 0].reshape(1, 1, len(policeman_positions)), 
                        (grid_height, grid_width, 1))[:, :, np.newaxis, :]
        policeman_col = np.tile(policeman_positions[:, 1].reshape(1, 1, len(policeman_positions)), 
                        (grid_height, grid_width, 1))[:, :, np.newaxis, :]

        angles = np.degrees(np.arctan2(yy - policeman_row, xx-policeman_col))
        angles = (360 - angles) % 360

        fov_lower = np.tile(min_max_fov[:, 0].reshape(1, 1, len(policeman_positions)), 
                           (angles.shape[0], angles.shape[1], 1))[:, :, np.newaxis, :]
        fov_upper = np.tile(min_max_fov[:, 1].reshape(1, 1, len(policeman_positions)), 
                           (angles.shape[0], angles.shape[1], 1))[:, :, np.newaxis, :]

        # Finding grids visible to any cop
        visible_grids = np.where((fov_lower < angles) & (angles < fov_upper))

        # Finding cops who can see the thief
        cop_ids = visible_grids[3][np.where((visible_grids[0] == thief_position[0]) & 
                                            (visible_grids[1] == thief_position[1]))] + 1
        cop_ids = list(set(cop_ids))
        
        # Creating a copy of original grid to populate grid values whether they are visible or safe
        new_grid = copy.deepcopy(grid)
        new_grid[visible_grids[0], visible_grids[1]] = 10
        new_grid[policeman_positions[:, 0], policeman_positions[:, 1]] = \
                                            grid[policeman_positions[:, 0], policeman_positions[:, 1]]
        
        # Finding safe grids and closest safe grid using Manhattan Distance
        safe_grids = np.where(new_grid == 0)
        distances = np.argsort(abs(safe_grids[1] - thief_position[1]) + abs(safe_grids[0] - thief_position[0]))
        closest_safe_grid = [safe_grids[0][distances[0]], safe_grids[1][distances[0]]]
        
        new_grid[closest_safe_grid[0], closest_safe_grid[1]] = -10

        tock = time.time()
        print(tick)
        print(tock)
        #######################################################################################################
        #########################################For plotting##############################################
        #######################################################################################################
        # For plotting   
        plt.text(thief_position[1], thief_position[0], 'T', color='red')

        # Enable minor grid lines
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        plt.gca().yaxis.set_minor_locator(minor_locator)

        # Customize the minor grid lines
        plt.grid(which='minor')

        plt.xticks(np.arange(0,grid.shape[1]))
        plt.yticks(np.arange(0,grid.shape[0]))

        plt.imshow(new_grid, cmap='viridis')
        plt.show()
        # End of plotting
        #######################################################################################################
        ########################################End of plotting#################################################
        #######################################################################################################

        return cop_ids, closest_safe_grid

#######################################################################################################
def main():
    
    grid = np.array([[0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [-20, 0, 0, 0, 2,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 1, 0, 0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 7, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 5, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

#     bad_grid = [[0, 0, 0, 0, 0],
#                 ['T', 0, 0, 0, 2],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 0]]
#     converted = np.array([[-20 if elem == 'T' else elem for elem in arr] for arr in bad_grid])
#     assert(np.all(converted == grid))

    orientations = np.array([180, 150, 180, 180, 180, 180, 180, 180 ])
    FoV = np.array([60, 60, 60, 60, 60, 60, 60, 60])
    cops, safe_grid = thief_and_cops(grid, orientations, FoV)
    print("Cops watching", cops)
    print("Safe grid", safe_grid)

    # # Display the grid using imshow
    # plt.imshow(grid, cmap='viridis')

    # # Add a colorbar for reference
    # plt.colorbar()

    # # Show the plot
    # plt.show()

if __name__ == "__main__":
    main()