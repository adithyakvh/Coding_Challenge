import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import copy
from typing import Tuple, List
import time



class ThiefsAndCops:
    
    def __init__(self, grid : np.ndarray, orientations : np.ndarray, fov : np.ndarray)->None:
        """_summary_

        Args:
            grid (np.ndarray): _description_
            orientations (np.ndarray): _description_
            fov (np.ndarray): _description_
        """
        self.grid = grid
        self.orientations = orientations
        self.fov = fov
        self._make_grid()
    
    def _make_grid(self):
        """Initializes the gridX, gridY matrices for processing
        """
        self._start_timer() # initializes the timer

        self.grid_width, self.grid_height = self.grid.shape
        gridX, gridY = np.meshgrid(np.arange(self.grid_width), np.arange(self.grid_height))

        # Adding 4 channels for each corner of every grid
        gridX = np.repeat(gridX[:, :, np.newaxis],  4, axis=2)
        gridY = np.repeat(gridY[:, :, np.newaxis],  4, axis=2)
        indices = np.argwhere(self.grid > 0)
        self.policeman_positions = indices[np.argsort(self.grid[indices[:, 0], indices[:, 1]])]
        # Adding 4th axis for number of policemen
        gridX = np.repeat(gridX[:, :, :, np.newaxis], len(self.policeman_positions), axis=3).astype(np.float32)
        gridY = np.repeat(gridY[:, :, :, np.newaxis], len(self.policeman_positions), axis=3).astype(np.float32)
        
        # 1st channel, top left point (x - 0.5, y)
        # 2nd channel, top right point (x + 0.5, y)
        # 3rd channel, bottom left point(x - 0.5, y + 0.5)
        # 4th channel, bottom right point(x + 0.5, y + 0.5)
        # Repeat for all cops, last dimension in array is for number of cops
        
        gridX[:, :, 0:3:2, :] = gridX[:, :, 0:3:2, :] - 0.5
        gridY[:, :, 0:2, :] =   gridY[:, :, 0:2, :] - 0.5
        gridX[:, :, 1:4:2, :] =   gridX[:, :, 1:4:2, :] + 0.5
        gridY[:, :, 2:, :] =   gridY[:, :, 2:, :] + 0.5

        self.gridX = gridX
        self.gridY = gridY
    
    def findClosestSafeGrid(self)->Tuple[List, List]:
        """_summary_

        Returns:
            _type_: _description_
        """
        policeman_row = np.tile(self.policeman_positions[:, 0].reshape(1, 1, len(self.policeman_positions)), 
                        (self.grid_height, self.grid_width, 1))[:, :, np.newaxis, :]
        policeman_col = np.tile(self.policeman_positions[:, 1].reshape(1, 1, len(self.policeman_positions)), 
                        (self.grid_height, self.grid_width, 1))[:, :, np.newaxis, :]

        angles = np.degrees(np.arctan2(self.gridY - policeman_row, self.gridX-policeman_col))
        angles = (360 - angles) % 360

        min_max_fov = np.array([self.orientations - self.fov/2, self.orientations + self.fov/2]).T

        fov_lower = np.tile(min_max_fov[:, 0].reshape(1, 1, len(self.policeman_positions)), 
                           (angles.shape[0], angles.shape[1], 1))[:, :, np.newaxis, :]
        fov_upper = np.tile(min_max_fov[:, 1].reshape(1, 1, len(self.policeman_positions)), 
                           (angles.shape[0], angles.shape[1], 1))[:, :, np.newaxis, :]

        self.visible_grids = np.where((fov_lower < angles) & (angles < fov_upper))

        self.thief_position = np.where(self.grid<0)

        # Finding cops who can see the thief
        cop_ids = self.visible_grids[3][np.where((self.visible_grids[0] == self.thief_position[0]) & 
                                            (self.visible_grids[1] == self.thief_position[1]))] + 1
        cop_ids = list(set(cop_ids))
        
        # Creating a copy of original grid to populate grid values whether they are visible or safe
        new_grid = self.grid
        new_grid[self.visible_grids[0], self.visible_grids[1]] = 10
        new_grid[self.policeman_positions[:, 0], self.policeman_positions[:, 1]] = \
                                            self.grid[self.policeman_positions[:, 0], self.policeman_positions[:, 1]]
        
        # Finding safe grids and closest safe grid using Manhattan Distance
        self.safe_grids = np.where(new_grid == 0)
        distances = np.argsort(abs(self.safe_grids[1] - self.thief_position[1]) + abs(self.safe_grids[0] - self.thief_position[0]))
        self.closest_safe_grid = [self.safe_grids[0][distances[0]], self.safe_grids[1][distances[0]]]
        
        new_grid[self.closest_safe_grid[0], self.closest_safe_grid[1]] = -10
        
        self.new_grid = new_grid

        self._stop_timer()
        return cop_ids, self.closest_safe_grid
    
    def visualize(self)->None:
        """For visualization of the Grid
        """
        plt.text(self.thief_position[1], self.thief_position[0], 'T', color='red')

        # Enable minor grid lines
        color_grid = np.zeros((self.new_grid.shape[0], self.new_grid.shape[1], 3))
        color_grid[self.safe_grids] = (255,255,255)
        color_grid[self.policeman_positions[:, 0], self.policeman_positions[:, 1]] = (255,0,0)
        color_grid[self.closest_safe_grid[0], self.closest_safe_grid[1]] = (0,255,0)
        color_grid[self.thief_position[0], self.thief_position[1]] = (0, 0, 255)
        color_grid[self.visible_grids[0], self.visible_grids[1]] = (155, 155, 0)
        color_grid[self.policeman_positions[:, 0], self.policeman_positions[:, 1]] = (255,0,0)

        plt.text(self.thief_position[1], self.thief_position[0], 'T', color='red')

        # Enable minor grid lines
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        plt.gca().yaxis.set_minor_locator(minor_locator)

        # Customize the minor grid lines
        plt.grid(which='minor')
        # plt.title("Yellow grids are visible grids, White grids are thief, green grid is closest safe grid, blue grids are safe grid")

        plt.xticks(np.arange(0,self.grid.shape[1]))
        plt.yticks(np.arange(0,self.grid.shape[0]))
        plt.imshow(color_grid)

        plt.show() 
    
    def _start_timer(self):
        """Starts timer
        """
        self.tick = time.time()

    def _stop_timer(self):
        """Stops Timer
        """
        self.tock = time.time() - self.tick
    
#######################################################################################################
def main():
    
    # grid = np.array([[0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [-20, 0, 0, 0, 2,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 1, 0, 0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 7, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 5, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #                  [0  , 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    bad_grid = [[0, 0, 0, 0, 0],
                ['T', 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]]
    grid = np.array([[-20 if elem == 'T' else elem for elem in arr] for arr in bad_grid])
    # assert(np.all(converted == grid))

    orientations = np.array([180, 150])
    FoV = np.array([60, 60])
    

    solver = ThiefsAndCops(grid, orientations, FoV)
    cops, safe_grid = solver.findClosestSafeGrid()
    solver.visualize()
    print("Time Taken : ", solver.tock)
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