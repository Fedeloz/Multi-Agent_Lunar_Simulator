import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import io

'''
Description : Class that describes the Map for motion planning
'''
class Map:
    def __init__(self,obstacle_list,\
                cfa,\
                x_limits = [0,10],\
                    y_limits = [0,10]):
        """[summary]
        Args:
            state_init ([type]): [description]
            state_final ([type]): [description]
            obstacle_list ([type]): [description]

        """
        # obstacles that are visible in the global map 
        self.obstacle_list = obstacle_list        
        # obstacles not visbile in the global map 
        # visible locally
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.cfa = cfa
        self.x_min = x_limits[0]
        self.x_max = x_limits[1]
        self.y_min = y_limits[0]
        self.y_max = y_limits[1]
        self.num_obstacles = len(self.obstacle_list)

    def generate_grid(self,resolution, map_file_name=None):
        num_rows = int((self.x_max - self.x_min)/resolution)
        num_cols = int((self.y_max - self.y_min)/resolution)
        grid = np.zeros(num_rows*num_cols, dtype=np.uint8)
        for i in range(num_rows):
            for j in range(num_cols):
                xd = self.x_min + (i*resolution)
                yd = self.y_min + (j*resolution)
                if self.in_collision(xd,yd):
                    grid[i + j*num_rows] = 100

        if map_file_name:
            # Reshape grid to 2D array
            grid_2d = grid.reshape((num_cols, num_rows))
            # Create PIL image from numpy array
            image = Image.fromarray(grid_2d)
            # Save the image
            image.save(map_file_name)

        return grid
    
    def generate_grid_mpl(self,resolution, map_file_name=None):


        def set_axes_size(w,h, ax=None):
            """ w, h: width, height in inches """
            if not ax: ax=plt.gca()
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(w)/(r-l)
            figh = float(h)/(t-b)
            ax.figure.set_size_inches(figw, figh)
        num_rows = int((self.x_max - self.x_min)/resolution)
        num_cols = int((self.y_max - self.y_min)/resolution)
        
        dpi = num_cols
        fig, ax = plt.subplots(figsize=(num_cols/dpi, num_rows/dpi), dpi=dpi)
        # fig, ax = plt.subplots(figsize=(1, 1*(num_rows/num_cols))) #One-inch box

        # Add the obstacles as discs
        obstacle_circles = [
            Circle((_obstacle.state[0], _obstacle.state[1]), _obstacle.radius)
            for _obstacle in self.obstacle_list
        ]
        obstacle_collection = PatchCollection(obstacle_circles)
        ax.add_collection(obstacle_collection)
        # Set the axes
        ax.set_axis_off()
        set_axes_size(1,1*(num_rows/num_cols),ax)
        ax.set(xlim=self.x_limits,ylim=self.y_limits)


        # Set the resolution
        io_buf = io.BytesIO()
        # fig.savefig(io_buf, format='raw', dpi=num_cols, pad_inches = 0, bbox_inches='tight')
        fig.savefig(io_buf, format='raw', dpi=num_cols, pad_inches=0)
        io_buf.seek(0)
        # grid_color = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        #                     newshape=(num_cols, num_rows, -1))
        w, h = fig.canvas.get_width_height()
        grid_color = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(h, w, -1))

        # grid = np.array([
        #     np.array([0 if sum(grid_color[_col,_row])<(255*4) else 100 for _col in range(num_cols)])
        #     for _row in range(num_rows)])
        io_buf.close()
        # for _col in range(num_cols):
        #     for _row in range(num_rows):
        #         for _rgba in range(4):
        #             if grid[_col,_row,_rgba] != 255:
        #                 print("Entry {} {} has entry {}".format(_col, _row, grid[_col,_row]))
        #                 break
        grid = np.zeros(num_rows*num_cols, dtype=np.uint8)
        for _row in range(num_rows):
            for _col in range(num_cols):
                if sum(grid_color[_col,_row])<(255*4):
                    grid[_row + (num_cols-_col-1)*num_rows] = 100
        #         xd = self.x_min + (i*resolution)
        #         yd = self.y_min + (j*resolution)
        #         if self.in_collision(xd,yd):
        #             grid[i + j*num_rows] = 100

        if map_file_name:
            # Reshape grid to 2D array
            grid_2d = grid.reshape((num_cols, num_rows))
            # Create PIL image from numpy array
            image = Image.fromarray(grid_2d)
            # Save the image
            image.save(map_file_name)
            # fig.savefig('RAW_'+map_file_name, dpi=num_cols, pad_inches = 0, bbox_inches='tight')
            fig.savefig('RAW_'+map_file_name, dpi=num_cols, pad_inches=0)
            plt.close(fig)



        return grid


    def in_collision(self,x,y):
        for jj in range(self.num_obstacles):
            if np.linalg.norm(np.array([x-self.obstacle_list[jj].state[0],y-self.obstacle_list[jj].state[1]])) <= self.obstacle_list[jj].radius:
                return True
        return False


    
    