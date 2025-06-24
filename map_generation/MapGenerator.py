import numpy as np
from . import Obstacle
from . import Map
import math
from pprint import pprint 


# np.random.seed(12345678)

def n_rocks_geq_d_exponential(cfa,d):
    """_summary_

    Args:
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    number_of_rocks_per_sqm = cfa*np.exp((1.79 + (0.152/cfa))*d)
    return number_of_rocks_per_sqm

def surveyor_1_cumulative_number_distribution(diameter_m: float):
    """ The number of rocks larger than a given parameter per the Surveyor 1 distribution

    :param diameter_m: the diameter of the rocks in use
    :type diameter: float
    :return: the number of rocks with diameter larger than diameter
    :rtype: float
    """

    number_of_rocks_per_sqm = 0.00234*diameter_m**(-2.11)
    return number_of_rocks_per_sqm

def min_diameter_from_height(height):
    return height/1.2

def height_from_diameter(diameter):
    # cfr Ben Hockman's email
    return max(0., np.random.normal(0.6, 0.28))*diameter

def rock_density_between_these_diameters(lower_diameter_m: float, upper_diameter_m: float, use_surveyor_power_law: bool= True, cfa: float=0):
    """The number of rocks between two given diameters per the Surveyor 1 distribution

    :param lower_diameter_m: the smallest diameter to consider, in m
    :type lower_diameter: float
    :param upper_diameter_m: the largest diameter to consider, in m
    :type upper_diameter: float
    :param use_surveyor_power_law: whether to use the Surveyor power law (default) or the exponential CFA law to generate , defaults to True
    :type use_surveyor_power_law: bool, optional
    :param cfa: the CFA to use. Ignored if use_surveyor_power_law is True, defaults to 0 
    :type cfa: float, optional
    :return: number of rocks
    :rtype: int
    """

    if use_surveyor_power_law:
        num_rocks_larger_than_smallest = surveyor_1_cumulative_number_distribution(lower_diameter_m)
        num_rocks_larger_than_largest = surveyor_1_cumulative_number_distribution(upper_diameter_m)
    else:
        num_rocks_larger_than_smallest = n_rocks_geq_d_exponential(cfa, lower_diameter_m)
        num_rocks_larger_than_largest = n_rocks_geq_d_exponential(cfa, upper_diameter_m)
        print("CFA: there are {} rocks under {} m and {} rocks under {} m".format(
            num_rocks_larger_than_smallest,
            lower_diameter_m,
            num_rocks_larger_than_largest,
            upper_diameter_m
        ))
    rock_density = num_rocks_larger_than_smallest-num_rocks_larger_than_largest

    return rock_density

class MapGenerator():
    def __init__(self,
                cfa,
                x_limits,
                y_limits,
                verbose: bool= False,
                ):
        """_summary_

        Args:
            Formation (_type_): _description_
            y_limits (_type_): _description_
            terrain_specs (bool, optional): _description_. Defaults to True.
            Document: 
                https://gateway.jpl.nasa.gov/sites/cadre/_layouts/15/WopiFrame2.aspx?sourcedoc={cba2d599-155c-458a-9031-e4e052ec9edf}&action=view&wdAccPdf=0&wdparaid=6349DD78

                Matlab Code: 
            https://gateway.jpl.nasa.gov/sites/cadre/
            Shared%20Documents/Forms/
            AllItems.aspx?RootFolder=%2Fsites%2Fcadre%2FShared%20Documents%2F08%5FMobility%2FTools&
            FolderCTID=0x012000101B4183F7FAF24697CDA5559D2D0458&View=%7B714A1D5E%2DCD8F%2D4A4D%2DA6D6%2D10CA49D4744D%7D

        """

        self.cfa = cfa
        self.x_limits = x_limits
        self.y_limits = y_limits

        self.obstacle_list = []
        self.crater_list = []

        self.area = (x_limits[1] - x_limits[0])*(y_limits[1]-y_limits[0])
        self.terrain_specs = {}
        # TODO define which ranges of rocks we are interested in and what is the 
        # self.terrain_specs['rocks'] = {"num":[], "radius_range":[np.array([0.083,0.125])/2]}
        self.verbose = verbose

    def verbprint(self, *messages):
        if self.verbose:
            print(*messages)

    def create_terrain_map(self,cfa: float=0.02, use_surveyor_power_law: bool= True, ground_clearance_m: float = 0.05, seed: float=None, crater_diameters: list=None, crater_densities: list=None, crater_avoid_other_craters=False, crater_avoid_other_obstacles=False, locations_to_avoid: list=None, max_rock_size: float = 10):

        # Obstacles
        self._populate_rock_distribution(cfa, use_surveyor_power_law, ground_clearance_m, max_rock_size)
        self._initialize_obstacles_valid_terrain_spec(ground_clearance_m=ground_clearance_m, seed=seed,locations_to_avoid=locations_to_avoid)

        # Craters
        if crater_diameters and crater_densities:
            self._populate_crater_distribution(crater_diameters, crater_densities)
            self._initialize_craters_valid_terrain_spec(
                seed=seed,
                avoid_craters=crater_avoid_other_craters,
                avoid_obstacles=crater_avoid_other_obstacles,
                obstacles_to_avoid=self.obstacle_list,
                locations_to_avoid=locations_to_avoid) # Avoid other obstacles

        return Map.Map(obstacle_list=self.obstacle_list + self.crater_list,cfa=cfa,x_limits=self.x_limits,y_limits=self.y_limits) # return a dummy Map class as defined in Map.py

    # def update_obstacle_space(self):
    #     self.obstacle_space = self.obstacle_list 

    def _populate_rock_distribution(self, cfa, use_surveyor_power_law, ground_clearance_m: float, max_rock_size_m: float= 10., height_stops: list=None):
        if height_stops is None:
            height_stops = np.logspace(np.log10(ground_clearance_m), np.log10(max_rock_size_m),num=20)
        diameter_ranges = [(min_diameter_from_height(height_stops[i]), min_diameter_from_height(height_stops[i+1])) for i in range(len(height_stops)-1)]
        self.rock_population_density_by_diameter = {
            dr: rock_density_between_these_diameters(dr[0], dr[1], use_surveyor_power_law=use_surveyor_power_law, cfa=cfa) for dr in diameter_ranges
        }

    def _initialize_obstacles_valid_terrain_spec(self, ground_clearance_m, seed:int=None, locations_to_avoid: list=None):
        # all subsequent obstacles need to check intersection with
        # current obstacles

        # Create range of radii we care about

        if seed != None:
            np.random.seed(seed)

        for diameter_range, rock_density in self.rock_population_density_by_diameter.items():
            number_of_rocks = round(rock_density*self.area)
            # print("There are {} rocks in diameter range {}-{}m".format(number_of_rocks, min(diameter_range), max(diameter_range)))
            counter = 0 
            while counter < number_of_rocks:
                x_obs = np.random.uniform(self.x_limits[0], self.x_limits[1])
                y_obs = np.random.uniform(self.y_limits[0], self.y_limits[1])
                # obstacle_radius = np.random.uniform(low=min(diameter_range)/2., high=max(diameter_range)/2.) 
                obstacle_radius = max(diameter_range)/2.
                obstacle_height = height_from_diameter(2*obstacle_radius)
                if obstacle_height<ground_clearance_m:
                    counter+=1 # There was a rock, it was just too short
                    continue
                other_obs = Obstacle.Obstacle(radius=obstacle_radius,location=np.array([x_obs,y_obs,0,0,0,0]))

                # No intersection with start and goal locations stored in `locations_to_avoid`
                if locations_to_avoid and self.check_obstacle_obstacle_intersection(other_obs, locations_to_avoid):
                    self.verbprint("Found locations_to_avoid with obstacles=", x_obs, y_obs, obstacle_radius)
                    continue

                # No intersection
                # Add object
                if self.check_obstacle_obstacle_intersection(other_obs, self.obstacle_list) == False:
                    self.obstacle_list.append(other_obs)
                    self.verbprint("Obstacles added=", x_obs,y_obs, obstacle_radius)
                    counter+=1
                # Intersection
                # Keep sampling
                else:
                    continue

    def _populate_crater_distribution(self, diameters: list=None, densities: list=None):
        self.crater_population_density_by_diameter = {key: value for key, value in zip(diameters, densities)}
    
    def _initialize_craters_valid_terrain_spec(self, seed:int=None, resolution = 0.01, avoid_obstacles=False, avoid_craters=False, obstacles_to_avoid: list=None, locations_to_avoid: list=None):
        # obstacles and craters may need to check intersection with current obstacles

        # Create range of radii we care about
        if seed != None:
            np.random.seed(seed)


        for diameter, crater_density in self.crater_population_density_by_diameter.items():
            number_of_craters = round(crater_density*math.sqrt(self.area))
            # print("There are {} craters with diameter {}m".format(number_of_craters, diameter))
            counter = 0 
            while counter < number_of_craters:
                x_crater = np.random.uniform(self.x_limits[0], self.x_limits[1])
                y_crater = np.random.uniform(self.y_limits[0], self.y_limits[1])
                crater_radius = diameter/2.
                other_craters = Obstacle.Obstacle(radius=crater_radius,location=np.array([x_crater,y_crater,0,0,0,0]))

                if locations_to_avoid and self.check_obstacle_obstacle_intersection(other_craters, locations_to_avoid):
                    self.verbprint("Found locations_to_avoid with crater=", x_crater,y_crater)
                    continue

                # No intersection
                # Add object
                if avoid_obstacles or avoid_craters:
                    # Other obstacles
                    if avoid_obstacles and self.check_obstacle_obstacle_intersection(other_craters, obstacles_to_avoid):
                        continue
                    # Other craters
                    if avoid_craters and self.check_obstacle_obstacle_intersection(other_craters, self.crater_list):
                        continue
                
                self.verbprint("Obstacles added=", x_crater,y_crater, crater_radius)
                self.crater_list.append(other_craters)
                counter+=1

    # def convert_to_grid_frame(self, x, y, limit_x: int, limit_y: int , resolution):
    #     x_in_grid = (x/resolution + limit_x/2)*resolution
    #     y_in_grid = (y/resolution + limit_y/2)*resolution

    #     return x_in_grid, y_in_grid

    def _initialize_rover_target_locations(self, x_list: list=None, y_list: list=None, diameter: float=None):

        rover_target_locationss = []
        # The x_list and y_list are in the world frame wheras the list of obstacles that we will use to compare is in the image frame.
        # A conversion here is necessary
        for x,y in zip(x_list, y_list):
            self.verbprint("Added avoid location at {} {} with diameter {}".format(x,y,diameter))
            target_location = Obstacle.Obstacle(radius=0.5*diameter,location=np.array([x,y,0,0,0,0]))
            rover_target_locationss.append(target_location)
        return rover_target_locationss

    def check_obstacle_obstacle_intersection(self,target_obstacle, obstacle_list: list=None):
        for i in range(len(obstacle_list)):
            other_obs_pos = obstacle_list[i].state[0:2]
            other_obs_radius = obstacle_list[i].radius  
            if np.linalg.norm(target_obstacle.state[0:2]-other_obs_pos) <= \
                target_obstacle.radius + other_obs_radius: #ß + 2*self.Formation.agents_list[self.Formation.leader_idx].radius:
                return True
        return False 

if __name__ == "__main__":


    resolution = 0.01
    map_width = 5
    map_height = 6
    rover_width = .065
    rover_length = 0.09

    x_start = -.5
    y_start = 0.5
    goal_x_m = 1.5
    goal_y_m = 1.0

    mg = MapGenerator(cfa = 0.02, x_limits = (0,map_width), y_limits=(0,map_height))

    # Craters
    crater_diameters = [0.65, 1.35]
    crater_densities = [0.15, 0.05]

    # # Obstacles
    # height_ranges = [
    #     (0.01, 0.02),
    #     (0.02, 0.03),
    #     (0.03, 0.05),
    #     (0.05, 0.07),
    #     (0.07, 0.1),
    #     (0.1, 0.15),
    # ]

    # height_stops = [hr[0] for hr in height_ranges]
    # height_stops.append(height_ranges[-1][1])

    # # Obstacles
    # mg._populate_rock_distribution(cfa=0.02, use_surveyor_power_law=True, ground_clearance_m=0.05, height_stops=height_stops)
    # mg._initialize_obstacles_valid_terrain_spec()

    # map = mg.create_terrain_map(crater_diameters=crater_diameters, crater_densities=crater_diameters)
    map = mg.create_terrain_map()
    grid = map.generate_grid(resolution=.01,
        map_file_name="just_grid.png")

    # print(grid)
    # print(sum(grid>0))
    # print(len(grid))
    # print("CFA: {}%".format(100*sum(grid>0)/len(grid)))