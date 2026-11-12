import numpy as np
import mujoco
import gymnasium
from gymnasium import spaces
from pathlib import Path

import math

from rover import Rover, STEP, TURN, REACHABLE_LEN, REACHABLE_ANGLE, DROP_DIST, BLOCK_HALF_LEN, ROBOT_HALF_LEN, ROBOT_HALF_HEIGHT, VIEWABLE_LEN, VIEWABLE_ANGLE


LAYOUT = Path(__file__).parent / "assets" / "layout_train_1.xml"

MAX_STEPS = 2000

# actions
N_ACTIONS = 5
#  0      1      2      3     4
FORWARD, LEFT, RIGHT, PICK, DROP = range(5)

# rewards
STEP_COST = -0.005
STEP_OUT_ARENA = -0.05
PICK_MATERIAL_FAIL = -0.1
PICK_MATERIAL_SUCCESS = 0.1
DROP_MATERIAL_FAIL = -0.1
DROP_MATERIAL_SUCCESS = 0.1
COMPLETE = 10.0
# ... the rest is still in todo


def _rect(model, name) -> tuple[tuple[float, float], tuple[float, float]]:
    geom_feature = model.geom(name)

    # <geom name="target_pad" class="marker" pos="1.8 0 0.002" size="0.5 0.3 0.001"/>
    return geom_feature.pos[:2].copy(), geom_feature.size[:2].copy()

# check if a point is inside a rectangle
def _check_is_inside(corr: tuple[float, float], center: tuple[float, float], half: tuple[float, float]) -> bool:
    return (center[0] - half[0]) <= corr[0] <= (center[0] + half[0]) and (center[1] - half[1]) <= corr[1] <= (center[1] + half[1])
    


class RoverEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__() # because we inherented from gym

        self.model = mujoco.MjModel.from_xml_path(str(LAYOUT)) # type: ignore
        self.data = mujoco. MjData(self.model) # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore  # compute positions once

        self.roby = Rover(self.model, self.data) # hiii~~~
        self.step_count = 0

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape = (19,), dtype = np.float32)
        self.steps = 0
        self.prev_action = None

        self.roby = Rover(self.model, self.data)

        # the positions
        # <geom name="arena" class="bounds" pos="0 0 0.001" size="3 3 0.001"/>
        self.arena_info = _rect(self.model, "arena") # tuple(position(x, y), half_size(len, width))

        # <geom name="material_zone" class="stash" pos="-1.8 0 0.002" size="0.9 0.8 0.001"/>
        self.material_zone_info = _rect(self.model, "material_zone")

        # <geom name="target_pad" class="marker" pos="1.8 0 0.002" size="0.5 0.3 0.001"/>
        self.target_pad_info = _rect(self.model, "target_pad")
        self.target_cell_len = BLOCK_HALF_LEN * 2
        self.target_grid = {}

        self.materials = [m for m in range(self.model.nbody)
                          if self.model.body(m).name.startswith("mat_")] # the ids

        target_cell_x0 = self.target_pad_info[0][0] - self.target_pad_info[0][1] + BLOCK_HALF_LEN
        target_cell_y0 = self.target_pad_info[1][0] - self.target_pad_info[1][1] + BLOCK_HALF_LEN

        max_x = round(2 * self.target_pad_info[0][0] / self.target_cell_len)
        max_y = round(2 * self.target_pad_info[0][1] / self.target_cell_len)

        self.target_grid_size = (max_x, max_y)     

        # generate the grid corr
        for i in range(max_x):
            for j in range(max_y):
                self.target_grid[(i, j)] = (
                    target_cell_x0 + i * self.target_cell_len,
                    target_cell_y0 + j * self.target_cell_len
                )

        # use sets to store what aterials has been used and which grod has alreasdy occupied
        self.occupied = set()
        self.placed_materials = set()



    def _check_seeable(self, ahead: float, side: float) -> bool:
        return math.sqrt(ahead ** 2 + side ** 2) <= VIEWABLE_LEN and abs(math.atan2(side, ahead)) <= VIEWABLE_ANGLE / 2

    def _camera(self) -> list[tuple[int, float, float]]:
        # this is very much a simulated version
        roby = self.roby
        cos_yaw, sin_yaw = math.cos(roby.yaw), math.sin(roby.yaw)


        seeable_materials = []
        for material_id in self.materials:
            if material_id == roby.carry: continue
            if material_id in self.placed_materials: continue

            mat_pos = self.data.xpos[material_id]
            # <body name="mat" pos="-2.6 -0.6 0.1"><freejoint/><geom class="material"/></body>
            temp_x = mat_pos[0] - roby.x 
            temp_y = mat_pos[1] - roby.y

            ahead = temp_x * cos_yaw + temp_y * sin_yaw
            side = -temp_x * sin_yaw + temp_y * cos_yaw
            if self._check_seeable(ahead, side):
                seeable_materials.append((material_id, ahead, side))

        return seeable_materials


    def _get_target_cell_corr(self, land_x: float, land_y: float) -> None | tuple[int, int]:
        """
        find the target grid cell containts the landin position
        """
        target_center = self.target_pad_info[0]
        target_half_lens = self.target_pad_info[1]

        if not _check_is_inside((land_x, land_y), target_center, target_half_lens): return None

        corner_edge_x0 = target_center[0] - target_half_lens[0]
        corner_edge_y0 = target_center[1] - target_half_lens[1]

        i = int((land_x - corner_edge_x0) / self.target_cell_len)
        j = int((land_x - corner_edge_y0) / self.target_cell_len)

        i = min(i, round(2 * target_half_lens[0] / self.target_cell_len) - 1)
        j = min(j, round(2 * target_half_lens[1] / self.target_cell_len) - 1)

        cell_corr = (i, j)

        if cell_corr not in self.target_grid: return None

        return cell_corr

    def _get_observation(self):
        """
        roby's position, x and y coordinates, (float, float)
        heading direction, cos and sin (float, float)
        carrying material or not, bool
        check if there is a material within reach
        the relative position between the material pile center and the robot, (float, float)
        the relative position between the tagte location and the robot, (float, float)
        nearest seen material, ahead, side, seeable or not seeable, (float, float, bool)
        ready to drop? bool
        what is the last action, int
        target grid occupancy, 15 values
        completion ratio, float
        """
        roby = self.roby
        arena_center = self.arena_info[0]
        arena_half_lens = self.arena_info[1]
        material_zone_center = self.material_zone_info[0]
        # material_zone_half_lens = self.material_zone_info[1]
        target_pad_center = self.target_pad_info[0]
        # target_pad_half_lens = self.target_pad_info[1]


        roby_position = [(roby.x - arena_center[0]) / arena_half_lens[0], (roby.y - arena_center[1]) / arena_half_lens[1]]
        roby_heading_direction = [math.cos(roby.yaw), math.sin(roby.yaw)]
        roby_carrying = 1.0 if roby.carry != None else 0.0

        seeable_materials = self._camera()
        check_reachable_material = any(roby._check_reachable(ahead, side) != None for _, ahead, side in seeable_materials)
        check_reachable_material = 1.0 if check_reachable_material == True else 0.0

        roby_material_zone_center = [(roby.x - material_zone_center[0]) / arena_center[0], (roby.y - material_zone_center[1]) / arena_center[1]]
        roby_target_pad_center = [(roby.x - target_pad_center[0]) / arena_center[0], (roby.y - target_pad_center[1]) / arena_center[1]]

        nearest_material = [0.0, 0.0, 0.0]
        if seeable_materials:
            _, ahead, side = min(seeable_materials, key = lambda material: material[1] ** 2 + material[2] ** 2)
            nearest_material = [ahead / VIEWABLE_LEN, side / VIEWABLE_LEN, 1.0]


        # check if the current planning frop material position has been occupied
        check_cell_not_occupied = 0.0

        if roby.carry != None:
            land_x, land_y = roby.get_drop_position()
            cell = self._get_target_cell_corr(land_x, land_y)

            if cell != None and cell not in self.occupied:
                check_cell_not_occupied = 1.0

        prev_action_ = -1.0 if self.prev_action == None else self.prev_action / (N_ACTIONS - 1)

        target_grid_occupancy = [ 1.0 if (i, j) in self.occupied else 0.0 for i in range(self.target_grid_size[0]) for j in range(self.target_grid_size[1])]           

        completion_ratio = len(self.occupied) / len(self.target_grid)

        return np.array(
            roby_position + 
            roby_heading_direction + 
            [roby_carrying, check_reachable_material] + 
            roby_material_zone_center + 
            roby_target_pad_center + 
            nearest_material + 
            [check_cell_not_occupied, prev_action_] + 
            target_grid_occupancy + [completion_ratio], dtype = np.float32)
        

    def step(self, action):
        """Apply one action; return (observation, reward, terminated, truncated, info)."""

        # FORWARD, LEFT, RIGHT, PICK, DROP = range(5)

        self.step_count += 1
        reward = STEP_COST

        action = int(action)

        if action == FORWARD:
            old_x, old_y = self.roby.x, self.roby.y

            self.roby.forward()
            if not _check_is_inside((self.roby.x, self.roby.y), self.arena_info[0], self.arena_info[1]):
                self.roby.x, self.roby.y = old_x, old_y
                self.roby._sync()

                reward += STEP_OUT_ARENA

        if action == LEFT:
            self.roby.turn_left()

        if action == RIGHT:
            self.roby.turn_right()

        if action == PICK:
            seeable_materials = self._camera()
            picked_material = self.roby.pick(seeable_materials)

            if picked_material == None:
                reward += PICK_MATERIAL_FAIL
                return 

            reward += PICK_MATERIAL_SUCCESS

        if action == DROP:
            if self.roby.carry == None:
                reward += DROP_MATERIAL_FAIL
            else:
                land_x, land_y = self.roby.get_drop_position()
                target_cell_corr = self._get_target_cell_corr(land_x, land_y)

                if target_cell_corr == None or target_cell_corr in self.occupied:
                    reward += DROP_MATERIAL_FAIL
                else:
                    cell_x, cell_y = self.target_grid[target_cell_corr]
                    dropped_material = self.roby.carry
                    self.roby.drop(cell_x, cell_y)

                    self.occupied.add(target_cell_corr)
                    self.placed_materials.add(dropped_material)

                    reward += DROP_MATERIAL_SUCCESS

        self.prev_action = action

        check_terminate = len(self.occupied) == len(self.target_grid)
        check_truncate = self.steps >= MAX_STEPS

        if check_terminate: reward += COMPLETE

        observation = self._get_observation()

        info = {
            "steps": self.steps,
            "occupied_cells": len(self.occupied),
            "total_cells": len(self.target_grid),
            "completion_ratio": len(self.occupied) / len(self.target_grid),
            "carrying": self.roby.carry is not None
        }

        return observation, reward, check_terminate, check_truncate, info
            

    def reset(self, seed = None, options = None):
        """
        Put the world back to the start; return (observation, info).
        this part is from chat
        """
        super().reset(seed=seed)

        # restore all MuJoCo bodies to their initial positions
        mujoco.mj_resetData(self.model, self.data) # type: ignore
        mujoco.mj_forward(self.model, self.data) # type: ignore

        self.steps = 0
        self.prev_action = None

        self.occupied = set()
        self.placed_materials = set()

        self.roby = Rover(self.model, self.data)

        return self._get_observation(), {}