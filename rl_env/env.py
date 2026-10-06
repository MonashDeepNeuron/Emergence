from pathlib import Path

import gymnasium
import mujoco

import numpy as np
from gymnasium import spaces

from rover import Rover, REACHABLE_L, REACHABLE_W, DROP_DIST, BLOCK_HALF

import math

LAYOUT = Path(__file__).parent / "assets" / "layout_train_1.xml"

TILE = 0.2 
MAX_STEPS = 2000

# actions
FORWARD, LEFT, RIGHT, PICK, DROP = range(5)
N_ACTIONS = 5

# rewards
STEP_COST = -0.005
TWD_GOAL = 0.03
PICK_PILE = 0.2
DROP_EMPTY = 0.5 
DROP_FAR = 0.5 
FAILED = -0.05   
ALL_FILLED = 10

# helppeeerrsss


# get a area's center and 1/2 l 1/2 w
def _rect(model, name):
    geom_feature = model.geom(name)
    return geom_feature.pos[:2].copy(), geom_feature.size[:2].copy()



# checks if a point is inside a area or points
def _is_inside(xy, centers, half):
    if not hasattr(xy[0], "__len__"): # single
        return all(abs(point - center) <= h for point, center, h in zip(xy, centers, half))

    # not single
    return [ 
        all(abs(p - center) <= h for p, center, h in zip(point, centers, half))
        for point in xy
    ]



class RoverEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self):

        self.model = mujoco.MjModel.from_xml_path(str(LAYOUT))

        self.data = mujoco.MjData(self.model)

        # areas,  the layout
        self.arena_c, self.arena_h = _rect(self.model, "arena")
        self.pile_c, self.pile_h = _rect(self.model, "material_zone")
        self.pad_c, self.pad_h = _rect(self.model, "target_pad")

        # sort the pad coners, to know which is the far most 
        corners = [(self.pad_c[0] + xx * self.pad_h[0], self.pad_c[1] + yy * self.pad_h[1])
                    for xx, yy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]

        dists = [round (math.dist (corner, self.pile_c), 6) for corner in corners]

        order = sorted(range(len(corners)), key = lambda i: (-dists[i], corners[i][1])) # far first, ties: lower y first
        self.corners = [corners[i] for i in order] # table

        # tile grid over the pad
        self.pad_lowest = self.pad_c - self.pad_h
        self.nx = round(2 * self.pad_h[0] / TILE)
        self.ny = round(2 * self.pad_h[1] / TILE)

        """
        tiles = []
        for i in range(self.nx):  # column
            for j in range(self.ny):  # row
                x = self.pad_lowest[0] + TILE * (i + 0.5)
                y = self.pad_lowest[1] + TILE * (j + 0.5)


                tiles.append((x, y))
        """
        # same as above
        self.tiles = [(self.pad_lowest[0] + TILE * (column + 0.5),
                       self.pad_lowest[1] + TILE * (row + 0.5))
                      for column in range(self.nx)
                      for row in range(self.ny)]

        # how far each tile is from the pile, -> [0, 1]
        d = [
                math.sqrt(sum((t - c) ** 2 for t, c in zip(tile, self.pile_c)))
                for tile in self.tiles
            ]

        
        self.tile_far = [(di - min(d)) / max(max(d) - min(d), 1e-9) for di in d]

        
        self.action_space = spaces.Discrete(N_ACTIONS) # from 0 - (n-1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype = np.float32) # -np.inf, np.inf means the highest and lowest vals
        # 28 == 2 + 2 + 1 + 1 + 2 + 8 + 1 + 3 + 3 + 5
        # corr x y + cos sin + is carry + is reach + pile cent + coners + on_pad + nearest block + far most tile + prev action

    def _blocks(self):
        # corr of blocks not carried

        b = self.rover.blocks

        if self.rover.carrying is not None: # None or num the id
            b = b[b != self.rover.carrying]

        return self.data.xpos[b, :2]

    def _tile_index(self, xy):
        # TODO: 
        indices = []
        for x, y in xy:
            i = math.floor((x - self.pad_lowest[0]) / TILE)   # column, counted from the pad's lowest corner
            j = math.floor((y - self.pad_lowest[1]) / TILE)   # row
            if 0 <= i < self.nx and 0 <= j < self.ny:
                indices.append(i * self.ny + j)               # position in self.tiles
            else:
                indices.append(-1)                            # off the pad
        return indices

    def _filled(self):
        """check fill"""
        idx = self._tile_index(self._blocks())

        filled = [False] * len(self.tiles)

        for tile_number in idx:
            if tile_number >= 0:
                filled[tile_number] = True

        return filled

    def _dist_to_goal(self):
        """Distance from the rover to its goal, the pile when no carry, the pad when carrying."""

        goal = self.pad_c if self.rover.carrying is not None else self.pile_c
        return math.dist((self.rover.x, self.rover.y), goal)

    def _get_observation(self):
        # TODO
        rover = self.rover

        # 2
        rover_position = [(rover.x - self.arena_c[0]) / self.arena_h[0], (rover.y - self.arena_c[1]) / self.arena_h[1]]

        # 2
        rover_heading = [math.cos(rover.yaw), math.sin(rover.yaw)]

        # 1
        is_carrying = float(rover.carrying is not None)

        block_positions = self._blocks()
        blocks_ahead, blocks_side = rover.dist_to_self(block_positions)
        block_in_reach = float(any(
                                        0 < ahead <= REACHABLE_L and abs(side) <= REACHABLE_W
                                        for ahead, side in zip(blocks_ahead, blocks_side)
                                    )
                                )


        view_range = rover.view_l
        is_on_pad = float(_is_inside((rover.x, rover.y), self.pad_c, self.pad_h))

        block_is_visible = rover.see_dist(block_positions)
        visible_blocks = [(ahead, side)
                          for ahead, side, visible in zip(blocks_ahead, blocks_side, block_is_visible)
                          if visible]
        if visible_blocks:
            nearest_ahead, nearest_side = min(visible_blocks,
                                              key = lambda block: math.sqrt(block[0] ** 2 + block[1] ** 2))
            nearest_block = [nearest_ahead / view_range, nearest_side / view_range, 1.0]
        else:
            nearest_block = [0.0, 0.0, 0.0]

        # empty tile in the camera cone that is farthest from the pile
        tile_visi_empty = [(not filled) and visible
                           for filled, visible in zip(self._filled(), rover.see_dist(self.tiles))]
        
        indices = [tile_number for tile_number, ok in enumerate(tile_visi_empty) if ok] # positions of every True value in the list
        if indices:
            far_tile = max(indices, key = lambda tile_number: self.tile_far[tile_number])
            
            ahead, side = rover.dist_to_self(self.tiles[far_tile])

            farthest_empty_tile = [ahead / view_range, side / view_range, 1.0]
        else:
            farthest_empty_tile = [0.0, 0.0, 0.0]

        last_action = np.zeros(N_ACTIONS)
        if self.prev_action is not None:
            last_action[self.prev_action] = 1.0


        pile_center_rela = np.array(rover.dist_to_self(self.pile_c)) / view_range
        pad_corners_rela = np.stack(rover.dist_to_self(self.corners), axis=1).ravel() / view_range
        
        return np.array(rover_position + rover_heading + [is_carrying, block_in_reach] + list(pile_center_rela) + \
                list(pad_corners_rela) + [is_on_pad] + nearest_block + farthest_empty_tile + list(last_action),
                dtype = np.float32)
    # cld
    def reset(self, seed = None, options = None):
            super().reset(seed = seed)
            mujoco.mj_resetData(self.model, self.data)
            self.rover = Rover(self.model, self.data)
            self.steps = 0
            self.prev_action = None
            self.last_dist = self._dist_to_goal()
            return self._get_observation(), {}

    def step(self, action):
        # TODO
        roby = self.rover
        reward = STEP_COST
        action = int(action)

        if action == FORWARD:
            old = (roby.x, roby.y)
            roby.forward()
            if not _is_inside((roby.x, roby.y), self.arena_c, self.arena_h):
                roby.x, roby.y = old # blocked by the arena edge
                roby._sync()
                reward += FAILED

        if action == LEFT: roby.turn_left()

        if action == RIGHT: roby.turn_right()

        if action == PICK:
            block_positions = self._blocks()
            blocks_ahead, blocks_side = roby.dist_to_self(block_positions)
            in_reach = [(ahead, (x, y))
                        for ahead, side, (x, y) in zip(blocks_ahead, blocks_side, block_positions)
                        if 0 < ahead <= REACHABLE_L and abs(side) <= REACHABLE_W]

            if roby.carrying is not None or not in_reach:
                reward += FAILED 
            else:
                _, target = min(in_reach)
                if _is_inside(target, self.pad_c, self.pad_h):
                    reward += FAILED # it's on pad
                else:
                    roby.pick()
                    if _is_inside(target, self.pile_c, self.pile_h):
                        reward += PICK_PILE

        if action == DROP:
            if roby.carrying is None:
                reward += FAILED
            else:
                land = np.array([[roby.x + DROP_DIST * np.cos(roby.yaw),
                                roby.y + DROP_DIST * np.sin(roby.yaw)]])
                tile_index = self._tile_index(land)[0]

                # check before dropping: only drop into an empty tile on the pad
                if tile_index < 0 or self._filled()[tile_index]:
                    reward += FAILED
                else:
                    block = roby.carrying
                    roby.drop()
                    roby._place(block, *self.tiles[tile_index], BLOCK_HALF) # TELEPORT
                    mujoco.mj_forward(self.model, self.data)
                    reward += DROP_EMPTY + DROP_FAR * self.tile_far[tile_index]

        # reward for getting closer to the goal, penalty for moving away
        dist = self._dist_to_goal()
        reward += TWD_GOAL * (self.last_dist - dist)
        self.last_dist = dist

        self.steps += 1
        self.prev_action = action

        filled = self._filled()

        terminated = False
        if all(filled):
            reward += ALL_FILLED
            terminated = True
        elif roby.carrying is None and not any(_is_inside(self._blocks(), self.pile_c, self.pile_h)):
            terminated = True                            # pile has run out
        truncated = not terminated and self.steps >= MAX_STEPS

        info = {"filled_tiles": sum(filled)}
        return self._get_observation(), float(reward), terminated, truncated, info