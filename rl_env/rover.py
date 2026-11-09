import numpy as np
import mujoco
from pathlib import Path
import math



LAYOUT = Path(__file__).parent / "assets" / "layout.xml"

# the robot is 0.6 long * 0.4 wide * 0.16 tall
STEP = 0.2 # bc of the block width
TURN = math.pi / 2
REACHABLE_LEN = 0.5 # 0.5 - 0.6 / 2 = 0.2 ahead of the face
REACHABLE_ANGLE = math.pi / 2
DROP_DIST = 0.4
BLOCK_HALF_LEN = 0.1 # half a block size, the height a placed block sits at otherwise it goes under the floor
ROBOT_HALF_HEIGHT = 0.08
VIEWABLE_LEN = 2.0
VIEWABLE_ANGLE = math.pi / 2

class Rover:
    def __init__(self, model, data, name = "rover", yaw = math.pi / 2, carry = None):
        self.model = model
        self.data = data
        self.name = name
        self.yaw = yaw
        self.carry = carry

        body_id = model.body(name).id
        self.mocap_id = model.body_mocapid[body_id]

        if self.mocap_id != -1:
            self.x, self.y, self.z = data.mocap_pos[self.mocap_id][:3]
        else:
            print("ERROR: need to config the rover to mocap = true")

    def forward(self):
        self.x += STEP * math.cos(self.yaw)
        self.y += STEP * math.sin(self.yaw)
        self._sync()

    def turn_left(self):
        self.yaw += TURN
        self._sync()

    def turn_right(self):
        self.yaw -= TURN
        self._sync()

    def _check_reachable(self, ahead: float, side: float): # the env decides which material to pick
        dist = math.sqrt(ahead ** 2 + side ** 2)
        if dist <= REACHABLE_LEN and abs(math.atan2(side, ahead)) <= REACHABLE_ANGLE / 2:
            return dist
        return None
        

    def pick(self, seen_materials: list[tuple[int, float, float]]):
        if self.carry != None: return None

        target_material = (-1, math.inf)
        for material_id, ahead, side in seen_materials:
            maybe_dist = self._check_reachable(ahead, side)
            if self._check_reachable(ahead, side):
                dist = maybe_dist
                if dist is not None and dist < target_material[1]:
                    target_material = (material_id, dist)

        if target_material[0] == -1: return None

        self.carry = target_material[0]
        self._sync()

        return self.carry

    def drop(self, seen_materials: list[tuple[int, float, float]]):
        if self.carry is None: return None

        for _, ahead, side in seen_materials:
            if abs(ahead - DROP_DIST) < BLOCK_HALF_LEN and abs(side) < BLOCK_HALF_LEN:
                return None

        land_x = self.x + DROP_DIST * math.cos(self.yaw)
        land_y = self.y + DROP_DIST * math.sin(self.yaw)

        # the below joint things are from claude. anyways
        # teleport the material onto the landing spot, upright and still
        joint = self.data.joint(self.model.body_jntadr[self.carry])
        joint.qpos[:] = [land_x, land_y, BLOCK_HALF_LEN, 1, 0, 0, 0]
        joint.qvel[:] = 0

        self.carry = None
        self._sync()
        return land_x, land_y

    # _sync is from claude. bassically for viewer and other updates
    def _sync(self):
        """Write the rover's pose (and the carried block) into MuJoCo, then update it."""
        self.data.mocap_pos[self.mocap_id] = [self.x, self.y, self.z]
        self.data.mocap_quat[self.mocap_id] = [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]

        if self.carry is not None:   # carried block rides on the roof
            joint = self.data.joint(self.model.body_jntadr[self.carry])
            joint.qpos[:] = [self.x, self.y, 2 * ROBOT_HALF_HEIGHT + BLOCK_HALF_LEN, 1, 0, 0, 0]
            joint.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data) # type: ignore

    def __repr__(self):
        return f"{self.name}: position: {self.x, self.y, self.z}, carrying?: {self.carry}"