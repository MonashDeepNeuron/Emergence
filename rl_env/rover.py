"""
Placeholder rover: forward, turn_left, turn_right, pick, drop. Nothing else.
Moving teleports the rover. Picking/dropping teleports a material block.
"""

import numpy as np
import mujoco
from pathlib import Path

LAYOUT = Path(__file__).parent / "assets" / "layout.xml"

STEP = 0.1
TURN = np.radians(90)
REACHABLE_L = 0.5
REACHABLE_W = 0.2
DROP_DIST = 0.45
BLOCK_HALF = 0.1
Z = 0.08

class Rover:
    def __init__(self, model, data, name = "rover", yaw = np.pi / 2):
        self.model = model   # the world we built, the layout and rover
        self.data = data # curr state of the world

        body = model.body(name)

        self.mocap = model.body_mocapid[body.id]

        self.x, self.y = model.body_pos[body.id][:2]
        self.z = model.geom("chassis").size[2]

        self.yaw = yaw
        self.carrying = None
        self.blocks = np.array([b for b in range(model.nbody)
                                if model.body(b).name.startswith("mat_")])
        self._sync() # push the starting pose into the simulation


    # 5 actions: forward, turn_left, turn_right, pick up, drop
    def forward(self):
        self.x += STEP * np.cos(self.yaw)
        self.y += STEP * np.sin(self.yaw)
        self._sync()

    def turn_left(self):
        self.yaw += TURN
        self._sync()

    def turn_right(self):
        self.yaw -= TURN
        self._sync()

    def pick(self):
        # pikcs up the closest block in the front and reaturn true on success
        if self.carrying is not None:
            return False

        # dx and dy all lists
        dx = self.data.xpos[self.blocks, 0] - self.x
        dy = self.data.xpos[self.blocks, 1] - self.y
        ahead = dx * np.cos(self.yaw) + dy * np.sin(self.yaw)
        side = -dx * np.sin(self.yaw) + dy * np.cos(self.yaw)

        in_front = (ahead > 0) & (ahead <= REACHABLE_L) & (np.abs(side) <= REACHABLE_W)
        if not in_front.any():
            return False
        
        self.carrying = min(zip(self.blocks[in_front], ahead[in_front]),
                    key = lambda pair: pair[1])[0]

        self._sync()
        return True

    def drop(self):
        if self.carrying is None:
            return False
        self._place(self.carrying,
                    self.x + DROP_DIST * np.cos(self.yaw),
                    self.y + DROP_DIST * np.sin(self.yaw),
                    BLOCK_HALF)
        self.carrying = None

        mujoco.mj_forward(self.model, self.data)
        return True


    # helpppeeersss by claudee
    def _place(self, block, x, y, z):
        """Teleport a block: set its position, reset rotation, stop it moving."""
        joint = self.data.joint(self.model.body_jntadr[block])
        joint.qpos[:] = [x, y, z, 1, 0, 0, 0]
        joint.qvel[:] = 0

    def _sync(self):
        """Write the rover's pose (and the carried block) into the simulation."""
        self.data.mocap_pos[self.mocap] = [self.x, self.y, self.z]
        self.data.mocap_quat[self.mocap] = [np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)]
        
        if self.carrying is not None:   # carried block rides on the roof
            self._place(self.carrying, self.x, self.y, 2 * self.z + BLOCK_HALF)
        mujoco.mj_forward(self.model, self.data)


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(str(LAYOUT))
    d = mujoco.MjData(m)
    rover = Rover(m, d)