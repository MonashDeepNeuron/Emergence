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
DROP_DIST = 0.4
BLOCK_HALF = 0.1
Z = 0.08
VIEW_L = 2.0            # how far the camera can see (m)
VIEW_ANGLE = np.radians(90) # camera field of view: 45 deg to each side

class Rover:
    def __init__(self, model, data, name = "rover", yaw = np.pi / 2):
        self.model = model   # the world we built, the layout and rover
        self.data = data # curr state of the world

        body = model.body(name)

        self.mocap = model.body_mocapid[body.id]

        self.x, self.y = model.body_pos[body.id][:2]
        self.z = model.geom("chassis").size[2]

        self.yaw = yaw
        self.view_l = VIEW_L
        self.view_angle = VIEW_ANGLE
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


    # sensing
    def dist_to_self(self, xy):
        """World xy (one point or an array of points) -> (ahead, side) from the rover.
        Positive side = left."""
        xy = np.asarray(xy, dtype=float)
        dx = xy[..., 0] - self.x
        dy = xy[..., 1] - self.y
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return dx * c + dy * s, -dx * s + dy * c

    def see_dist(self, xy):
        """True for points within VIEW_L and inside the view_angle cone in front."""
        ahead, side = self.dist_to_self(xy)
        dist = np.hypot(ahead, side)
        angle = np.arctan2(side, ahead)
        return (dist <= self.view_l) & (np.abs(angle) <= self.view_angle / 2)


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
    model = mujoco.MjModel.from_xml_path(str(LAYOUT))
    data = mujoco.MjData(model)

    rover = Rover(model, data)