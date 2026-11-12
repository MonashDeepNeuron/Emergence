import math

import mujoco


# the robot is 0.6 long * 0.4 wide * 0.16 tall
STEP = 0.2 # bc of the block width
TURN = math.pi / 2
REACHABLE_LEN = 0.55 # 0.55 - 0.6 / 2 = 0.25 ahead of the face
REACHABLE_ANGLE = math.pi / 2
DROPABLE_LEN = REACHABLE_LEN # max distance, not a fixed one (same idea as REACHABLE_LEN)
BLOCK_HALF_LEN = 0.1 # half a block size, the height a placed block sits at otherwise it goes under the floor
ROBOT_HALF_LEN = 0.3
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

        body_id = model.body(name).id # global body index
        self.mocap_id = model.body_mocapid[body_id] # mocap-only index

        if self.mocap_id != -1:
            self.x, self.y, self.z = data.mocap_pos[self.mocap_id][:3]
        else:
            raise ValueError("ERROR: need to config the rover to mocap = true")


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
        

    def pick(self, seeable_materials: list[tuple[int, float, float]]) -> None | int:
        if self.carry != None: return None

        target_material = (-1, math.inf)
        for material_id, ahead, side in seeable_materials:
            maybe_dist = self._check_reachable(ahead, side)
            if self._check_reachable(ahead, side):
                dist = maybe_dist
                if dist is not None and dist < target_material[1]:
                    target_material = (material_id, dist)

        if target_material[0] == -1: return None

        self.carry = target_material[0]
        self._sync()

        return self.carry

    def drop(self, land_x: float, land_y: float) -> None | tuple[float, float]:
        if self.carry is None:
            return None

        # teleport the material onto the landing spot, upright and still
        # the rest of the part all from chatGPT
        joint_id = self.model.body_jntadr[self.carry]
        qpos_id = self.model.jnt_qposadr[joint_id]
        qvel_id = self.model.jnt_dofadr[joint_id]

        self.data.qpos[qpos_id:qpos_id + 7] = [
            land_x, land_y, BLOCK_HALF_LEN, 1, 0, 0, 0
        ]
        self.data.qvel[qvel_id:qvel_id + 6] = 0

        self.carry = None
        self._sync()

        return land_x, land_y

    # _sync is from claude. basically for viewer and other updates
    def _sync(self):
        """Write the rover's pose (and the carried block) into MuJoCo, then update it."""
        self.data.mocap_pos[self.mocap_id] = [self.x, self.y, self.z]
        self.data.mocap_quat[self.mocap_id] = [
            math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)
        ]

        if self.carry is not None:   # carried block rides on the roof
            joint_id = self.model.body_jntadr[self.carry]
            qpos_id = self.model.jnt_qposadr[joint_id]
            qvel_id = self.model.jnt_dofadr[joint_id]

            self.data.qpos[qpos_id:qpos_id + 7] = [
                self.x, self.y, 2 * ROBOT_HALF_HEIGHT + BLOCK_HALF_LEN,
                1, 0, 0, 0
            ]
            self.data.qvel[qvel_id:qvel_id + 6] = 0

        mujoco.mj_forward(self.model, self.data) # type: ignore

    def __repr__(self):
        return f"{self.name}: position: {self.x, self.y, self.z}, carrying?: {self.carry}"