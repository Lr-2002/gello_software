import os
import gymnasium as gym
import mani_skill
from mani_skill.utils.display_multi_camera import display_camera_views
import pinocchio as pin
import glob
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from cv2 import GC_EVAL
import cv2
import numpy as np

from gello.agents.agent import Agent
from gello.robots.dynamixel import DynamixelRobot


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


panda_offset = np.load("./gello_offset.npy")[:-1]
print([*panda_offset, 3.9])
PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
    #     joint_ids=(1, 2, 3, 4, 5, 6, 7),
    #     joint_offsets=(
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         -1 * np.pi / 2 + 2 * np.pi,
    #         1 * np.pi / 2,0a0
    #         1 * np.pi / 2,
    #     ),
    #     joint_signs=(1, 1, 1, 1, 1, 1, 1),
    #     gripper_config=(8, 279, 279 - 50),
    # ),
    # panda
    # "/dev/cu.usbserial-FT3M9NVB": DynamixelRobotConfig(
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NP9G-if00-port0": DynamixelRobotConfig(
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        joint_offsets=[*panda_offset, 3.9],
        joint_signs=(1, 1, 1, 1, 1, -1, 1),
        gripper_config=(7, 107, 65),
    ),
    # best offsets               :  ['6.283', '0.000', '3.142', '6.283', '4.712', '4.712', '4.712']
    # best offsets function of pi: gripper open (degrees)        276.040234375
    # gripper close (degrees)       234.240234375
    # Left UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2 + 0 * np.pi,
            0 * np.pi + np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
    ),
    # Right UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi + 0 * np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
    ),
}


class GelloAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray] = None) -> np.ndarray:
        act = self._robot.get_joint_state()
        return act
        dyna_joints = self._robot.get_joint_state()
        # current_q = dyna_joints[:-1]  # last one dim is the gripper
        current_gripper = dyna_joints[-1]  # last one dim is the gripper

        print(current_gripper)
        if current_gripper < 0.2:
            self._robot.set_torque_mode(False)
            return obs["joint_positions"]
        else:
            self._robot.set_torque_mode(False)
            return dyna_joints


class Panda:
    def __init__(self):
        try:
            from example_robot_data import load

            self.robot = load("panda")
            self.model = self.robot.model
            print("成功从 example-robot-data 加载 Panda 模型。")
        except ImportError:
            print("无法导入 example_robot_data。请确保已安装该库。")
            print("尝试查找 Panda URDF 文件路径...")
            # 如果没有 example-robot-data，你需要提供 Panda URDF 文件的路径
            # 例如: urdf_filename = "/path/to/your/panda.urdf"
            # 这里假设有一个环境变量或已知路径，你需要根据实际情况修改
            try:
                # 这是一个可能的路径，具体取决于你的安装方式
                import os
                from pinocchio.robot_wrapper import RobotWrapper

                # package_dir = "/opt/openrobots/share/example-robot-data/robots/panda_description/"  # 或者其他你的安装路径
                # urdf_filename = package_dir + "urdf/panda.urdf"
                urdf_filename = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/mani_skill/assets/robots/panda/panda_v3.urdf"
                # 使用 RobotWrapper 加载模型，它能处理 mesh 路径
                self.robot = RobotWrapper.BuildFromURDF(
                    urdf_filename, package_dirs=[os.path.dirname(urdf_filename)]
                )
                self.model = self.robot.model
                print(f"成功从 URDF 文件 {urdf_filename} 加载 Panda 模型。")
            except Exception as e:
                print(f"加载 Panda 模型失败: {e}")
                print("请确保 Pinocchio 已正确安装，并且能够找到 Panda 的 URDF 文件。")
                exit()
        self.position_scale = 3
        self.rotation_scale = 3

        # 创建用于计算的数据结构
        self.data = self.model.createData()

    def act(self, q):
        # --- 计算正向运动学 ---
        if q.shape != self.model.nq:
            q = np.array([*q, q[-1]])
        # 运行正向运动学算法
        # 这会更新 data 结构中的所有关节和连杆的位置、速度、加速度等信息 (这里我们只关心位置)
        pin.forwardKinematics(self.model, self.data, q)

        # 更新所有坐标系 (frames) 的位姿
        # forwardKinematics 之后通常需要调用这个来确保 data.oMf 是最新的
        pin.updateFramePlacements(self.model, self.data)

        # 获取末端执行器坐标系 (通常是 'panda_link8' 或 'panda_hand') 的 ID
        # 你需要知道你关心的坐标系的名称，这通常在 URDF 文件中定义
        end_effector_frame_name = "panda_hand_tcp"  # 这是 Panda 的法兰盘 (flange)
        try:
            end_effector_frame_id = self.model.getFrameId(end_effector_frame_name)
            #     f"找到末端执行器坐标系 '{end_effector_frame_name}' (ID: {end_effector_frame_id})"
            # print(
            # )

            # 从 data 中获取末端执行器的位姿 (相对于世界坐标系)
            # data.oMf 是一个列表，存储了所有坐标系的位姿 (Placement)
            # oMf 表示 "Transformation from Frame to World (Origin)"
            end_effector_placement = self.data.oMf[end_effector_frame_id]

            # SE3 对象包含旋转矩阵 (rotation) 和平移向量 (translation)
            position = end_effector_placement.translation
            rotation_matrix = end_effector_placement.rotation
            euler_angles_rpy = pin.rpy.matrixToRpy(rotation_matrix)
            #
            # print("\n--- 计算结果 ---")
            # print(f"末端执行器 ({end_effector_frame_name}) 的位姿:")
            # print(f"  位置 (Position) [x, y, z]: \n{position}")
            # print(f"  姿态 (Orientation) [Rotation Matrix]: \n{rotation_matrix}")
            #
            # 你也可以获取其他坐标系的位姿，例如 panda_link0 (基座)
            # base_frame_id = model.getFrameId("panda_link0")
            # base_placement = data.oMf[base_frame_id]
            # print(f"\n基座坐标系 (panda_link0) 的位姿: \n{base_placement}")

        except ValueError:
            # print(f"错误：在模型中找不到名为 '{end_effector_frame_name}' 的坐标系。")
            # print("可用的坐标系有:")
            for frame in self.model.frames:
                print(f"  - {frame.name} (ID: {self.model.getFrameId(frame.name)})")

        return position, euler_angles_rpy

    def delta_act(self, q):
        if not hasattr(self, "last"):
            self.last = np.zeros(7)
        position, rotation = self.act(q)
        # print(q)
        # breakpoint()
        # print(position, rotation, q)
        position = position * self.position_scale
        rotation = rotation * self.rotation_scale
        pose = np.concat([position, rotation, np.array([q[-1]])])
        # pose = np.stack([position, rotation, q[-1]])

        if pose.shape[0] != (7):
            raise ValueError("pose shape shoule be (7,)")
        delta_pose = pose - self.last
        self.last = pose.copy()
        if max(delta_pose) > 0.001:
            print(delta_pose)
        return delta_pose


if __name__ == "__main__":
    # agent = GelloAgent()
    gello_port = None
    start_joints = None
    if gello_port is None:
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError("No gello port found, please specify one or plug in gello")

    agent = GelloAgent(port=gello_port, start_joints=start_joints)
    panda = Panda()
    env = gym.make(
        "Tabletop-Pick-Apple-v1",
        # env_name,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",  # Use delta position control
        robot_uids="panda_wristcam",
        render_mode="human",
    )
    obs, info = env.reset()
    display_camera_views(obs)
    while True:
        joint = agent.act()
        action = panda.delta_act(joint)
        obs, _, _, _, _ = env.step(action)
        display_camera_views(obs)
        cv2.waitKey(1)
