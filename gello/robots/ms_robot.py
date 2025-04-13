import pickle
import cv2

import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq

# from dm_control import mjcf
from gello import env
import mani_skill
from mani_skill.utils.display_multi_camera import display_camera_views
import gymnasium as gym
from gello.robots.robot import Robot

from mani_skill.utils.wrappers.record import RecordEpisode

assert mujoco.viewer is mujoco.viewer


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MSRobotServer:
    def __init__(
        self,
        env_name="Tabletop-Pick-Apple-v1",
        host: str = "127.0.0.1",
        port: int = 5556,
        # print_joints: bool = True,
    ):
        # self._has_gripper = gripper_xml_path is not None
        # arena = build_scene(xml_path, gripper_xml_path)

        assets: Dict[str, str] = {}
        # for asset in arena.asset.all_children():
        #     if asset.tag == "mesh":
        #         f = asset.file
        #         assets[f.get_vfs_filename()] = asset.file.contents
        #
        # xml_string = arena.to_xml_string()
        # save xml_string to file
        # with open("arena.xml", "w") as f:
        #     f.write(xml_string)
        #
        # self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        # self._data = mujoco.MjData(self._model)
        self.env = gym.make(
            # "Tabletop-Pick-Apple-v1",
            env_name,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",  # Use delta position control
            robot_uids="panda_wristcam",
            render_mode="human",
        )
        output_dir = "teleoperation_dataset"
        import datetime

        self.save_video = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_name = f"trajectory_{timestamp}"
        self.env = RecordEpisode(
            self.env,
            output_dir=output_dir,
            trajectory_name=trajectory_name,
            save_video=self.save_video,
            info_on_video=False,
            source_type="ar_teleoperation",
            source_desc="teleoperation via MujocoARConnector",
        )
        obs, info = self.env.reset()
        self.obs = obs
        self._num_joints = 8

        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self._has_gripper = True
        # self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self.obs["agent"]["qpos"].cpu().numpy()[0][: self._num_joints]
        # print(joint_positions)
        joint_velocities = self.obs["agent"]["qvel"].cpu().numpy()[0]
        ee_pos_quat = self.obs["extra"]["tcp_pose"].cpu().numpy()[0]
        # ee_pos_quat = self.obs["extra"]
        gripper_pos = joint_positions
        # print("self.obs keys is ", self.obs["extra"].keys())
        # ee_pos = self.obs["extra_obs"]
        # pass
        # joint_positions = self._data.qpos.copy()[: self._num_joints]
        # joint_velocities = self._data.qvel.copy()[: self._num_joints]
        # ee_site = "attachment_site"
        # try:
        #     ee_pos = self._data.site_xpos.copy()[
        #         mujoco.mj_name2id(self._model, 6, ee_site)
        #     ]
        #     ee_mat = self._data.site_xmat.copy()[
        #         mujoco.mj_name2id(self._model, 6, ee_site)
        #     kkk
        #     ]
        #     ee_quat = np.zeros(4)
        #     mujoco.mju_mat2Quat(ee_quat, ee_mat)
        # except Exception:
        #     ee_pos = np.zeros(3)
        #     ee_quat = np.zeros(4)
        #     ee_quat[0] = 1
        # gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": gripper_pos,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()

        # with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
        #
        done = False

        while not done:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            action = self._joint_cmd
            action = np.array(action.tolist() + [0])
            # self._data.qpos[:] = self._joint_cmd
            # print("the control data is ", action)
            # mujoco.mj_step(self._model, self._data)
            # self._joint_state = self._data.qpos.copy()[: self._num_joints]

            # if self._print_joints:
            #     print(self._joint_state)
            # action = action.cpu().numpy()[: self._num_joints]
            action = action[: self._num_joints]
            obs, _, done, trun, _ = self.env.step(action)
            obs = self.obs
            done = trun or done
            # Example modification of a viewer option: toggle contact points every two seconds.
            # with viewer.lock():
            #     # TODO remove?
            #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
            #         self._data.time % 2
            #     )

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # viewer.sync()
            # display_camera_views()
            initial_img, title = display_camera_views(obs)
            cv2.waitKey(1)
            cv2.moveWindow(title, 500, 400)

            time.sleep(0.02)
            #
            # # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()

    def reset(self):
        self.env.reset()
