from dataclasses import dataclass
from pathlib import Path

import tyro
from zmq import MAX_SOCKETS

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"


def launch_robot_server(args: Args):
    port = args.robot_port
    from gello.robots.ms_robot import MSRobotServer

    env_name = "Tabletop-Clean-For-Dinner-v1"
    # env_name = "Tabletop-Pick-Apple-v1"
    server = MSRobotServer(env_name=env_name, port=port, host=args.hostname)
    # for x in range(10):
    server.serve()
    print("Task finished ")
    server.stop()
    # server.reset()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
