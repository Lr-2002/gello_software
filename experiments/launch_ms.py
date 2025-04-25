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
    env_name: str = "Tabletop-Close-Microwave-v1"


def launch_robot_server(args: Args):
    port = args.robot_port
    from gello.robots.ms_robot import MSRobotServer

    # env_name = "Tabletop-Clean-For-Dinner-v1"
    # env_name = "Tabletop-Find-Book-From-Shelf-v1"
    env_name = "Tabletop-v2"
    env_name = "Tabletop-Pick-Cube-WithStick-v1"
    env_name = "Tabletop-Open-Cabinet-v1"
    env_name = "Tabletop-Find-Dice-v1"
    env_name = "Tabletop-Find-Seal-v1"
    env_name = "Tabletop-Finish-Hanoi-v1"
    env_name = "Tabletop-Rotate-Holder-v1"
    env_name = "Tabletop-Pull-Pivot-v1"
    # env_name = "Tabletop-Rotate-Cube-v1"
    env_name = "Tabletop-Rotate-USB-v1"
    env_name = "Tabletop-Insert-Objects-v1"
    env_name = "Tabletop-Keep-Pivot-Balance-v1"
    # env_name = "Tabletop-Keep-Pivot-Balance-v1"
    env_name = "Tabletop-Merge-USB-v1"
    env_name = "Tabletop-Move-Balls-WithPivot-v1"
    env_name = "Tabletop-Move-Cube-WithPivot-v1"
    env_name = "Tabletop-Open-Cabinet-WithSwitch-v1"
    env_name = "Tabletop-Pick-Cylinder-WithObstacle-v1"
    env_name = "Tabletop-Pick-Eraser-FromHolder1-v1"
    env_name = "Tabletop-Pick-Heavy-One-v1"
    env_name = "Tabletop-Pick-Cube-WithStick-v1"
    env_name = "Tabletop-Pick-Heavy-One-v1"
    env_name = "Tabletop-Find-Cube-WithPivot-v1"
    env_name = "Tabletop-Move-Cube-DynamicFriction-v1"
    env_name = "Tabletop-Rotate-Cube-Twice-v1"
    env_name = "Tabletop-Seek-Objects-WithObstacle-v1"
    env_name = "Tabletop-Stack-Books-v1"
    env_name = "Tabletop-Stack-Books-OnBox-v1"
    env_name = "Tabletop-Stack-LongObjects-v1"
    env_name = "Tabletop-Pick-Bottle-v1"
    env_name = "Tabletop-Open-Drawer-v1"
    env_name = "Tabletop-Close-Microwave-v1"
    env_name = "Tabletop-Pick-Pen-v1"
    env_name = "Tabletop-Put-Fork-OnPlate-v1"
    # env_name = "Tabletop-Rotate-Cube-v1"
    # env_name = "Tabletop-Merge-Box-v1"
    # for x in range(10):

    server = MSRobotServer(env_name=args.env_name, port=port, host=args.hostname)
    server.serve()
    print("Task finished ")
    server.stop()
    # server.reset()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
