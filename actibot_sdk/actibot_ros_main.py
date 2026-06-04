#!/usr/bin/env python3

import argparse
import actibot_ros_control

def main():
    parser = argparse.ArgumentParser(description="Actibot ROS Control Node")
    parser.add_argument("--wo_init_motors", action="store_true", 
                        help="Initialize motor on startup")
    args = parser.parse_args()
    print("wo_init_motors: ", args.wo_init_motors)
    actibot_ros_control.main(init_motor=not args.wo_init_motors)

if __name__ == "__main__":
    main()