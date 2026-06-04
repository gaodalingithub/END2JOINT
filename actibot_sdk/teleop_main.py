import teleop_vr
import argparse
def main():
    parser = argparse.ArgumentParser(description="Teleop VR")
    parser.add_argument("--wait_teleop_call", action="store_true", 
                        help="Initialize motor on startup")
    args = parser.parse_args()
    print("wait_teleop_call: ", args.wait_teleop_call)
    teleop_vr.main(wait_teleop_call=False) # 暂不开启

if __name__ == "__main__":
    main()