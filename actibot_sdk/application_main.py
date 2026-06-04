import application
import argparse
def main():
    parser = argparse.ArgumentParser(description="Actibot Application Node")
    parser.add_argument("--auto_can_activate", action="store_true", 
                        help="Auto activate CAN")
    args = parser.parse_args()
    application.main(auto_can_activate=args.auto_can_activate)

if __name__ == "__main__":
    main()