"""Main CLI entry point."""

import argparse
import sys
from .commands import simulate, train, diagnose

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="ABCNRE: ABC with Neural Ratio Estimation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add subcommands
    simulate.add_parser(subparsers)
    train.add_parser(subparsers)  
    diagnose.add_parser(subparsers)
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        simulate.run(args)
    elif args.command == "train":
        train.run(args)
    elif args.command == "diagnose":
        diagnose.run(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
