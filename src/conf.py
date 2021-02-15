import sys
import importlib
from types import SimpleNamespace
import argparse

sys.path.append("../configs")

parser = argparse.ArgumentParser(description='')

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)

print("Using config file", parser_args.config)

args = importlib.import_module(parser_args.config).args

other_args = importlib.import_module(parser_args.config).other_args
