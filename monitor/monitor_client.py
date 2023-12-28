import zmq
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('cmd', type=str)
parser.add_argument('--port', type=int, default=1579)
args = parser.parse_args()

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(f"tcp://localhost:{args.port}")

socket.send_string(args.cmd)
