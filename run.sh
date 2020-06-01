#!/bin/sh
nohup xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 example.py