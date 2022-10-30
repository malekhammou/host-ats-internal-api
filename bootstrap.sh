#!/bin/sh
echo Reading config parameters...
ARGS_STRING=$(python3 - << EOF
import os
import json
ARGS_STRING=""
configFile = open('config.json')
arguments = json.load(configFile)
for key,value in arguments.items():
 ARG= f"{key} {value}"
 ARGS_STRING=ARGS_STRING+" "+ARG
print(ARGS_STRING)
EOF
)

cd code
python3 create_thumbnail.py $ARGS_STRING ../data/videos