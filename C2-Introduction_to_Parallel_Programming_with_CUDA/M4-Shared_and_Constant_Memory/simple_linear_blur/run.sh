#!/usr/bin/env bash
make clean build

make run ARGS="-i sloth.png -o sloth-blur.png"