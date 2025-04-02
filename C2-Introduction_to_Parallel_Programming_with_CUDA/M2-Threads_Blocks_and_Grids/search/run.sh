#!/usr/bin/env bash
make clean build

make run ARGS="-p TFpFX"
make run ARGS="-p TZ5dO"
make run ARGS="-s true -t 128 -n 100 -p fnOOK"
make run ARGS="-s false -t 128 -n 1028 -p Vmwc9"
make run ARGS="-s false -t 128 -n 1028 -v 93301624 -f test_data.csv -p IzONJ"