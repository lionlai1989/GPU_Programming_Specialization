#!/usr/bin/env bash
make clean build

make run-memory-allocation ARGS="-p ZafI7 -o add -n 128 -t 128 -o add"
make run-memory-allocation ARGS="-p Tg70w -o sub -n 128 -t 128 -f test_data.csv -o sub"
make run-memory-copy ARGS="-p IYn5V -n 128 -t 128"
make run-memory-copy ARGS="-p kWuuk -n 128 -t 128 -f test_float_data.csv"
make run-broken-paged-pinned-memory-allocation ARGS="-p pOwge -n 128 -t 128"
make run-broken-mapped-memory-allocation ARGS="-p avffI -n 128 -t 128 "