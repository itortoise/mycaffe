#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=./examples/deep-iris/deep_iris_solver.prototxt

