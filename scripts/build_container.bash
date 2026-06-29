#!/usr/bin/env bash
set -e

docker build \
  --platform linux/amd64 \
  -f docker/pose_estimation.dockerfile \
  -t kineticbody:latest .