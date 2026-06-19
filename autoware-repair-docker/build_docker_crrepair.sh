#!/bin/bash

source ./autoware-repair-docker/compose_launch/.env

ROS_DISTRO="humble"
DOCKERFILE_DIRECTORY="autoware-repair-docker/Dockerfile"
IMAGE_BASE="gitlab.lrz.de:5005/av2.0/av_software/autoware/microservice/planning:humble-${architecture}-${planning}"
IMAGE_TAG="gitlab.lrz.de:5005/av2.0/av_software/autoware/microservice/planning:humble-${architecture}-${planning_repair_autoware}"

cd autoware-repair-docker
mkdir -p commonroad_dependencies
cd commonroad_dependencies
if [ ! -d commonroad-qp-planner ]; then
    git clone -b feature/repair-all git@gitlab.lrz.de:cps/commonroad-qp-planner.git
fi
if [ ! -d commonroad-stl-monitor ]; then
    git clone --recursive -b feature/autoware git@gitlab.lrz.de:cps/commonroad-stl-monitor.git
fi
if [ ! -d commonroad-model-predictive-robustness ]; then
    git clone -b feature/repair-all git@gitlab.lrz.de:cps/commonroad-model-predictive-robustness.git
fi
if [ ! -d commonroad-criticality-measures ]; then
    git clone git@gitlab.lrz.de:cps/commonroad/commonroad-criticality-measures.git
fi
if [ ! -d commonroad-reach-semantic ]; then
    git clone -b feature/repair-new git@gitlab.lrz.de:cps/commonroad/commonroad-reach-semantic.git
fi

# Comment commonroad-mpr from the commonroad-stl-monitor/pyproject.toml
sed -i'' '/^[[:space:]]*#/!s/^[[:space:]]*commonroad-mpr/#&/' "commonroad-stl-monitor/pyproject.toml"

cd ../..

# Set DOCKER_BUILDKIT to 1 to use pip cache
export DOCKER_BUILDKIT=1

docker build --ssh default="$SSH_AUTH_SOCK"  --tag "$IMAGE_TAG" \
    --build-arg ROS_DISTRO=$ROS_DISTRO \
    --build-arg BASE_IMAGE=$IMAGE_BASE \
    --build-arg BUILD_DATE=$(date +%Y-%m-%d:%H:%M:%S) \
    --network host \
    --progress=plain \
    -f "$DOCKERFILE_DIRECTORY" .
    # --no-cache \
