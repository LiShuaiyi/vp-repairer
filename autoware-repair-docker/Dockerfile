ARG BASE_IMAGE
FROM $BASE_IMAGE AS base
ARG ROS_DISTRO

# Set environment variables
ENV ROS_DISTRO=${ROS_DISTRO}
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Set this ARG to the current date to invalidate the cache, using the latest cacc-test repo
ARG BUILD_DATE

# Rebuild the cr2autoware package with --symlink-install
WORKDIR /autoware/src/universe/autoware.universe/planning/tum_commonroad_planning
RUN rm -rf dfg-car
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan gitlab.lrz.de >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone -b vp-repair-autoware git@gitlab.lrz.de:cps/dfg-car.git dfg-car

WORKDIR /autoware
# Remove the existing cr2autoware package and dependencies
RUN bash -c "rm -rf build/cr2autoware install/cr2autoware"

# Build cr2autoware package
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
             source /autoware/install/setup.bash && \
             colcon build \
             --cmake-args -DCMAKE_BUILD_TYPE=Release \
             --packages-select \
             cr2autoware"

# Create a symbolic link for the cr2autoware_param_file.param.yaml file
RUN bash -c "rm /autoware/install/cr2autoware/share/cr2autoware/param/cr2autoware_param_file.param.yaml"
RUN bash -c "ln -s /autoware/src/universe/autoware.universe/planning/tum_commonroad_planning/dfg-car/src/cr2autoware/param/cr2autoware_param_file.param.yaml /autoware/install/cr2autoware/share/cr2autoware/param/cr2autoware_param_file.param.yaml"

# Fix the bug "AttributeError: module 'enum' has no attribute 'IntFlag'"
RUN pip uninstall enum34 -y

# Install crrepair python library and dependencies
# WORKDIR /crrepair
# RUN git clone -b feature/repair/miqp https://gitlab.lrz.de/cps/commonroad-qp-planner.git
# RUN git clone --recursive -b feature/repair/miqp https://gitlab.lrz.de/cps/commonroad-stl-monitor.git
# RUN git clone -b feature/repair/miqp https://gitlab.lrz.de/cps/commonroad-model-predictive-robustness.git

# Comment commonroad-mpr from the commonroad-stl-monitor/pyproject.toml
# RUN sed -i'' '/commonroad-mpr/s/^/#/' "commonroad-stl-monitor/pyproject.toml"

# WORKDIR /crrepair/commonroad-qp-planner
# RUN --mount=type=cache,target=/root/.cache pip install -e .

# WORKDIR /crrepair/commonroad-stl-monitor
# RUN --mount=type=cache,target=/root/.cache pip install -e .

# WORKDIR /crrepair/commonroad-model-predictive-robustness
# RUN --mount=type=cache,target=/root/.cache pip install -e .

# Copy crrepair library
WORKDIR /crrepair/commonroad-repairer
COPY . .

WORKDIR /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-qp-planner
RUN --mount=type=cache,target=/root/.cache pip install -e .

WORKDIR /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-stl-monitor
RUN --mount=type=cache,target=/root/.cache pip install -e .

WORKDIR /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-model-predictive-robustness
RUN --mount=type=cache,target=/root/.cache pip install -e .

WORKDIR /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-criticality-measures
RUN --mount=type=cache,target=/root/.cache pip install -e .
RUN ln -s /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-criticality-measures /crrepair/commonroad-criticality-measures

# Use opensuse deb source to install the latest Spot instead of official deb source, see https://gitlab.lre.epita.fr/spot/spot/-/issues/544
RUN echo 'deb [trusted=true] https://download.opensuse.org/repositories/home:/adl/xUbuntu_22.04/ ./' >> /etc/apt/sources.list
# RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | sudo tee /etc/apt/keyrings/lrde-spot.gpg
# RUN echo 'deb [signed-by=/etc/apt/keyrings/lrde-spot.gpg] http://www.lrde.epita.fr/repo/debian/ stable/' | sudo tee -a /etc/apt/sources.list
RUN sudo apt-get update
RUN sudo apt-get install -y  \
libspot-dev  \
libbddx-dev  \
libspot0  \
libspotltsmin0  \
libbddx0 \
libspotgen0 \
python3-spot
RUN sudo apt-get install -y libomp-dev libboost-all-dev libeigen3-dev libyaml-cpp-dev libspdlog-dev pybind11-dev libgtest-dev libgmock-dev g++-10

# Add ninja installation
RUN sudo apt-get install -y ninja-build

WORKDIR /crrepair/commonroad-repairer/autoware-repair-docker/commonroad_dependencies/commonroad-reach-semantic
# RUN export CXX=/usr/bin/g++-10 && pip install -v -e .
RUN python3 -m pip install --upgrade "pip<25" setuptools "wheel<0.47" && \
    python3 -m pip install --force-reinstall \
      "packaging==23.2" \
      "pyproject-metadata==0.9.0" \
      "scikit-build-core==0.9.8" \
      "pathspec>=0.10" \
      "cmake==3.27.9" \
      "pybind11"
RUN export CXX=/usr/bin/g++-10 && \
    export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" && \
    python3 -m pip install --no-build-isolation -v -e .

WORKDIR /crrepair/commonroad-repairer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libcdd-dev \
      libgmp-dev && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache pip install -e .

# Fix the bug "importerror: cannot import name 'docstring' from 'matplotlib'"
RUN pip uninstall matplotlib -y

RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    "commonroad-drivability-checker==2024.1" \
    "commonroad-io==2024.2"
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    "setuptools==69.0.2" \
    "wheel==0.42.0" \
    "packaging==23.2"
# WORKDIR /crrepair/commonroad-repairer
# RUN python3 -m pip install -e .

# RUN sudo apt-get install -y libxcb-cursor-dev
RUN set -eux; \
    mkdir -p /tmp/apt-disabled; \
    find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) -print | while read f; do \
      if grep -Eq "packages\.ros\.org|download\.opensuse\.org|autonomoustuff-repo" "$f"; then \
        mv "$f" "/tmp/apt-disabled/$(basename "$f")"; \
      fi; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends libxcb-cursor-dev; \
    rm -rf /var/lib/apt/lists/*; \
    find /tmp/apt-disabled -type f -print | while read f; do \
      mv "$f" "/etc/apt/sources.list.d/$(basename "$f")"; \
    done
RUN --mount=type=cache,target=/root/.cache pip install PyQt6

# Keep NumPy on the 1.x ABI expected by Ubuntu/ROS binary Python packages
# such as matplotlib, numexpr, and bottleneck.
RUN python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps "commonroad-route-planner==2024.2.0"

# Create entrypoint
RUN echo '#!/usr/bin/env bash' > /crrepair_entrypoint.sh
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /crrepair_entrypoint.sh
RUN echo 'source /autoware/install/setup.bash' >> /crrepair_entrypoint.sh
RUN echo 'export SETUPTOOLS_USE_DISTUTILS=stdlib' >> /crrepair_entrypoint.sh
RUN echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> /crrepair_entrypoint.sh
RUN echo 'exec "$@"' >> /crrepair_entrypoint.sh
RUN chmod +x /crrepair_entrypoint.sh

RUN echo 'source /autoware/install/setup.bash' >> ~/.bashrc

# Edit file "/usr/local/lib/python3.10/dist-packages/commonroad_reach/utility/configuration.py"
# find the line containing "eps2: float = 1e-4" and change it to "eps2: float = 30"
# To fix the bug:
#    File "/usr/local/lib/python3.10/dist-packages/commonroad_reach/utility/configuration.py", line 189, in compute_initial_state_cvln
#    p_lon, p_lat = config.planning.CLCS.convert_to_curvilinear_coords(
#    ValueError: x and/or y coordinate outside of projection domain
RUN sed -i 's/eps2: float = 1e-4/eps2: float = 30/' /usr/local/lib/python3.10/dist-packages/commonroad_reach/utility/configuration.py

ENTRYPOINT ["/crrepair_entrypoint.sh"]
CMD ["bash"]
