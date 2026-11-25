# Vendored Halder & Althoff planner

`minimum-violation-velocity-planner-master/` is the unmodified release of the
minimum-violation velocity planner accompanying Halder and Althoff (ITSC 2022),
extracted verbatim from:

- archive: `minimum-violation-velocity-planner-master.zip` (ITSC 2022 software data)
- archive comment (source commit): `282d312af5c4105a1e01adb6d3d7a18bf49c2a85`
- archive SHA-256: `ce04c7e5173b44b1a2cd80dac3e4ea8dd5c4386c7436d934e7f2ee259b8597f2`
- package version: `2022.1` (`setup.py`)
- license: MIT (classifier in `setup.py`; the archive contains no separate
  license file)

`../author_reference_planner.py` loads this copy by default; set
`HALDER_AUTHOR_SOURCE` to use another checkout. The code is treated as
read-only: compatibility shims for newer CommonRoad versions live in the
adapter, not here.
