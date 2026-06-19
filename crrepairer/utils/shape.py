from typing import Tuple


def shape_dimensions(shape) -> Tuple[float, float]:
    """
    Return shape dimensions as ``(length, width)``.

    Most planner code expects rectangular shapes exposing ``length`` and
    ``width``. Some CommonRoad scenarios use ``Circle`` instead, which only
    provides ``radius``. In that case we approximate the footprint by its
    diameter on both axes.
    """
    length = getattr(shape, "length", None)
    width = getattr(shape, "width", None)
    if length is not None and width is not None:
        return float(length), float(width)

    radius = getattr(shape, "radius", None)
    if radius is not None:
        diameter = 2.0 * float(radius)
        return diameter, diameter

    raise AttributeError(
        f"Unsupported shape type {type(shape).__name__}: expected length/width or radius"
    )
