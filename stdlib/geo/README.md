# stdlib/geo

Geometric primitives and epistemic geometry operations.

## Architecture

- `pure/types.sio` - Point2D, Point3D, LineSegment2D, Triangle2D
- `pure/epistemic.sio` - Epistemic variants with GUM variance propagation
- `lib.sio` - Public API

## Basic Types

- Point2D, Point3D with x, y, z coordinates
- LineSegment2D with start/end points
- Triangle2D with three vertices

## Operations

- Distance calculations (point2d_distance, point3d_distance)
- Triangle area computation
- Centroid calculations
- Segment intersection

## Epistemic Variants

- EPoint2D, EPoint3D with uncertainty
- epoint2d_distance, epoint3d_distance with variance propagation
- etriangle_area_2d with uncertainty

## Usage

```
use geo::lib

let p1 = point2d_new(0.0, 0.0)
let p2 = point2d_new(3.0, 4.0)
let dist = point2d_distance(&p1, &p2)  // 5.0

let ep1 = epoint2d_new(Epistemic::measured(0.0, 0.1), Epistemic::measured(0.0, 0.1))
let ep2 = epoint2d_new(Epistemic::measured(3.0, 0.1), Epistemic::measured(4.0, 0.1))
let edist = epoint2d_distance(&ep1, &ep2)
```

## Tests

`tests/stdlib/geo/test_geo_core.sio` (check-only, Madaros gate)