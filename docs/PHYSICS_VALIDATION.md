# Physics methods and validation status

## Implemented methods

`backend/research.py` adds a set of transparent screening/research functions.
They are separate from the existing infinite-slope prediction path, so existing
API behavior is preserved.

| Method | Current implementation | Assumptions / important limits | Status |
|---|---|---|---|
| Infinite slope | Existing `backend/physics.py` Mohr-Coulomb screening equation | Uniform plane parallel to face; prototype blast coefficient; no calibrated stratigraphy | Existing baseline only |
| Bishop simplified | Iterative moment-equilibrium FoS for caller-supplied slices | Circular failure surface; simplified interslice shear treatment; no automated surface search | Formula and response tests pass; not benchmark-certified |
| Janbu simplified | Force-equilibrium FoS, uncorrected | Simplified formulation; empirical correction factor omitted | Formula and response tests pass; not benchmark-certified |
| Hoek–Brown to equivalent Mohr–Coulomb | Fits a line to the generalized Hoek–Brown stress envelope | Equivalent cohesion/friction depend on selected confinement interval and least-squares fit | Formula range-response test passes |
| Planar kinematics | Daylighting, joint-vs-slope dip, friction, and acute direction-window screen | No cohesion, water pressure, lateral restraint, stereonet, wedge, or toppling mechanics | Boundary tests pass |
| Rainfall infiltration | Explicit substeps of Green–Ampt infiltration capacity, capped by rainfall | Homogeneous column, ponded infiltration, no runoff routing or layered soil | Bounds tests pass |

## Published benchmark comparison

**No published FoS benchmark values have been verified and reproduced in this
workspace.** TALUS therefore does not report a comparison table populated with
purported published values. The acceptance target of five reproduced published
benchmarks remains unmet. Required next inputs are independently traceable
benchmark problems, original geometry/material/pore-pressure definitions, and
the publication's reported solution and tolerance. Unit tests in
`tests/test_research_methods.py` check formula behavior and controlled cases;
they are not substitutes for published benchmark validation.

The current implementation also does not perform automatic critical circular
surface search, draw a computed slip surface, full wedge/toppling analysis, or
validate an equivalent Hoek–Brown conversion against a published example.
Those features should not be represented as complete.

## Test scope

```powershell
python -m pytest -q tests/test_research_methods.py
```

The tests verify positive/convergent example outputs, direction of changes with
cohesion/pore pressure, validation errors, method reproducibility, and finite
parameter behavior. None demonstrate accuracy for a real slope.
