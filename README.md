# taichi_lbibcell

In this repo, a implementation of the cell simulation framework [LBIBCell](https://tanakas.bitbucket.io/lbibcell/index.html) using [Taichi](https://github.com/taichi-dev/taichi) programming language.

## Example1: Lid-driven Cavity Flow

<div align="center">
<img src="data/lid_driven_cavity_with_immersed_body.gif" height="400px">
</div>

This is a benchmark fluid-dynamics problem used to verify the solver accuracy. To compare simulation results based on different unit-systems, the flow Reynolds number ``Re`` should keep the same. In this case, ``Re`` is defined as ``Re = U * L / niu``, so a solver with `` Re = 1000 `` can be given by:

```python
lbib = lbib_solver(256, 256, 0.0255, [0, 0, 0, 0],
                    [[0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], steps=30000)
```

Here ``Re = U * (nx-1) * dx / niu = 0.1 * 255.0 / 0.0255``. The velocity magnitude is shown in the contour below and x-component of velocity in the middle line is compared with result from literature.

## Misc

1. [How to see if a point is inside of a Polygon](https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon)
2. [A fast and robust way to judge `point in polygon` in javascript](https://github.com/rowanwins/point-in-polygon-hao)
3. [Runtime profiler for Taichi program](https://taichi.readthedocs.io/en/stable/profiler.html)