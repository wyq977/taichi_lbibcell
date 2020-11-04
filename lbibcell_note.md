## Process of the Simulation in LBIBCell

`PhysicalNode`: Point/Node representing a grid point in the LBM model, pair of integers, i.e. [123, 198], with attribute like force (IBM), velocity (IBM, LBM), distribution (LBM)

`Geometry`: Consist of Points/Nodes defining the cell geometry, pair of floats, i.e. [0.4, 0.323], a closed, convex polygon (*Not Sure*?) represented by a list/array of `Connection`: [1, 2]: `GeometryNode 1 --> GeometryNode 2`

`BoundaryNode`: See below

Basically, `fluidSolver` control the behavior the fluid particle in which cells are interacting like water or intercellular fluid. `CDESolver` control the diffusion, reaction term for the morphogen or any molecules of interest.

In LBIBCell output, it seems like the boundary condition for the box is that:

* What is the `BoundaryNode`? what is the neighbours to that?
  * A subset of `GeometryNode`, at **EVERY** iteration, each `Connection` (a line) will "cut" into the lattices of the fluid
  * In the case of a simple rectangle or triangle, one may only need 4/3 points to define the geometry (`Geometry`, `Connection`) but this triangle will intersect the lattice at many many points horizontally and vertically. And that array/list of points will be used for describe the boundary/shape of the triangle in a discrete coordiantes.
  * Say a triangle *ABC* intersects with grid line (lattices) vertically at point [12, 843.323] (noted that the x is integer), the closest `PhysicalNode` would be [12, 843] and that would be used ultimately to represent the outside shape in the lattices
  * `connectGeometryNodesToPhysicalNodes`, this method connect all nearby nodes to the boundary (cells in our case) by hashing, 
  * See line 645 of `GeometryHandler`, its surrounding `[+/- 2.0, +/- 2.0]`, 16 `PhysicalNode`, will be the near fluid grid point (`PhysicalNode`/lattices) where the nearby `PhysicalNode` will be subject a force from `GeometryNode`
  * Similarly, the velocity of the nearby 16, will affect the velocity corresponding 
* What is Geometry doing?
   * Store the connection, might get updated on each iteration
   * At each iteration, the connection will be re-meshed if a particle connection is too long, i.e. `dist(123, 124) > MAXLENGTH`
* In LBIBCell:
  * A geometryNode has 16 `neighbourPhysicalNodes`,  neighbours are the closest nodes in lattice 9 neighbouring physical node for 9 possible direction for D2Q9 scheme


In the IB method, another grid of the same size were generated, 

1. **calculateForce** First the force on `geometryNode` were calculated, see ForceStructs and the force.txt see `ForceStructs`

2. **distributeForce** `distributeForce`: distribute force on all **geometryNode**
   1. Force of all physical node in lattice are reset to (0,0)
   2. Of 16 `neighbourPhysicalNodes` of every geometryNode, every one is added with a force using calculateDiscreteDeltaDirac based on the distance `physicalNode -> geometryNode`
   
3. **Advection**
   1. **preAdvection** on every **boundaryNode** and **boundarySolver**
      1. Basically for every `boundaryNode`, 
      2. and every boundarySolver on that, do a `preAdvect`
   2. **advection** on every **physicalNode** and **boundaryNode**
      1. For every physicalNode, get `fluidSolver().advect`
      2. For every CDESolver on it, do advect nothing
   3. **postAdvection** on **boundaryNode**
      1. Basically for every `boundaryNode`, 
      2. and every boundarySolver on that, do a `postAdvect`

4. **collide** on every **physicalNode** for all the CDESolver and fluidSolver

   1. For every physicalNode,
   2. fluidSolver collide()
   3. CDESolver collide()

5. **collectVelocity** on every **geometryNode**

   1. each geometry node collect velocity
   2. look into details to help understand the

6. **moveGeometry** `moveLattice` by the `GeomtryHanlder`

7. **remeshBoundary** `remeshBoundary`



