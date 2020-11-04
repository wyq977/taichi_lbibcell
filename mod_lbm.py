# Fluid solver based on lattice boltzmann method using taichi language
# About taichi : https://github.com/taichi-dev/taichi
# Author : Wang (hietwll@gmail.com)

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

@ti.data_oriented
class lbm_solver:
    def __init__(self, nx, ny, niu, bc_type, bc_value, cy=0,
                 cy_para=[0.0, 0.0, 0.0], steps=60000):
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # density
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))

        # fluid velocity vector at each physical node (terms in LBIBCell)
        # see FluidSolver (ux, uy)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))

        # binary value for whether a lattice is occupied by the obstacle
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))

        # store the previous and new particle distribution
        # see distribution_
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))

        # weight corresponding geometry of the lattice
        self.w = ti.field(dtype=ti.f32, shape=9)

        # fluid velocity?  discrete velocity directions
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))

        # boundary
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))


        # cylindrical obstacle
        self.cy = cy
        self.cy_para = ti.field(dtype=ti.f32, shape=3)
        self.steps = steps

        # Start with a triangle in the middle
        # the nodes are connected in clock wise manner
        self.boundary = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_force = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_vel = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_connection = ti.field(dtype=ti.i32, shape=(4, 2))
        self.num_connection = 4
        self.boundary_neighbours = ti.Vector.field(2, dtype=ti.i32, shape=(4, 16))

        # change this later using a hashing func
        arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(arr)
        arr = np.array([[self.nx / 2, self.ny / 2],
                        [self.nx * 7 / 16, self.ny * 5 / 8],
                        [self.nx / 2, self.ny * 3 / 4],
                        [self.nx * 5 / 8, self.ny * 5 / 8]], dtype=np.float32)
        self.boundary.from_numpy(arr)
        arr = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        self.boundary_connection.from_numpy(arr)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy_para.from_numpy(np.array(cy_para, dtype=np.float32))

    @ti.func
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(self.e[k, 1],
                                                                         ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            # fluid velocity initially zero
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0

            # density set to 1.0
            self.rho[i, j] = 1.0

            # no obstacle
            self.mask[i, j] = 0.0

            # calculate the particle distribution
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]

            # if a cylindrical obstacle were present
            if(self.cy == 1):
                # if lattice was in the cylinder
                if ((ti.cast(i, ti.f32) - self.cy_para[0])**2.0 + (ti.cast(j, ti.f32)
                                                                   - self.cy_para[1])**2.0 <= self.cy_para[2]**2.0):
                    self.mask[i, j] = 1.0

    # see 3.1
    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] + \
                    self.f_eq(ip, jp, k) * self.inv_tau
    @staticmethod
    @ti.func
    def calculate_discrete_delta_dirac(xpos, ypos):
        # for calculate the force in fluid
        return ((1.0 + ti.cos(0.5 * np.pi * xpos)) * (1.0 + ti.cos(0.5 * np.pi * ypos))) / 16

    @ti.func
    def connect_boundary_to_neighbour(self, i):
        min_x = ti.cast(ti.ceil(self.boundary[i][0] - 2.0), ti.i32)
        max_x = ti.cast(ti.floor(self.boundary[i][0] + 2.0), ti.i32)
        min_y = ti.cast(ti.ceil(self.boundary[i][1] - 2.0), ti.i32)
        max_y = ti.cast(ti.floor(self.boundary[i][1] + 2.0), ti.i32)

        for x, y in ti.ndrange((0, self.nx), (0, self.ny)):
            if x < min_x: continue
            elif x > max_x: continue
            elif y < min_y: continue
            elif y > max_y: continue

            hashed_idx = ti.cast((x * 4 + y) % 16, ti.int32)
            self.boundary_neighbours[i, hashed_idx][0] = x
            self.boundary_neighbours[i, hashed_idx][1] = y

    @ti.kernel
    def connect_boundary_to_physical_nodes(self):
        for i in ti.ndrange(0, self.num_connection):
            self.connect_boundary_to_neighbour(i)

    @ti.kernel
    def collect_velocity(self):
        for i in ti.static(range(4)):
            self.connect_boundary_to_neighbour(i)
            self.boundary_vel[i][0] = 0.0
            self.boundary_vel[i][1] = 0.0
            for k in ti.static(range(16)):
                idx_x = self.boundary_neighbours[i, k][0]
                idx_y = self.boundary_neighbours[i, k][1]
                vel_x = self.vel[idx_x, idx_y][0]
                vel_y = self.vel[idx_x, idx_y][1]

                delta_dirac = self.calculate_discrete_delta_dirac(idx_x - self.boundary[i][0], idx_y - self.boundary[i][1])
                # print('Neibour', idx_y, idx_x, vel_x, vel_y, delta_dirac)

                self.boundary_vel[i][0] += vel_x * delta_dirac
                self.boundary_vel[i][1] += vel_y * delta_dirac

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) *
                                      self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) *
                                      self.f_new[i, j][k])
            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):
        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if (self.cy == 1 and self.mask[i, j] == 1):
                self.vel[i, j][0] = 0.0  # velocity is zero at solid boundary
                self.vel[i, j][1] = 0.0
                inb = 0
                jnb = 0
                if (ti.cast(i, ti.f32) >= self.cy_para[0]):
                    inb = i + 1
                else:
                    inb = i - 1
                if (ti.cast(j, ti.f32) >= self.cy_para[1]):
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if (outer == 1):  # handle outer boundary
            if (self.bc_type[dr] == 0):  # Dirichlet
                self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):  # Neumann
                self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
                self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]

        # change local density to [inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = self.f_eq(ibc, jbc, k) - self.f_eq(inb, jnb, k) + \
                self.f_old[inb, jnb][k]

    def solve(self):
        # change the res to integer
        gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.init()
        for i in range(self.steps):
            self.collide_and_stream()
            self.update_macro_var()
            # self.connect_boundary_to_physical_nodes()
            self.collect_velocity()
            self.apply_bc()
            # code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0]**2.0 + vel[:, :, 1]**2.0)**0.5
            # color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
                      (0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)

            bound_cooridnates = self.boundary.to_numpy()
            bound_vel = self.boundary_vel.to_numpy()
            # print(bound_cooridnates, bound_cooridnates.shape)
            print(np.max(bound_vel), bound_vel.shape)
            # exit(1)
            # bound_vel = np.zeros((self.nx, self.ny))
            # bound_img = cm.plasma(bound_vel / 0.15)
            # bound_vel
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()
            if (i % 1000 == 0):
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')

    def pass_to_py(self):
        return self.vel.to_numpy()[:, :, 0]


if __name__ == '__main__':
    flow_case = 1
    if (flow_case == 0):  # von Karman vortex street: Re = U*D/niu = 200 
        # [0.1 * (201 - 1)] / 0.01
        # see https://en.wikipedia.org/wiki/Reynolds_number
        # Re = 
        # U, flow speed: 0.01
        # niu, kinematic viscosity, 0.01
        # L, domain length
        lbm = lbm_solver(801, 201, 0.01, [0, 0, 1, 0],
             [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
             1,[160.0, 100.0, 20.0])
        lbm.solve()
    elif (flow_case == 1):  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = lbm_solver(256, 256, 0.0255, [0, 0, 0, 0],
                         [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        lbm.solve()
        # # compare with literature results
        # y_ref, u_ref = np.loadtxt(
        #     'data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 2))
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
        # axes.plot(np.linspace(0, 1.0, 256), lbm.pass_to_py()
        #           [256 // 2, :] / 0.1, 'b-', label='LBM')
        # axes.plot(y_ref, u_ref, 'rs', label='Ghia et al. 1982')
        # axes.legend()
        # axes.set_xlabel(r'Y')
        # axes.set_ylabel(r'U')
        # plt.tight_layout()
        # plt.show()
