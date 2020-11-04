import taichi as ti
import taichi_glsl as ts
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

FORCE = 0.02


@ti.data_oriented
class lbibcell_solver:
    def __init__(self, size_x, size_y, niu, steps, bc_type, bc_value):
        self.size_x = size_x
        self.size_y = size_y
        self.steps = steps
        self.bc_type = bc_type
        self.bc_value = bc_value

        # boundary
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))

        self.domain_id = ti.field(dtype=ti.f32, shape=(size_x, size_y))
        # set fluid viscocity constant
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        self.rho = ti.field(dtype=ti.f32, shape=(size_x, size_y))

        self.fluid_vel = ti.Vector.field(
            2, dtype=ti.f32, shape=(size_x, size_y))
        self.fluid_force = ti.Vector.field(
            2, dtype=ti.f32, shape=(size_x, size_y))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))

        self.weight = ti.field(dtype=ti.f32, shape=9)
        self.lattice_dir = ti.field(dtype=ti.i32, shape=(9, 2))

        # Start with a triangle in the middle
        # the nodes are connected in clock wise manner
        self.boundary = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_force = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_vel = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_connection = ti.field(dtype=ti.i32, shape=(4, 2))
        self.num_connection = 4

        # change this later using a hashing func
        self.boundary_neighbours = ti.Matrix.field(16, 2, dtype=ti.i32, shape=(4,))

        self.materialization()

    def materialization(self):
        # give default values after init all data structure
        # weight and direction for D2Q9
        arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.weight.from_numpy(arr)

        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.lattice_dir.from_numpy(arr)

        # init geo
        arr = np.array([[self.size_x / 2, self.size_y / 2],
                        [self.size_x * 7 / 16, self.size_y * 5 / 8],
                        [self.size_x / 2, self.size_y * 3 / 4],
                        [self.size_x * 5 / 8, self.size_y * 5 / 8]], dtype=np.float32)
        self.boundary.from_numpy(arr)

        arr = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        self.boundary_connection.from_numpy(arr)
    
    @staticmethod
    @ti.func
    def calculate_discrete_delta_dirac(xpos, ypos):
        # for calculate the force in fluid
        return ((1.0 + ti.cos(0.5 * np.pi * xpos)) * (1.0 + ti.cos(0.5 * np.pi * ypos))) / 16

    @ti.func
    def calculate_membrane_tension(self, i, j, force_const):
        # cannot use nested func
        self.boundary_force[i][0] += (
            self.boundary[j][0] - self.boundary[i][0]) * force_const  # x dir for point i
        self.boundary_force[i][1] += (
            self.boundary[j][1] - self.boundary[i][1]) * force_const  # y dir for point i

        self.boundary_force[i][0] += (
            self.boundary[i][0] - self.boundary[j][0]) * force_const  # x dir for point j
        self.boundary_force[i][0] += (
            self.boundary[i][1] - self.boundary[j][1]) * force_const  # y dir for point j

    @ti.func
    def connect_boundary_to_neighbour(self, i):
        min_x = ti.cast(ti.ceil(self.boundary[i][0] - 2.0), ti.i32)
        max_x = ti.cast(ti.floor(self.boundary[i][0] + 2.0), ti.i32)
        min_y = ti.cast(ti.ceil(self.boundary[i][1] - 2.0), ti.i32)
        max_y = ti.cast(ti.floor(self.boundary[i][1] + 2.0), ti.i32)

        for x, y in ti.ndrange((0, self.size_x), (0, self.size_y)):
            if x < min_x: continue
            elif x > max_x: continue
            elif y < min_y: continue
            elif y > max_y: continue

            hashed_idx = (x * 4 + y) % 16
            self.boundary_neighbours[i][hashed_idx, 0] = x
            self.boundary_neighbours[i][hashed_idx, 1] = y

            # reset the force on those grid
            self.fluid_force[x, y][0] = 0.0
            self.fluid_force[x, y][1] = 0.0

    @ti.kernel
    def connect_boundary_to_physical_nodes(self):
        for i in ti.ndrange(0, self.num_connection):
            self.connect_boundary_to_neighbour(i)
    
    @ti.kernel
    def calculate_force(self):
        for i in ti.ndrange(0, self.num_connection):
            self.calculate_membrane_tension(self.boundary_connection[i, 0], self.boundary_connection[i, 1], FORCE)

    @ti.kernel
    def distribute_force(self):
        for i in ti.ndrange(0, self.num_connection):
            force_x = self.boundary_force[i][0]
            force_y = self.boundary_force[i][1]
            for k in ti.static(range(16)):
                idx_x = self.boundary_neighbours[i][k, 0]
                idx_y = self.boundary_neighbours[i][k, 1]

                delta_dirac = self.calculate_discrete_delta_dirac(idx_x - self.boundary[i][0], idx_y - self.boundary[i][1])

                self.fluid_force[idx_x, idx_y][0] += force_x * delta_dirac
                self.fluid_force[idx_x, idx_y][1] += force_y * delta_dirac
    
    @ti.func
    def f_eq_d2q9(self, i, j, k):
        eu = ti.cast(self.lattice_dir[k, 0], ti.f32) * self.fluid_vel[i, j][0] \
           + ti.cast(self.lattice_dir[k, 1], ti.f32) * self.fluid_vel[i, j][1]
        uv = self.fluid_vel[i, j][0]**2.0 + self.fluid_vel[i, j][1]**2.0
        return self.weight[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @ti.func
    def advect_fluid(self, i, j):
    #     arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
    #                     [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.f_old[i, j][1] = self.f_old[i - 1, j][3]     # E --> W neighbour's W
        self.f_old[i, j][5] = self.f_old[i - 1, j - 1][7] # NE --> SW neighbour's SW
        self.f_old[i, j][2] = self.f_old[i, j - 1][4]     # N  --> S neighbour's S
        self.f_old[i, j][6] = self.f_old[i + 1, j - 1][8] # NW --> SE neighbour's SE

    @ti.kernel
    def advect(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            self.advect_fluid(i, j)

    @ti.kernel
    def collide_fluid(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            for k in ti.static(range(9)):
                ip = i - self.lattice_dir[k, 0]
                jp = j - self.lattice_dir[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] + self.f_eq_d2q9(ip, jp, k) * self.inv_tau #\
                                    # + 3.0 * self.weight[k] * self.fluid_force[i, j][0] * ti.cast(self.lattice_dir[k, 0], ti.f32) \
                                    # + 3.0 * self.weight[k] * self.fluid_force[i, j][1] * ti.cast(self.lattice_dir[k, 1], ti.f32)

    @ti.kernel
    def update_fluid_vel(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            self.rho[i, j] = 0.0
            self.fluid_vel[i, j][0] = 0.0
            self.fluid_vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k] # accumalate fluid density
                self.fluid_vel[i, j][0] += (ti.cast(self.lattice_dir[k, 0], ti.f32) * self.f_new[i, j][k])
                self.fluid_vel[i, j][1] += (ti.cast(self.lattice_dir[k, 1], ti.f32) * self.f_new[i, j][k])
            self.fluid_vel[i, j][0] /= self.rho[i, j]
            self.fluid_vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def collect_velocity(self):
        for i in ti.ndrange(0, self.num_connection):
            self.boundary_vel[i][0] = 0.0
            self.boundary_vel[i][1] = 0.0
            for k in ti.static(range(16)):
                idx_x = self.boundary_neighbours[i][k, 0]
                idx_y = self.boundary_neighbours[i][k, 1]
                vel_x = self.fluid_vel[idx_x, idx_y][0]
                vel_y = self.fluid_vel[idx_x, idx_y][1]

                delta_dirac = self.calculate_discrete_delta_dirac(idx_x - self.boundary[i][0], idx_y - self.boundary[i][1])

                self.boundary_vel[i][0] += vel_x * delta_dirac
                self.boundary_vel[i][1] += vel_y * delta_dirac

                print("Vel.x: {}".format(self.boundary_vel[i][0]))
                print("Vel.y: {}".format(self.boundary_vel[i][1]))

    @ti.kernel
    def move_geometry(self):
        for i in ti.ndrange(0, self.num_connection):
            self.boundary[i][0] += self.boundary_vel[i][0]
            self.boundary[i][1] += self.boundary_vel[i][1]
    
    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if (outer == 1):  # handle outer boundary
            if (self.bc_type[dr] == 0):  # Dirichlet
                self.fluid_vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.fluid_vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):  # Neumann
                self.fluid_vel[ibc, jbc][0] = self.fluid_vel[inb, jnb][0]
                self.fluid_vel[ibc, jbc][1] = self.fluid_vel[inb, jnb][1]

        # change local density to [inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = self.f_eq_d2q9(ibc, jbc, k) - self.f_eq_d2q9(inb, jnb, k) + \
                self.f_old[inb, jnb][k]
    @ti.kernel
    def apply_bc(self):
        # left and right
        for j in ti.ndrange(1, self.size_y - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.size_x - 1, j, self.size_x - 2, j)

        # top and bottom
        for i in ti.ndrange(self.size_x):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.size_x - 1, i, self.size_x - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

    @ti.kernel
    def init_solver(self):
        for i, j in self.rho:
            self.fluid_vel[i, j][0] = 0.0
            self.fluid_vel[i, j][1] = 0.0
            self.fluid_force[i, j][0] = 0.0
            self.fluid_force[i, j][1] = 0.0

            self.rho[i, j] = 1.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq_d2q9(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]
        
        for i in ti.ndrange(0, self.num_connection):
            self.boundary_force[i][0] = 0.0
            self.boundary_force[i][1] = 0.0
            self.boundary_vel[i][1] = 0.0
            self.boundary_vel[i][1] = 0.0

            for k in ti.static(range(9)):
                self.boundary_neighbours[i][k, 0] = 0
                self.boundary_neighbours[i][k, 1] = 0

    def solve(self):
        # change the res to integer
        gui = ti.GUI('lbm solver', (self.size_x, 2 * self.size_y))
        self.init_solver()
        for i in range(self.steps):
            self.calculate_force()
            self.distribute_force()
            # self.advect()
            self.collide_fluid()
            self.collect_velocity()
            self.move_geometry()
            self.update_fluid_vel()
            self.apply_bc()
            # code fragment displaying vorticity is contributed by woclass
            vel = self.fluid_vel.to_numpy()
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
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()
            if (i % 1000 == 0):
                print('Step: {:}'.format(i))


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    lbm = lbibcell_solver(501, 501, 0.01, 10000, [0, 0, 0, 0],
                         [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lbm.solve()
