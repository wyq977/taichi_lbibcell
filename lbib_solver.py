import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm

FORCE = 0.02  # membrane tension between each node on the geometry of the cell


@ti.data_oriented
class lbib_solver:
    def __init__(self, size_x, size_y, niu, bc_type, bc_value, steps=60000):
        self.size_x = size_x
        self.size_y = size_y
        self.steps = steps
        self.rho = ti.field(dtype=ti.f32, shape=(size_x, size_y))
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(size_x, size_y))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))

        # weight and direction vector in D2Q9
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))

        # TODO: store the BoundaryNode produced dynamically
        # TODO: Maybe use a large list and a hashing function will do the jobs, need to check how many
        # TODO: Divide and reconnet self.boundary
        # FIXME:
        self.boundary = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_force = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_vel = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        self.boundary_connection = ti.field(dtype=ti.i32, shape=(4, 2))
        self.num_connection = 4
        self.boundary_neighbours = ti.Vector.field(
            2, dtype=ti.i32, shape=(4, 16))

        # Box boundary Dirichlet or Neumann
        # Dirichlet: when hiting a box, set the local velocity to a certain
        # value, calculate f_old (distribution in D2Q9) again
        # If velocity set to zero, it is essentially an open "fluid" or "gas"
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))

        # WARNING: after giving value to ti.field, no further variables can be
        # init. afterwards
        self.materialization(bc_type, bc_value)

    def materialization(self, bc_type, bc_value):
        # give default values after init all data structure
        arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(arr)
        arr = np.array([[self.size_x / 2, self.size_y / 2],
                        [self.size_x * 7 / 16, self.size_y * 5 / 8],
                        [self.size_x / 2, self.size_y * 3 / 4],
                        [self.size_x * 5 / 8, self.size_y * 5 / 8]], dtype=np.float32)

        # the nodes are connected in clock wise manner, and closed
        self.boundary.from_numpy(arr)
        arr = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        self.boundary_connection.from_numpy(arr)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

    @ti.func
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] \
            + ti.cast(self.e[k, 1], ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @staticmethod
    @ti.func
    def calculate_discrete_delta_dirac(xpos, ypos):
        # for calculate the force in fluid
        return ((1.0 + ti.cos(0.5 * np.pi * xpos)) * (1.0 + ti.cos(0.5 * np.pi * ypos))) / 16

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            # fluid velocity initially zero
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0

            # density set to 1.0
            self.rho[i, j] = 1.0

            # essentially collide during init.
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]

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

    @ti.kernel
    def calculate_force(self):
        for k in ti.ndrange(0, self.num_connection):
            self.calculate_membrane_tension(
                self.boundary_connection[k, 0], self.boundary_connection[k, 1], FORCE)

    @ti.kernel
    def collide_fluid(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] + \
                    self.f_eq(ip, jp, k) * self.inv_tau

    @ti.kernel
    def update_fluid_vel(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
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

    @ti.func
    def connect_boundary_to_neighbour(self, i):
        min_x = ti.cast(ti.ceil(self.boundary[i][0] - 2.0), ti.i32)
        max_x = ti.cast(ti.floor(self.boundary[i][0] + 2.0), ti.i32)
        min_y = ti.cast(ti.ceil(self.boundary[i][1] - 2.0), ti.i32)
        max_y = ti.cast(ti.floor(self.boundary[i][1] + 2.0), ti.i32)

        for x, y in ti.ndrange((0, self.size_x), (0, self.size_y)):
            if x < min_x:
                continue
            elif x > max_x:
                continue
            elif y < min_y:
                continue
            elif y > max_y:
                continue

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

                delta_dirac = self.calculate_discrete_delta_dirac(
                    idx_x - self.boundary[i][0], idx_y - self.boundary[i][1])

                self.boundary_vel[i][0] += vel_x * delta_dirac
                self.boundary_vel[i][1] += vel_y * delta_dirac

    @ti.kernel
    def move_geometry(self):
        # for i in ti.ndrange((0, self.num_connection)):
        for i in ti.static(range(4)):
            self.boundary[i][0] += self.boundary_vel[i][0]
            self.boundary[i][1] += self.boundary_vel[i][1]
            # self.boundary[i][1] += 0.05 # sanity check for this method
            # self.boundary[i][1] += 0.05

    @ti.func
    def apply_bc_core(self, dr, ibc, jbc, inb, jnb):
        # Apply boundary condition to lattice [Box grid only now]
        if (self.bc_type[dr] == 0):  # Dirichlet
            self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
            self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
        elif (self.bc_type[dr] == 1):  # Neumann
            self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
            self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]

        self.rho[ibc, jbc] = self.rho[inb, jnb]  # change local density
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = self.f_eq(
                ibc, jbc, k) - self.f_eq(inb, jnb, k) + self.f_old[inb, jnb][k]

    @ti.kernel
    def apply_bc(self):
        # left and right
        for j in ti.ndrange(1, self.size_y - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(0, 0, j, 1, j)

            # right: dr = 2; ibc = size_x-1; jbc = j; inb = size_x-2; jnb = j
            self.apply_bc_core(2, self.size_x - 1, j, self.size_x - 2, j)

        # top and bottom
        for i in ti.ndrange(self.size_x):
            # top: dr = 1; ibc = i; jbc = size_y-1; inb = i; jnb = size_y-2
            self.apply_bc_core(1, i, self.size_y - 1, i, self.size_y - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(3, i, 0, i, 1)

    def solve(self, video_dir):
        if video_dir:
            video_manager = ti.VideoManager(
                output_dir=video_dir, framerate=24, automatic_build=False)
        gui = ti.GUI('LBIB solver', (2 * self.size_x, 2 * self.size_y))
        self.init()
        for i in range(self.steps):
            self.calculate_force()
            self.collide_fluid()
            self.update_fluid_vel()
            self.connect_boundary_to_physical_nodes()
            self.collect_velocity()
            self.move_geometry()
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

            boundary_neighbours = self.boundary_neighbours.to_numpy()
            bound_loc = np.zeros((self.size_x, self.size_y))
            for bound_idx in range(self.num_connection):
                for j in boundary_neighbours[bound_idx, :, :]:
                    bound_loc[j[0], j[1]] = 1
            bound_img = cm.plasma(bound_loc)
            # img = np.concatenate((vor_img, vel_img, bound_img), axis=1)
            img_l = np.concatenate((vor_img, vel_img), axis=1)
            img_r = np.concatenate((bound_img, bound_img), axis=1)
            img = np.concatenate((img_l, img_r), axis=0)
            gui.set_image(img)
            gui.show()
            if i % 100 == 0:
                print('Step {} is recorded'.format(i))
                if video_dir:
                    video_manager.write_frame(img)
            #     print('Max for vorticity\t{}'.format(np.max(vor)))
            #     print('Max for velocity\t{}'.format(np.max(vel_mag)))

        if video_dir:
            print('Exporting .mp4 and .gif videos...')
            video_manager.make_video(gif=True, mp4=True)
            print(
                'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
            print(
                'GIF video is saved to {video_manager.get_output_filename(".gif")}')


if __name__ == '__main__':
    # Re = U*D/niu = 200
    # [0.0255 * (256 - 1)] / 0.01
    # see https://en.wikipedia.org/wiki/Reynolds_number
    # U, flow speed: 0.01
    # niu, kinematic viscosity, 0.01
    # L, domain length
    ti.init(arch=ti.gpu)
    lbib = lbib_solver(256, 256, 0.0255, [0, 0, 0, 0],
                       [[0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], steps=30000)
    lbib.solve('./lbib')
