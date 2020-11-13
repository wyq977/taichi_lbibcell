from numpy.core.fromnumeric import product
import taichi as ti
import taichi_glsl as ts
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from helper import load_geometry_txt
from PIL import Image
import time

FORCE = 0.02  # membrane tension between each node on the geometry of the cell
SIGNAL_initalcondition = 1.0
SIGNAL_production = 1e-3
SIGNAL_decay = 1e-5
box = [190, 190, 210, 210]


@ti.data_oriented
class lbib_solver:
    def __init__(self, size_x, size_y, niu, bc_type, bc_value, input_geo, steps=60000):
        self.size_x = size_x
        self.size_y = size_y
        self.steps = steps
        self.rho = ti.field(dtype=ti.f32, shape=(size_x, size_y))
        self.niu = niu
        self.tau = 2  # 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(size_x, size_y))
        self.force = ti.Vector.field(2, dtype=ti.f32, shape=(size_x, size_y))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))

        # weight and direction vector in D2Q9
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.e = ti.field(dtype=ti.i32, shape=(9, 2))

        # Add CDE solver D2Q5
        self.cde_f_old = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        self.cde_f_new = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        # weight and direction vector in D2Q9
        self.cde_w = ti.field(dtype=ti.f32, shape=9)
        self.cde_e = ti.field(dtype=ti.i32, shape=(9, 2))
        self.c = ti.field(dtype=ti.f32, shape=(size_x, size_y))

        # Box boundary Dirichlet or Neumann
        # Dirichlet: when hiting a box, set the local velocity to a certain
        # value, calculate f_old (distribution in D2Q9) again
        # If velocity set to zero, it is essentially an open "fluid" or "gas"
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))

        # WARNING: after giving value to ti.field, no further variables can be
        # init. afterwards
        self.materialization(input_geo, bc_type, bc_value)

    def materialization(self, filename, bc_type, bc_value):
        try:
            boundary_points, _ = load_geometry_txt(filename)
            # fig, ax = plt.subplots()
            # ax.scatter(boundary_points[:, 0], boundary_points[:, 1])
            # fig.savefig('test.pdf', dpi=300)
        except IOError:
            print(IOError)
            exit()

        # exit()

        # TODO: store the BoundaryNode produced dynamically
        # TODO: Maybe use a large list and a hashing function will do the jobs, need to check how many
        # TODO: Divide and reconnet self.boundary
        # FIXME:
        self.num_connection = boundary_points.shape[0]
        self.boundary = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_connection,))
        self.boundary_force = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.num_connection,)
        )
        self.boundary_vel = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.num_connection,)
        )
        # the nodes are connected in clock wise manner, and closed
        self.boundary.from_numpy(boundary_points)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        # give default values after init all data structure
        arr = np.array(
            [
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
            ],
            dtype=np.float32,
        )
        self.w.from_numpy(arr)
        arr = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ],
            dtype=np.int32,
        )
        self.e.from_numpy(arr)

        # # weight and direction for D2Q5
        # arr = np.array([2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
        #                 1.0 / 6.0, 1.0 / 6.0], dtype=np.float32)
        # self.cde_w.from_numpy(arr)
        # arr = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
        # self.cde_e.from_numpy(arr)

    @ti.func
    def f_eq(self, i, j, k):
        eu = (
            ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0]
            + ti.cast(self.e[k, 1], ti.f32) * self.vel[i, j][1]
        )
        uv = self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

    @ti.func
    # speed would explode now
    def f_eq_with_force(self, i, j, k):
        force_term = (
            ti.cast(self.e[k, 0], ti.f32) * self.force[i, j][0]
            + ti.cast(self.e[k, 1], ti.f32) * self.force[i, j][1]
        )
        return self.w[k] * self.rho[i, j] * 3 * force_term

    @staticmethod
    @ti.func
    def calculate_discrete_delta_dirac(xpos, ypos):
        # cosine delta dirac function for IBM and LBM
        # for calculate the force in fluid
        return (
            (1.0 + ti.cos(0.5 * np.pi * xpos)) * (1.0 + ti.cos(0.5 * np.pi * ypos))
        ) / 16

    @ti.func
    def f_eq_cde(self, i, j, k):
        eu = (
            ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0]
            + ti.cast(self.e[k, 1], ti.f32) * self.vel[i, j][1]
        )
        uv = self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0
        return self.w[k] * self.c[i, j] * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

    @ti.func
    # speed would explode now
    def react(self, i, j, k):
        min_x, min_y, max_x, max_y = box
        production = -SIGNAL_decay * self.c[i, j]
        if (i >= min_x and i < max_x) and (j >= min_y and j < max_y):
            production = SIGNAL_production
        # return self.cde_w[k] * production
        return self.w[k] * production

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

    @ti.func
    def calculate_membrane_tension(self, i, j, force_const):
        # cannot use nested func
        # dist = ti.sqrt(
        #     (self.boundary[j][0] - self.boundary[j][0]) ** 2
        #     + (self.boundary[j][0] - self.boundary[j][0]) ** 2
        # )
        # self.boundary_force[i] += (
        #     (self.boundary[j] - self.boundary[i]) * force_const / dist
        # )  # for point i
        # self.boundary_force[j] += (
        #     (self.boundary[i] - self.boundary[j]) * force_const / dist
        # )  # for point j
        # dist = ts.vector.distance(self.boundary[i], self.boundary[j])
        # Use Taichi GLSL func
        self.boundary_force[i] += (
            ts.vector.normalize((self.boundary[j] - self.boundary[i])) * force_const
        )
        self.boundary_force[j] += (
            ts.vector.normalize((self.boundary[i] - self.boundary[j])) * force_const
        )

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.c[i, j] = 0.0

        min_x, min_y, max_x, max_y = box
        for i, j in ti.ndrange((min_x, max_x), (min_y, max_y)):
            self.c[i, j] = SIGNAL_initalcondition
            for k in ti.static(range(9)):
                self.cde_f_old[i, j][k] = SIGNAL_initalcondition / 9.0

        for i, j in self.rho:
            # fluid velocity initially zero
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.force[i, j][0] = 0.0
            self.force[i, j][1] = 0.0

            # density set to 1.0
            self.rho[i, j] = 1.0

            # essentially collide during init.
            for k in ti.static(range(9)):
                # init. no force so use f_eq
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]

            for k in ti.static(range(9)):
                self.cde_f_new[i, j][k] = self.f_eq_cde(i, j, k)
                self.cde_f_old[i, j][k] = self.cde_f_new[i, j][k]

    @ti.kernel
    def reset_boundary_force(self):
        for k in range(self.num_connection):
            self.boundary_force[k][0] = 0.0
            self.boundary_force[k][1] = 0.0

    @ti.kernel
    def calculate_force(self):
        for k in range(self.num_connection):
            # assume the boundary points are organized in circle-wise manner
            self.calculate_membrane_tension(k, (k + 1) % self.num_connection, FORCE)

    @ti.kernel
    def distribute_force(self):
        for k in range(self.num_connection):
            base = (self.boundary[k] - 2.0).cast(int)

            # Loop over 4x4 grid node neighborhood
            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                dpos = base + offset - self.boundary[k]
                delta_dirac = self.calculate_discrete_delta_dirac(dpos[0], dpos[1])
                # distribute force to the neighbour
                self.force[base + offset] += self.boundary_force[k] * delta_dirac

    @ti.kernel
    def collide_fluid(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            for k in ti.static(range(9)):
                # here it seems to do the advection
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                # fluid
                self.f_new[i, j][k] = (
                    (1.0 - self.inv_tau) * self.f_old[ip, jp][k]
                    + self.f_eq(ip, jp, k) * self.inv_tau
                    + self.f_eq_with_force(i, j, k)
                )
                # cde
                self.cde_f_new[i, j][k] = (1.0 - self.inv_tau) * self.cde_f_old[ip, jp][
                    k
                ] + self.f_eq_cde(
                    ip, jp, k
                ) * self.inv_tau  # + self.react(i, j, k)

    @ti.kernel
    def update_fluid_vel(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_y - 1)):
            # reset
            self.rho[i, j] = 0.0
            self.c[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.force[i, j][0] = 0.0
            self.force[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += ti.cast(self.e[k, 0], ti.f32) * self.f_new[i, j][k]
                self.vel[i, j][1] += ti.cast(self.e[k, 1], ti.f32) * self.f_new[i, j][k]
                # update cde solver concentration
                self.cde_f_old[i, j][k] = self.cde_f_new[i, j][k]
                self.c[i, j] += self.cde_f_new[i, j][k]

            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def collect_velocity_fast(self):
        for k in range(self.num_connection):
            self.boundary_vel[k][0] = 0.0
            self.boundary_vel[k][1] = 0.0

            base = (self.boundary[k] - 2.0).cast(int)

            # Loop over 4x4 grid node neighborhood
            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                dpos = base + offset - self.boundary[k]
                delta_dirac = self.calculate_discrete_delta_dirac(dpos[0], dpos[1])
                self.boundary_vel[k] += self.vel[base + offset] * delta_dirac

    @ti.kernel
    def move_geometry(self):
        for i in ti.ndrange((0, self.num_connection)):
            # for i in ti.static(range(self.num_connection)):
            self.boundary[i] += self.boundary_vel[i]

    @ti.func
    def apply_bc_core(self, dr, ibc, jbc, inb, jnb):
        # Apply boundary condition to lattice [Box grid only now]
        if self.bc_type[dr] == 0:  # Dirichlet
            self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
            self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
        elif self.bc_type[dr] == 1:  # Neumann
            self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
            self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]

        self.rho[ibc, jbc] = self.rho[inb, jnb]  # change local density
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = (
                self.f_eq(ibc, jbc, k)
                - self.f_eq(inb, jnb, k)
                + self.f_old[inb, jnb][k]
            )

        for k in ti.static(range(9)):
            self.cde_f_old[ibc, jbc][k] = (
                self.f_eq_cde(ibc, jbc, k)
                - self.f_eq_cde(inb, jnb, k)
                + self.cde_f_old[inb, jnb][k]
            )

    def draw_c(self):
        c = self.c.to_numpy()
        # c_img = np.stack((c_density, c_density, c_density, c_density), axis=2)
        c_img = cm.ScalarMappable(
            norm=mpl.colors.LogNorm(vmin=1e-20, vmax=1), cmap="Spectral"
        ).to_rgba(self.c.to_numpy())
        return c_img

    def draw_fluid_vel(self):
        vel = self.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        return vel_mag

    def render(self, gui):
        # code fragment displaying vorticity is contributed by woclass
        # https://github.com/taichi-dev/taichi/issues/1699#issuecomment-674113705 for img
        vel = self.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        vel_img = cm.plasma(vel_mag / 0.15)
        density = self.rho.to_numpy()
        den_img = np.stack((density, density, density, density), axis=2)
        c_density = self.c.to_numpy()
        # c_img = np.stack((c_density, c_density, c_density, c_density), axis=2)
        c_img = cm.ScalarMappable(
            norm=mpl.colors.LogNorm(vmin=1e-20, vmax=1), cmap="Spectral"
        ).to_rgba(self.c.to_numpy())
        zero = np.zeros_like(vel_img)
        img_r = np.concatenate((vel_img, den_img), axis=1)
        img_l = np.concatenate((c_img, c_img), axis=1)
        img = np.concatenate((img_l, img_r), axis=0)
        boundary = self.boundary.to_numpy()
        boundary[:, 0] /= 2 * self.size_x
        boundary[:, 1] /= 2 * self.size_y
        gui.set_image(img)
        gui.circles(boundary, radius=1)
        gui.show()

    def solve(self, render_dir=None):
        if render_dir:
            # gui = ti.GUI("LBIB Solver", (2 * self.size_x, 2 * self.size_y))
            video_manager = ti.VideoManager(
                output_dir=render_dir, framerate=24, automatic_build=False
            )
        self.init()
        for i in range(self.steps):
            self.reset_boundary_force()
            self.calculate_force()
            self.distribute_force()
            self.collide_fluid()
            self.update_fluid_vel()
            self.collect_velocity_fast()
            self.move_geometry()
            self.apply_bc()
            # if render_dir:
            #     self.render(gui)

            # if i % 50 == 0 and render_dir:
            if render_dir:
                # img = gui.get_image()  # cannot get image
                img = self.draw_c()
                self.draw_fluid_vel()
                print(f"\rFrame {i}/{self.steps} is recorded", end="")
                video_manager.write_frame(img)

        if render_dir:
            video_manager.make_video(gif=True, mp4=True)


if __name__ == "__main__":
    start = time.perf_counter()

    # Re = U*D/niu = 200
    # [0.0255 * (256 - 1)] / 0.01
    # see https://en.wikipedia.org/wiki/Reynolds_number
    # U, flow speed: 0.01
    # niu, kinematic viscosity, 0.01
    # L, domain length
    ti.init(arch=ti.gpu)
    # ti.init(cpu_max_num_threads=2)

    lbib = lbib_solver(
        400,
        400,
        0.0399,
        [0, 0, 0, 0],
        [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        # "data/parameters_400_by_400_radius_50_center_res_50.txt",
        # [[0.0, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 'parameters_400_by_400_radius_50_center_res_100.txt', steps=10000)
        # [[0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "data/parameters_400_by_400_radius_40_center_res_360.txt",
        steps=5000,
    )
    lbib.solve("results/")
    end = time.perf_counter()
    print("\nSimulation time for {:d} iter. : {:.2f}s".format(lbib.steps, end - start))
