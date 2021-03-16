from numpy.core.fromnumeric import product
import taichi as ti
import taichi_glsl as ts
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from helper import load_geometry_txt, create_evenly_spaced_circle
from PIL import Image
import time

FORCE = 0.02  # membrane tension between each node on the geometry of the cell
SIGNAL_initalcondition = 0.0
SIGNAL_influx = 0.5
SIGNAL_production = 1e-1
SIGNAL_decay = 0
RES = 200  # fixed resolution for now
box = [190, 190, 210, 210]


@ti.data_oriented
class lbib_solver:
    def __init__(self, size_x, size_y, niu, bc_type, bc_value, input_geo, steps=60000):
        self.size_x = size_x
        self.size_y = size_y
        self.steps = steps
        self.rho = ti.field(dtype=ti.f32, shape=(size_x, size_y))
        self.celltype = ti.field(dtype=ti.i32, shape=(size_x, size_y))
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
            # boundary_points, _ = load_geometry_txt(filename)
            boundary_points = create_evenly_spaced_circle(
                self.size_x, self.size_y, 50, res=200
            )
        except IOError:
            print(IOError)
            exit()

        # TODO: store the BoundaryNode produced dynamically
        # TODO: Maybe use a large list and a hashing function will do the jobs, need to check how many
        # TODO: Divide and reconnet self.boundary
        # FIXME:
        self.num_connection = boundary_points.shape[0]
        self.boundary = ti.field(dtype=ti.f32, shape=(self.num_connection, RES, 2))
        # self.boundary_force = ti.field(
        #     dtype=ti.f32, shape=(self.num_connection, RES, 2)
        # )
        # self.boundary_vel = ti.field(dtype=ti.f32, shape=(self.num_connection, RES, 2))
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
    def is_point_inside_polygon(self, i, j, polygon_lst, idx, length):
        # https://github.com/taichi-dev/taichi/blob/master/taichi/math/geometry_util.h
        count = 0
        # qx, qy = 123532.0, 532421123.0
        a = ti.Vector([i, j])
        b = ti.Vector([123532.0, 532421123.0])
        for k in range(length):
            # assume the boundary points are organized in circle-wise manner
            # c = polygon[k]
            # d = polygon[(k + 1) % length]
            c = ti.Vector([polygon_lst[idx, k, 0], polygon_lst[idx, k, 1]])
            d = ti.Vector(
                [
                    polygon_lst[idx, (k + 1) % length, 0],
                    polygon_lst[idx, (k + 1) % length, 1],
                ]
            )

            term_1 = (c - a).cross(b - a) * (b - a).cross(d - a)
            term_2 = (a - d).cross(c - d) * (c - d).cross(b - d)
            if term_1 > 0 and term_2 > 0:  # see if intersect
                count += 1

        return count

    @ti.func
    def get_min_max_box_polygon(self, idx, length):
        min_x = float(self.size_x)
        min_y = float(self.size_y)
        max_x = 0.0
        max_y = 0.0
        for i in range(length):
            if min_x > self.boundary[idx, i, 0]:
                min_x = self.boundary[idx, i, 0]
            if max_x <= self.boundary[idx, i, 0]:
                max_x = self.boundary[idx, i, 0]
            if min_y > self.boundary[idx, i, 1]:
                min_y = self.boundary[idx, i, 1]
            if max_y <= self.boundary[idx, i, 1]:
                max_y = self.boundary[idx, i, 1]
        return (
            int(min_x),
            int(min_y),
            ti.cast(ti.ceil(max_x), int),
            ti.cast(ti.ceil(max_y), int),
        )

    @ti.kernel
    def mesh_lattice(self):
        for k in range(self.num_connection):
            # min_x, min_y, max_x, max_y = self.get_min_max_box_polygon(k, RES)
            for i, j in ti.ndrange((0, self.size_x), (0, self.size_y)):
                # for i, j in ti.ndrange((min_x, max_x), (min_y, max_y)):
                count = self.is_point_inside_polygon(i, j, self.boundary, k, RES)
                if count % 2 == 1:
                    self.celltype[i, j] = k
        # for i, j in ti.ndrange((0, self.size_x), (0, self.size_y)):
        #     # count = self.is_point_inside_polygon(i, j, self.boundary, 360)
        #     # if count % 2 == 1:
        #     #     self.celltype[i, j] = 0
        #     # else:
        #     #     self.celltype[i, j] = 1
        #     for k in ti.ndrange((0, self.num_connection)):
        #         count = self.is_point_inside_polygon(i, j, self.boundary, k, RES)
        #         if count % 2 == 1:
        #             self.celltype[i, j] = k

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
    # speed would explode now
    def influx_bottom(self):
        for j in ti.ndrange((0, self.size_y)):
            self.c[0, j] = SIGNAL_influx
            for k in ti.static(range(9)):
                self.cde_f_old[0, j][k] = SIGNAL_initalcondition / 9.0

    @ti.kernel
    def apply_bc(self):
        # left and right
        for j in ti.ndrange(1, self.size_y - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            # self.apply_bc_core(0, 0, j, 1, j)

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
            self.celltype[i, j] = 0

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
                self.cde_f_new[i, j][k] = (
                    (1.0 - self.inv_tau) * self.cde_f_old[ip, jp][k]
                    + self.f_eq_cde(ip, jp, k) * self.inv_tau
                    + self.react(i, j, k)
                )

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
            self.rho[ibc, jbc] = self.rho[inb, jnb]  # change local density
        elif self.bc_type[dr] == 1:  # Neumann
            self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
            self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
            self.rho[ibc, jbc] = self.rho[inb, jnb]  # change local density
        elif self.bc_type[dr] == 2:  # iso-pressure wall condition
            inb, jnb = ibc, jbc  # density stay
            rho = self.rho[ibc, jbc]  # keep constant pressure at 1.0, speed untouched
            for k in ti.static(range(9)):
                self.f_old[ibc, jbc][k] /= rho

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

    def draw_c_vor_vel(self):
        SHIFT = 1e-10  # prevent zero in LogNorm case
        vel = self.vel.to_numpy()
        c = self.c.to_numpy()
        c += SHIFT
        ugrad = np.gradient(vel[:, :, 0])
        vgrad = np.gradient(vel[:, :, 1])
        vor = ugrad[1] - vgrad[0]
        colors = [
            (1, 1, 0),
            (0.953, 0.490, 0.016),
            (0, 0, 0),
            (0.176, 0.976, 0.529),
            (0, 1, 1),
        ]
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
        vor_img = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
        ).to_rgba(vor)
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        vel_img = cm.plasma(vel_mag / 0.15)
        c_img_lognorm = cm.ScalarMappable(
            norm=mpl.colors.LogNorm(vmin=SHIFT, vmax=1 + SHIFT),
            cmap="coolwarm",
        ).to_rgba(c)
        c_img_linear = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=SHIFT, vmax=1 + SHIFT),
            cmap="coolwarm",
        ).to_rgba(c)
        img_r = np.concatenate((vel_img, vor_img), axis=1)
        img_l = np.concatenate((c_img_lognorm, c_img_linear), axis=1)
        img = np.concatenate((img_l, img_r), axis=0)
        return img

    def render(self, gui):
        # code fragment displaying vorticity is contributed by woclass
        # https://github.com/taichi-dev/taichi/issues/1699#issuecomment-674113705 for img
        vel = self.vel.to_numpy()
        ugrad = np.gradient(vel[:, :, 0])
        vgrad = np.gradient(vel[:, :, 1])
        vor = ugrad[1] - vgrad[0]
        colors = [
            (1, 1, 0),
            (0.953, 0.490, 0.016),
            (0, 0, 0),
            (0.176, 0.976, 0.529),
            (0, 1, 1),
        ]
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
        vor_img = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
        ).to_rgba(vor)
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        vel_img = cm.plasma(vel_mag / 0.15)
        # density = self.rho.to_numpy()
        # den_img = np.stack((density, density, density, density), axis=2)
        # c_density = self.c.to_numpy()
        # c_img = np.stack((c_density, c_density, c_density, c_density), axis=2)
        c_img = cm.ScalarMappable(
            norm=mpl.colors.LogNorm(vmin=1e-20, vmax=1),
            # norm=mpl.colors.Normalize(vmin=0, vmax=1),
            cmap="coolwarm"
            # cmap="Spectral"
        ).to_rgba(self.c.to_numpy())
        mask = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=self.num_connection), cmap="coolwarm"
        ).to_rgba(self.celltype.to_numpy())
        img_r = np.concatenate((vel_img, vor_img), axis=1)
        img_l = np.concatenate((mask, c_img), axis=1)
        img = np.concatenate((img_l, img_r), axis=0)
        boundary = self.boundary.to_numpy()
        boundary = boundary.reshape(self.num_connection * RES, 2)
        boundary[:, 0] /= 2 * self.size_x
        boundary[:, 1] /= 2 * self.size_y
        gui.set_image(img)
        gui.circles(boundary, radius=1)
        gui.show()

    def solve(self, render_dir=None, show_gui=False):
        if show_gui:
            gui = ti.GUI("LBIB Solver", (2 * self.size_x, 2 * self.size_y))

        if render_dir:
            video_manager = ti.VideoManager(
                output_dir=render_dir, framerate=24, automatic_build=False
            )
        self.init()
        self.mesh_lattice()
        for i in range(self.steps):
            # self.reset_boundary_force()
            # self.calculate_force()
            # self.distribute_force()
            self.collide_fluid()
            self.update_fluid_vel()
            # self.collect_velocity_fast()
            # self.move_geometry()
            # self.mesh_lattice()
            # self.influx_bottom()
            self.apply_bc()
            self.mesh_lattice()

            if i % 50 == 0:
                print(f"\rSteps {i}/{self.steps}", end="")
                # np.save("saved_c_{:d}.npy".format(i), self.c.to_numpy())
                # np.save("saved_rho_{:d}.npy".format(i), self.rho.to_numpy())
                # np.save("saved_boundary_{:d}.npy".format(i), self.boundary.to_numpy())

            if show_gui:
                self.render(gui)

            if render_dir:
                img = self.draw_c_vor_vel()
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
    # ti.init(arch=ti.gpu)
    ti.init(cpu_max_num_threads=4)

    lbib = lbib_solver(
        400,
        400,
        0.0399,
        [0, 0, 0, 0],
        [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.1]],
        # "data/parameters_400_by_400_radius_50_center_res_50.txt",
        # [[0.0, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 'parameters_400_by_400_radius_50_center_res_100.txt', steps=10000)
        # [[0.0, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "data/parameters_400_by_400_radius_40_center_res_360.txt",
        steps=500,
    )
    # lbib.solve(show_gui=1)
    lbib.solve()
    end = time.perf_counter()
    print("\nSimulation time for {:d} iter. : {:.2f}s".format(lbib.steps, end - start))
