import taichi as ti
import taichi_glsl as ts
import numpy as np

# https://taichi.readthedocs.io/en/stable/syntax.html
# see pass by value for ti.func
deltaT = 1.0
SWITCHOFF_TIME = 5000
SIGNAL_decay = 1e-5
SIGNAL_production = 1e-4
SIGNAL_initalcondition = 1.0

EPSILON = 10E-10  # The delta used in double comparisons
PERTURBATION = 10E-6  # The perturbation used to avoid clashes


# Now with CDE D2Q5 tutorial 1
# CDE would only change when the domain_id != 0 (not water)
# Connection help identify inside or outside

@ti.data_oriented
class lbibcell_solver:
    """Solver Warpper for LBM (2D)

    Attributes
    ----------
    rho : ti.field
        Fluid rensity at each lattice, shape=(size_x, size_y), equal to the sum of the distribution at [i, j]
    tau : float
        Relaxation time, constant
    inv_tau : float
        Invertion of tau
    domain_id : ti.field
        Integer indicator for fluid, shape=(size_x, size_y)
    f_old : ti.Vector
        Particle distribution in 9 direction in previous time step,             ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
    f_new : ti.Vector
        Particle distribution in 9 direction in next time step,                 ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
    fluid_force : ti.Vector
        Physical nodes get a force from the neighbouring geometry node          ti.Vector.field(2, dtype=ti.f32, shape=(size_x, size_y))
    fluid_vel : ti.Vector
        Fluid velocity ti.Vector.field(2, dtype=ti.f32, shape=(size_x, size_y))
    weight : ti.field
        weighting factor corresponding geometry of the lattice (D2Q9), shape=9 page 31
    lattice_dir : ti.Vector
        Lattice streaming directions (D2Q9), shape=(9, 2) page 101
    lattice_boundary_neighbours : ti.Matrix

    lattice_boundary_neighbours_domain_id : ti.Vector

        
    c : ti.field
        Concentration distribution for the signalling molecules
    cde_f_old : ti.Vector
        Particle distribution in 5 direction in previous time step, ti.Vector.field(5, dtype=ti.f32, shape=(size_x, size_y))
    cde_f_new : ti.Vector
        Particle distribution in 5 direction in next time step,     ti.Vector.field(5, dtype=ti.f32, shape=(size_x, size_y))
    cde_weight : ti.field
        weighting factor corresponding geometry of the lattice (D2Q5), shape=5
    cde_lattice_dir : ti.Vector
        Lattice streaming directions (D2Q5), shape=(5, 2)
    """

    def __init__(self, size_x, size_y, niu, debug=False, steps=10000):
        self.debug = debug
        self.size_x = size_x
        self.size_y = size_y
        self.steps = steps
        self.domain_id = ti.field(dtype=ti.f32, shape=(size_x, size_y))
        # set fluid viscocity constant
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        ################################################################################
        #                                     LBM                                      #
        ################################################################################
        self.rho = ti.field(dtype=ti.f32, shape=(size_x, size_y))

        self.fluid_vel = ti.Vector.field(
            2, dtype=ti.f32, shape=(size_x, size_y))
        self.fluid_force = ti.Vector.field(
            2, dtype=ti.f32, shape=(size_x, size_y))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(size_x, size_y))

        self.weight = ti.field(dtype=ti.f32, shape=9)
        self.lattice_dir = ti.field(dtype=ti.i32, shape=(9, 2))

        ################################################################################
        #                                     CDE                                      #
        ################################################################################
        # each physical grid has pointer to its boundary nodes
        self.lattice_boundary_neighbours = ti.Matrix.field(
            5, 2, dtype=ti.f32, shape=(size_x, size_y))
        self.lattice_boundary_neighbours_domain_id = ti.Vector.field(
            5, dtype=ti.i32, shape=(size_x, size_y))

        self.cde_f_old = ti.Vector.field(
            5, dtype=ti.f32, shape=(size_x, size_y))
        self.cde_f_new = ti.Vector.field(
            5, dtype=ti.f32, shape=(size_x, size_y))

        self.cde_weight = ti.field(dtype=ti.f32, shape=5)
        self.cde_lattice_dir = ti.field(dtype=ti.i32, shape=(5, 2))

        ################################################################################
        #                                      IB                                      #
        ################################################################################

        # Start with a triangle in the middle
        # the nodes are connected in clock wise manner
        self.init_geo = ti.Vector.field(2, dtype=ti.f32, shape=(4,))
        # self.init_geo = ti.field(dtype=ti.f32, shape=(3, 2))
        # self.init_geo_connections = ti.Vector.field(2, dtype=ti.f32, shape=(3,))

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

        # weight and direction for D2Q5
        arr = np.array([2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
                        1.0 / 6.0, 1.0 / 6.0], dtype=np.float32)
        self.cde_weight.from_numpy(arr)
        arr = np.array(
            [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
        self.cde_lattice_dir.from_numpy(arr)

        # init geo
        arr = np.array([[self.size_x / 2, self.size_y / 2],
                        [self.size_x * 7 / 16, self.size_y * 5 / 8],
                        [self.size_x / 2, self.size_y * 3 / 4],
                        [self.size_x * 5 / 8, self.size_y * 5 / 8]], dtype=np.float32)
        # arr = np.array([[25, 25], [20, 30], [25, 35], [30, 30]], dtype=np.float32)
        self.init_geo.from_numpy(arr)

    # see https://raw.githubusercontent.com/taichi-dev/taichi/0298bca363d7eb51201d53985836d066bb0ffbff/examples/odop_solar.py
    # for random
    @staticmethod
    @ti.func
    def perturb_connections(vec, i, j):
        # TODO: should I put a if here ??
        # TODO: check precision latter for ti.f32
        # assume i != j
        # vec = ti.Vector.field(2, dtype=ti.f32, shape=(n))
        # index_vec = ti.Vector.field(2, dtype=ti.i32, shape=(n))
        # equals to the connection in LBIBCell scope
        if (vec[i][0] == vec[j][0]):  # x_i == x_j change to
            tmp_rand = -PERTURBATION + ti.random() * 2 * PERTURBATION
            while (tmp_rand < 5.0 * EPSILON):
                tmp_rand = -PERTURBATION + ti.random() * 2 * PERTURBATION
            vec[i][0] += tmp_rand

        if (vec[i][1] == vec[j][1]):  # y_i == y_j
            tmp_rand = -PERTURBATION + np.random.rand() * 2 * PERTURBATION
            while (tmp_rand < 5.0 * EPSILON):
                tmp_rand = -PERTURBATION + ti.random() * 2 * PERTURBATION
            vec[i][1] += tmp_rand

    @ti.func
    def generate_boundary_nodes_on_lattice_grid(self, vec, i, j):
        # get two points from vector (vec[i][0], vec[i][1]) (vec[j][0], vec[j][0])
        # obtain a striaght line, get all the nodes intercepted with lattice grid y = ax + b
        # the polygona connections is clockwise: Vector i->j clockwise
        # between every nodes, create nodes intercepted by lattice grid

        a = (vec[j][1] - vec[i][1]) / (vec[j][0] - vec[i][0])
        b = vec[j][1] - a * vec[j][0]
        min_x = ti.ceil(ti.min(vec[i][0], vec[j][0]))
        max_x = ti.floor(ti.max(vec[i][0], vec[j][0]))
        min_y = ti.ceil(ti.min(vec[i][1], vec[j][1]))
        max_y = ti.floor(ti.max(vec[i][1], vec[j][1]))

        print('[Debug!] Generating boundary nodes...')
        # TODO: change
        # unroll matrix, recommended way
        # for x, y in self.lattice_boundary_neighbours:
        # for x in ti.static(range(self.size_x)): # too slow
        count = 0
        for x in ti.ndrange((0, self.size_x)):
            if x < min_x: continue
            elif x > max_x: continue
            # get all nodes intercepted with vectical grid lines
            y_hat = a * x + b
            idx_y_s = ti.cast(ti.floor(y_hat), ti.int32)
            idx_y_n = ti.cast(ti.ceil(y_hat), ti.int32)

            # ERROR: here
            # TODO: condition on inside outside is more difficult
            if (vec[i][0] < vec[j][0]):
                # print('[Debug!] North is inside')
                # north is inside N == 2 S == 4, Dont forget to inverse that !!
                self.lattice_boundary_neighbours[x, idx_y_n][4, 0] = x
                self.lattice_boundary_neighbours[x, idx_y_n][4, 1] = y_hat
                self.lattice_boundary_neighbours_domain_id[x, idx_y_n][4] = 1

                self.lattice_boundary_neighbours[x, idx_y_s][2, 0] = x
                self.lattice_boundary_neighbours[x, idx_y_s][2, 1] = y_hat
                self.lattice_boundary_neighbours_domain_id[x, idx_y_n][2] = 0
            else:
                # print('[Debug!] South is inside')
                # south is inside N == 2 S == 4, Dont forget to inverse that !!
                self.lattice_boundary_neighbours[x, idx_y_n][4, 0] = x
                self.lattice_boundary_neighbours[x, idx_y_n][4, 1] = y_hat
                self.lattice_boundary_neighbours_domain_id[x, idx_y_n][4] = 0

                self.lattice_boundary_neighbours[x, idx_y_s][2, 0] = x
                self.lattice_boundary_neighbours[x, idx_y_s][2, 1] = y_hat
                self.lattice_boundary_neighbours_domain_id[x, idx_y_n][2] = 1


        for y in ti.ndrange((0, self.size_y)):
            if y < min_y: continue
            elif y > max_y: continue
            # get all nodes intercepted with horizontal grid lines
            x_hat = (y - b) / a
            idx_x_w = ti.cast(ti.floor(x_hat), ti.int32)
            idx_x_e = ti.cast(ti.ceil(x_hat), ti.int32)
            if (vec[i][1] > vec[j][1]):
                # east is inside E == 1 W == 3, Dont forget to inverse that !!
                self.lattice_boundary_neighbours[idx_x_e, y][3, 0] = x_hat
                self.lattice_boundary_neighbours[idx_x_e, y][3, 1] = y
                self.lattice_boundary_neighbours_domain_id[idx_x_e, y][3] = 1

                self.lattice_boundary_neighbours[idx_x_w, y][1, 0] = x_hat
                self.lattice_boundary_neighbours[idx_x_w, y][1, 1] = y
                self.lattice_boundary_neighbours_domain_id[idx_x_w, y][1] = 0

                # update the domain id
                # print('[Debug!] Updating domain id to 1 at [{}, {}]'.format(idx_x_e, y))
                # self.domain_id[idx_x_e, y] = 1
                count += 1
            else:
                # west is inside E == 1 W == 3, Dont forget to inverse that !!
                self.lattice_boundary_neighbours[idx_x_e, y][3, 0] = x_hat
                self.lattice_boundary_neighbours[idx_x_e, y][3, 1] = y
                self.lattice_boundary_neighbours_domain_id[idx_x_e, y][3] = 0

                self.lattice_boundary_neighbours[idx_x_w, y][1, 0] = x_hat
                self.lattice_boundary_neighbours[idx_x_w, y][1, 1] = y
                self.lattice_boundary_neighbours_domain_id[idx_x_w, y][1] = 1  


        print(count, '\t')
        print('boundary nodes added to the east of the geo node:')
        # print('{} boundary nodes added to the east of the geo node: [bound --- physical]'.format(count))
        
    @ti.func
    def f_eq_d2q9(self, i, j, k):
        eu = ti.cast(self.lattice_dir[k, 0], ti.f32) * self.fluid_vel[i, j][0] \
            + ti.cast(self.lattice_dir[k, 1], ti.f32) * self.fluid_vel[i, j][1]
        uv = self.fluid_vel[i, j][0]**2.0 + self.fluid_vel[i, j][1]**2.0
        return self.weight[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)
       
    @ti.func
    def update_all_domain_id(self):
        for i, j in ti.ndrange((0, self.size_x), (0, self.size_x)):
            if self.lattice_boundary_neighbours_domain_id[i, j][3] != 0:
                # print('[Debug!] Updating domain id to 1 at [{}, {}]'.format(i, j))
                self.domain_id[i, j] = 1
            # else:
                # print('[Debug!] Not updating domain id to 1 at [{}, {}]'.format(i, j))


    @ti.kernel
    def cure_lattice(self):
        # cureLattice
        # 1. perturbConnections all the original geo nodes will be deleted
        # 2. generateBoundaryNodes and store in self.lattice_boundary_neighbours
        # see https://github.com/taichi-dev/taichi/blob/a58a9af78b3c2708178db11499ff0796f8ce338f/benchmarks/mpm2d.py
        # np.random.seed(0)  # set seed before pertuabtion
        # TODO: change the hack now to loop get the connection from the last one to the first one        
        print('[Debug!] Perturbating: {} and {}'.format(self.init_geo.shape[0] - 1, 0))
        self.perturb_connections(self.init_geo, self.init_geo.shape[0] - 1, 0)
        self.generate_boundary_nodes_on_lattice_grid(self.init_geo, self.init_geo.shape[0] - 1, 0)
        for i in ti.static(range(self.init_geo.shape[0])):
            for j in ti.static(range(self.init_geo.shape[0])):
                if (i % self.init_geo.shape[0] + 1 != j % self.init_geo.shape[0]):
                    # ERROR with continue, for some reason continue does not work
                    pass
                else:
                    print('[Debug!] Perturbating: {} and {}'.format(i, j))
                    self.perturb_connections(self.init_geo, i, j)
                    self.generate_boundary_nodes_on_lattice_grid(self.init_geo, i, j)
                    # reinit CDE solver
        
        # 3. update all domain identifiers of the
        self.update_all_domain_id()

    # @ti.kernel
    # def pre_advect_cde(self):
    #     # only one cde for now
    #     # for every physical nodes get their neibour
    #     for i, j in self.lattice_boundary_neighbours:
    #         print()
    #     pass

    @ti.kernel
    def advect():
        pass

    @ti.kernel
    def init_cde_solver(self):
        for i, j in ti.ndrange((0, self.size_x), (0, self.size_x)):
            if self.domain_id[i, j] == 0:
                for k in ti.static(range(5)):
                    self.cde_f_old[i, j][k] = 0.0
                    self.cde_f_new[i, j][k] = 0.0
            else:
                for k in ti.static(range(5)):
                    self.cde_f_old[i, j][k] = SIGNAL_initalcondition / 5.0
                    self.cde_f_new[i, j][k] = SIGNAL_initalcondition / 5.0
                
    @ti.kernel
    def init_fluid_solver(self):
        # TODO: add force
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.lattice_dir[k, 0]
                jp = j - self.lattice_dir[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] \
                                    + self.f_eq_d2q9(ip, jp, k) * self.inv_tau    
    @ti.func
    def get_cde_c(self, i, j):
        return ts.vector.summation(self.cde_f_new[i, j])

    @ti.kernel
    def fluid_collide(self):
        # TODO: add force
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.lattice_dir[k, 0]
                jp = j - self.lattice_dir[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] \
                                    + self.f_eq_d2q9(ip, jp, k) * self.inv_tau

    @ti.kernel
    def test_ti_scope(self):
        self.cde_f_new[1,2][4] = 5.0
        print(self.get_cde_c(1,2))



    def debug_show(self):
        filename = f'frame_lbibcell.png'   # create filename with suffix png
        # change the res to integer
        # gui = ti.GUI('lbm solver', (self.size_x, self.size_y))
        # print('[Debug!] before')
        self.cure_lattice()
        self.test_ti_scope()
        # print('[Debug!] after')
        img = self.domain_id.to_numpy()
        print(np.min(img), np.max(img))
        # img[0:50] = img[0:5].astype(numpy.float32)
        # gui.set_image(img)
        # gui.show(filename)  # export and show in GUI
        print(f'Frame is recorded in {filename}')
        ti.imwrite(self.domain_id.to_numpy(), filename)

if __name__ == '__main__':
        ti.init(arch=ti.gpu)
        lbm = lbibcell_solver(500, 500, 0.01, 100)
        lbm.debug_show()
