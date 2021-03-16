import taichi as ti
from functools import wraps
from time import time


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it


start = int(round(time() * 1000))
ti.init(arch=ti.cpu)
num = 2 ** 10
print("length: {}".format(num))
b = ti.Vector.field(2, dtype=ti.f32, shape=(num,))
c = ti.Vector.field(2, dtype=ti.i32, shape=(num, num))
end_ = int(round(time() * 1000)) - start
print(f"Init. ti.Matrix.field: {end_ if end_ > 0 else 0} ms")


# TOO SLOW!!!
@measure
@ti.kernel
def create_static():
    # b = ti.Vector.field(2, dtype=ti.i32, shape=(num, ))
    for i in ti.static(range(num)):
        b[i][0] += 2
        b[i][1] += 2


@measure
@ti.kernel
def create_ndrange():
    # for i in ti.ndrange((0, num)):
    for i in range(num):
        inc = ti.random()
        b[i][0] += i + inc
        b[i][1] += i + inc


@measure
@ti.kernel
def create_ndrange_block_dim():
    ti.block_dim(1024)
    # for i in ti.ndrange((0, num)):
    for i in range(num):
        b[i][0] += 2
        b[i][1] += 2


@ti.func
def dist(vec, i):
    return ti.cast(ti.sqrt(vec[i][0] ** 2 + vec[i][1] ** 2), ti.i32)


@measure
@ti.kernel
def create_ndrange_func():
    ti.block_dim(1024)
    for i in ti.ndrange((0, num)):
        b[i][0] = dist(b, i)


# create_static()
create_ndrange()
# create_ndrange_block_dim()
# create_ndrange_func()


@ti.kernel
def create():
    # for i in ti.ndrange((0, num)):
    for i, j in ti.ndrange((0, num), (0, num)):
        c[i, j][0] = i
        c[i, j][1] = j


@ti.kernel
def get_nei():
    base = (b[123] - 2.0).cast(int)
    fx = b[123] - base.cast(float)
    print(b[123][0], b[123][1])
    print(base[0], base[1])
    print(fx[0], fx[1])
    for i, j in ti.static(ti.ndrange(4, 4)):  # Loop over 3x3 grid node neighborhood
        offset = ti.Vector([i, j])
        dpos = offset.cast(float) - fx
        tmp = c[base + offset]
        print(tmp[0], tmp[1])


create()
get_nei()
