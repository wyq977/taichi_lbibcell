import taichi as ti
import numpy as np
from aux import test_func

mat = np.ones(shape=(1000, 300, 2), dtype=np.float32)
# mat = np.random.rand(1000, 300, 2)
# mat.astype(np.float32)

a = ti.field(dtype=ti.f32, shape=(1000, 300, 2))
# b = ti.Matrix.field(300, 2, dtype=ti.f32, shape=(1000,)) # not recommand for large matrix

a.from_numpy(mat)


@ti.kernel
def create_ndrange():
    # for i in ti.ndrange((0, num)):
    # for i, j in a:
    #     inc = ti.random()
    #     print(i, j)
    #     # print(a[i, j][1, 1])
    #     # b[i][0] += i + inc
    #     # b[i][1] += i + inc
    for i in b:
        inc = ti.random()
        print(i)
        # print(a[i, j][1, 1])
        # b[i][0] += i + inc
        # b[i][1] += i + inc


@ti.kernel
def create_field():
    # for i in ti.ndrange((0, num)):
    # for i, j in a:
    #     inc = ti.random()
    #     print(i, j)
    #     # print(a[i, j][1, 1])
    #     # b[i][0] += i + inc
    #     # b[i][1] += i + inc
    for i, j, k in a:
        inc = ti.random()
        a[i, j, k] = test_func()
        # print(i, j, k)
        # print(a[i, j][1, 1])
        # b[i][0] += i + inc
        # b[i][1] += i + inc


# create_ndrange()
create_field()
print(a[1], a[999, 299, 0])
print(a[999, 299, 0])
print(a[999, 299, 2])
print(a[999, 299, 4])
print(a.shape)

test = a.to_numpy()
test_1 = test.reshape(1000 * 300, 2)
print(test_1.shape)

# tmp = ti.Vector(a[999, 299, 0:1])