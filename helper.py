import numpy as np
from vtk import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt


def load_geometry_txt(filename):
    # now skip the first header and the ignore the connections
    res, connection = None, None
    try:
        f_out = open(filename, "r")
        x, y = [], []
        connection_0, connection_1 = [], []
        for i, line in enumerate(f_out):
            line = line.rstrip("\n").split("\t")
            if i == 0:
                continue
            if line[0] == "#Nodes (id":
                break
            x.append(line[1])
            y.append(line[2])

        found_connection = False
        for i, line in enumerate(f_out):
            line = line.rstrip("\n").split("\t")
            if found_connection:
                connection_0.append(line[0])
                connection_1.append(line[1])
            if line[0] == "#Connection (nodeId1":
                found_connection = True
                continue

        res = np.array([x, y], dtype=np.float32)
        res = res.T
        connection = np.array([connection_0, connection_1], dtype=np.int32)
        connection = connection.T
        connection -= 1  # zero-based in orignal files
        f_out.close()
    except IOError:
        print("{} cannot be opened".format(filename))
        raise IOError("Geometry {} cannot be loaded.".format(filename))

    return res, connection


def create_points_on_cell(radius, center_x, center_y, res):
    x_pos, y_pos = [], []

    for i in range(res):
        angle = (2 * np.pi) * (i / res)
        x = center_x + (radius * np.cos(angle))
        y = center_y + (radius * np.sin(angle))
        x_pos.append(x)
        y_pos.append(y)

    return x_pos, y_pos


def create_evenly_spaced_circle(size_x, size_y, radius, res=200):
    num_x = size_x // (2 * radius + 2)
    num_y = size_y // (2 * radius + 2)
    # print(num_x * num_y)
    boundary = np.ones(shape=(num_x, num_y, res, 2), dtype=np.float32)
    tmp = np.ones(shape=(num_x, num_y, 2), dtype=np.float32)

    for i in range(num_x):
        for j in range(num_y):
            center_x = (2 * i + 1) * radius + radius / 5 * (i + 1)
            center_y = (2 * j + 1) * radius + radius / 5 * (j + 1)
            # print(center_x, center_y, i, j)
            # print(i + j)
            x_pos, y_pos = create_points_on_cell(radius, center_x, center_y, res)
            boundary[i, j, :, 0] = x_pos
            boundary[i, j, :, 1] = y_pos
            tmp[i, j, 0] = center_x
            tmp[i, j, 1] = center_y

    # return tmp.reshape(num_x * num_y, 2)
    return boundary.reshape(num_x * num_y, res, 2)


def laod_vtp(filename):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polyData = reader.GetOutput()
    polyDataPointData = polyData.GetPointData()
    nbOfCells = polyData.GetNumberOfPolys()

    a = []

    for i in range(nbOfCells):
        cell = polyData.GetCell(i)
        # need to iter. thr. all Points of that polygon
        nbOfPoints = cell.GetNumberOfPoints()
        a.append(nbOfPoints)
        # for i in range(nbOfPoints):
        #     Id = cell.GetPointIds().GetId(i)
        #     cell_type.SetValue(Id, celltype_id)

    # return mat

    print(max(a))


if __name__ == "__main__":
    # laod_vtp("thin_200_id_2/Cells_0/Cells_0_0.vtp")

    a = create_evenly_spaced_circle(400, 400, 50, res=200)
    print(a.shape)
    # print(a)
    a = a.reshape(a.shape[0] * a.shape[1], 2)
    # plt.scatter([1, 2, 1, 2], [1, 1, 2, 2])
    plt.scatter(a[:, 0], a[:, 1])
    plt.show()