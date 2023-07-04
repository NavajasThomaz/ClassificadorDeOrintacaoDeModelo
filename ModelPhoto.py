import os
import cv2
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.collections import PolyCollection
from OBJtools import *

def OBJ(PathU,PathL):
    V, F = [], []
    with open(PathL) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                F.append([int(x) for x in values[1:4]])
    V, F = np.array(V), np.array(F) - 1
    V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))

    MVP = perspective(25, 1, 1, 100) @ translate(0, 0, -3.5) @ xrotate(20) @ yrotate(45)

    V = np.c_[V, np.ones(len(V))] @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    V = V[F]

    T = V[:, :, :2]
    Z = -V[:, :, 2].mean(axis=1)

    zmin, zmax = Z.min(), Z.max()

    Z = (Z - zmin) / (zmax - zmin)
    C = pyplot.get_cmap("magma")(Z)

    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]

    #fig = pyplot.figure(figsize=(6, 6))
    #ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
    collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
    #ax.add_collection(collection)
    V, F = [], []
    with open(PathU) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                F.append([int(x) for x in values[1:4]])
    V, F = np.array(V), np.array(F) - 1
    V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))

    MVP = perspective(25, 1, 1, 100) @ translate(0, 0, -3.5) @ xrotate(20) @ yrotate(45)

    V = np.c_[V, np.ones(len(V))] @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    V = V[F]

    T = V[:, :, :2]
    Z = -V[:, :, 2].mean(axis=1)

    zmin, zmax = Z.min(), Z.max()

    Z = (Z - zmin) / (zmax - zmin)
    C = pyplot.get_cmap("magma")(Z)

    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]

    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
    collection1 = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
    ax.add_collection(collection)
    ax.add_collection(collection1)
    pyplot.show()

def STL(PathU,PathL,nameL,nameU):
    #pyplot.title(f"Teste STL {nameL[10]}")
    figure = pyplot.figure(f"{nameL}")
    pyplot.axis('off')
    pyplot.rcParams['axes.grid'] = False

    axes = mplot3d.Axes3D(figure)
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    auto_add_to_figure = False

    your_mesh = mesh.Mesh.from_file(PathL)
    your_mesh2 = mesh.Mesh.from_file(PathU)

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh2.vectors))

    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    os.mkdir(f'E:/TensorFlowPyCharm/Prints/STL/{nameL}')
    #figure.suptitle(f"Teste STL {nameL[10]} Visão Superior", fontsize=28)
    axes.view_init(90, -90)
    pyplot.savefig(f'E:/TensorFlowPyCharm/Prints/STL/{nameL}/{nameL}_VisãoSuperior.jpg')
    #pyplot.show()
    #figure.suptitle(f"Teste STL {nameL[10]} Visão Frontal", fontsize=28)
    axes.view_init(0, 90)
    pyplot.savefig(f'E:/TensorFlowPyCharm/Prints/STL/{nameL}/{nameL}_VisãoTraseira.jpg')
    #pyplot.show()
    #figure.suptitle(f"Teste STL {nameL[10]} Visão LateralE", fontsize=28)
    axes.view_init(0, 0)
    pyplot.savefig(f'E:/TensorFlowPyCharm/Prints/STL/{nameL}/{nameL}_VisãoLateralE.jpg')
    # pyplot.show()



def imgSimetry(path):
    view_data = []
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        view_data.append(img_array)
        IMG_SIZE = 50
        #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        pyplot.imshow(view_data, cmap='gray')

DATADIR = "E:\TensorFlowPyCharm\Modelos"
CATEGORIES = ["STL","OBJ"]
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for folder in os.listdir(path):
        PathU = None
        PathL = None
        Path1 = os.path.join(path, folder)
        for models in os.listdir(Path1):
            name = str(models)
            if category == "STL":
                if PathL == None:
                    nameL = name
                    PathL = f'E:\TensorFlowPyCharm\Modelos\STL\{folder}\{nameL}'
                    #Path = f'E:/TensorFlowPyCharm/Prints/STL/{nameL[10]}'
                else:
                    nameU = name
                    PathU = f'E:\TensorFlowPyCharm\Modelos\STL\{folder}\{nameU}'

                    #STL(PathU,PathL,nameL,nameU)
                    #imgSimetry(Path)
            elif category == "OBJ":
                if PathL == None:
                    nameL = name
                    PathL = f'E:\TensorFlowPyCharm\Modelos\OBJ\{folder}\{nameL}'
                    # Path = f'E:/TensorFlowPyCharm/Prints/STL/{nameL[10]}'
                else:
                    nameU = name
                    PathU = f'E:\TensorFlowPyCharm\Modelos\OBJ\{folder}\{nameU}'
                    OBJ(PathU,PathL)
