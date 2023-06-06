import numpy as np
import matplotlib.pyplot as plt
import ctypes
import time

lib = ctypes.cdll.LoadLibrary('./libcavity.so')


class Config(object):
    def __init__(self, nx, ny, nt, nit, dt, rho, nu):
        self.obj = lib.conf_new(
            ctypes.c_int32(nx),
            ctypes.c_int32(ny),
            ctypes.c_int32(nt),
            ctypes.c_int32(nit),
            ctypes.c_double(dt),
            ctypes.c_double(rho),
            ctypes.c_double(nu)
        )
        self.nx = nx
        self.ny = ny
        self.nt = nt


class Simulator(object):
    def __init__(self, conf):
        self.obj = lib.simu_new(conf.obj)
        self.conf = conf
        self.c_array_p = ctypes.POINTER(ctypes.c_double * (conf.nx * conf.ny))

    def update(self):
        lib.simu_update(self.obj)

    def get_u(self):
        ptr = ctypes.cast(lib.simu_get_u(self.obj), self.c_array_p)
        mat = np.ctypeslib.as_array(ptr.contents).reshape((conf.ny, conf.nx))
        return mat

    def get_v(self):
        ptr = ctypes.cast(lib.simu_get_v(self.obj), self.c_array_p)
        mat = np.ctypeslib.as_array(ptr.contents).reshape((conf.ny, conf.nx))
        return mat

    def get_p(self):
        ptr = ctypes.cast(lib.simu_get_p(self.obj), self.c_array_p)
        mat = np.ctypeslib.as_array(ptr.contents).reshape((conf.ny, conf.nx))
        return mat


conf = Config(
    nx=41,
    ny=41,
    nt=500,
    nit=50,
    dt=.01,
    rho=1,
    nu=.02,
)

x = np.linspace(0, 2, conf.nx)
y = np.linspace(0, 2, conf.ny)
X, Y = np.meshgrid(x, y)

simu = Simulator(conf)

sum_time = 0.0
for n in range(conf.nt):
    start = time.time()
    simu.update()
    end = time.time()
    sum_time += end - start
    print(f'step {n:0>3} : took {end - start} s')

    u = simu.get_u()
    v = simu.get_v()
    p = simu.get_p()

    plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.pause(.01)
    plt.clf()
plt.show()

print(f'time per step : {sum_time / n}')
