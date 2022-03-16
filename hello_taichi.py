# first some imports
import taichi as ti

import IPython
import numpy as np

# Matplotlib settings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

ti.init(
    random_seed=42,
    arch=ti.cpu,
    debug=1,
    advanced_optimization=0,
    excepthook=1,
    cpu_max_num_threads=1
)
ti.root.lazy_grad()
"""
# Taichi code requirements:

1. no continue/break in loops
2. only either one for loop or a group of other statements per indentation level
3. no recursive assignments of form x = f(x)
6. some index updating requirements, see documentation, ti.atomic_add
4. init all ti.Vector/ti.Field vars outside of kernel
5. all taichi code must be called into from a kernel function (decorated with @ti.kernel),
    which may call into inner taichi functions (decorated with @ti.func)

# for grad stuff, see
# https://docs.taichi.graphics/docs/lang/articles/advanced/differentiable_programming
6. the set_* kernels above must be outside of the tape scope, presumably inside tape scope the params
    w.r.t. which I autodiff may not be written to anymore by me? It could also be that its just
    the ti.random calls (those dont work in friction either!)
"""

# init ti vars

# reset: use this in jupyter notebook or click reload all

# ------- BASIC SIMULATION HYPERPARAMETERS --------

DEBUG = True

dim = 2
steps = 50

height = 550
width_to_height_ratio = 16/9.
width = int(width_to_height_ratio * height)
DY = 1/height # difference is smaller of height, width

num_species = 6
particles_per_species = 20
N = num_species * particles_per_species
Ndim = N * dim

lr = 0.01

# ti parameter type helper funcs
real = ti.f32

# 1. hyperparameters:
scalar = lambda **kwargs: ti.field(dtype=real, **kwargs, shape=())
# 2. fields (arrays) of scalar or vector for each particle
scalars = lambda **kwargs: ti.field(dtype=real, **kwargs, shape=[N])
vectors = lambda **kwargs: ti.Vector.field(dim, dtype=real, **kwargs, shape=[steps, N])

# ------- MORE SIMULATION HYPERPARAMETERS --------
dt = scalar(needs_grad=False)

params_need_grad = False

# particle parameters
repelling_force = scalar(needs_grad=params_need_grad) # force active if r < separation radius
temperature = scalar(needs_grad=params_need_grad) # controls random fluctuations in particle velocities -> increases gradient variance
friction_coef = scalar(needs_grad=params_need_grad)
separation_radius = scalar(needs_grad=params_need_grad) # mean separation radius
interaction_radius = scalar(needs_grad=params_need_grad) # mean interaction radius
force_strength = scalar(needs_grad=params_need_grad) # inter-particle force strength
close_range_factor = scalar(needs_grad=params_need_grad) # force strength multiplier at r=0
dist_range_factor = scalar(needs_grad=params_need_grad) # force strength multiplier at r=self.height
stdev = scalar(needs_grad=params_need_grad) # spread in species parameters -> increases gradient variance
seed_range = scalar(needs_grad=params_need_grad) # initial position spread

params = {
    "dt": dt,
    "repelling_force": repelling_force,
    "temperature": temperature,
    "friction_coef": friction_coef,
    "separation_radius": separation_radius,
    "interaction_radius": interaction_radius,
    "force_strength": force_strength,
    "close_range_factor": close_range_factor,
    "dist_range_factor": dist_range_factor,
    "stdev": stdev,
}

P = len(params)

arrays_need_grad = False
species = np.array([[spec]*particles_per_species for spec in range(num_species)]).reshape(-1)

# interparticle forces (species a, species b)
interaction_params = lambda: ti.field(dtype=real, needs_grad=True, shape=[N,N])

force_radii = interaction_params()
separation_ = interaction_params()
repulsive_f = interaction_params()
intrprtcl_f = interaction_params()


# Game state:
R = vectors(needs_grad=arrays_need_grad)
V = vectors(needs_grad=arrays_need_grad)
F = vectors(needs_grad=arrays_need_grad)

# loss
complexity = ti.field(dtype=real, needs_grad=True, shape=[])

arrays = {
    "force_radii": force_radii,
    "separation_": separation_,
    "intrprtcl_f": intrprtcl_f,
    "repulsive_f": repulsive_f,
    "R": R,
    "V": V,
    "F": F,
    "complexity": complexity
}


# ---------------- place ti vars -----------------

# def place_ti_vars():
#     assert False
#
#     # ti.reset() # reset ti kernel to be allowed to init variables
#     # ti.root.deactivate_all()
#     # print(ti.root.get_children())
#     # print(dir(ti.root))
#
#     # place scalars
#     ti.root.place(
#         *(list(params.values()) + [complexity])
#     )
#
#     # place N x 2 vectors
#     # e.g. V = [steps, N] x 2
#     ti.root.dense(ti.l, steps).dense(ti.i, N).place(
#         V, R, F
#     )
#     ti.root.dense(ti.ij, (N, N)).place(
#         force_radii, separation_, intrprtcl_f, repulsive_f
#     )
#
#     ti.root.lazy_grad()
#


# initialization kernels
idx = lambda spec, prtcl: num_species * prtcl + spec

@ti.kernel
def set_ti_scalars():
    dt[None] = 0.02
    seed_range[None] = 0.9 # initial position spread

    # set particle parameters
    repelling_force[None] = 35.0 # force active if r < separation_radius
    temperature[None] = 5.0 # controls random fluctuations in particle velocities -> increases gradient variance
    friction_coef[None] = 90.0
    separation_radius[None] = 6.0 # mean separation radius
    interaction_radius[None] = 20.0 # mean interaction radius
    force_strength[None] = 30.0 # inter-particle force strength
    close_range_factor[None] = 3.0 # force strength multiplier at r=0
    dist_range_factor[None] = 0.01 # force strength multiplier at r=self.height
    stdev[None] = 0.1 # spread in species parameters

    # scalar objective function
    complexity[None] = 0.

@ti.kernel
def set_ti_vectors():
    # takes in R, V, F and updates them, t should be 0
    # t: ti.i32 = 0
    t = 0
    # center_x = width/2
    # center_y = height/2
    for prtcl in range(N):
        R[t, prtcl][0] = ti.random() * width # + width/(2*seed_range)
        R[t, prtcl][1] = ti.random() * height # + height/(2*seed_range)
        V[t, prtcl][0] = ti.random() * 2 - 1 # -1 to 1
        V[t, prtcl][1] = ti.random() * 2 - 1

@ti.func
def particle_assignment_loop(spec_a, spec_b, fr, sep, f, rep):
    pps = particles_per_species
    for prtcl_a, prtcl_b in ti.ndrange(pps, pps):
        # assign
        force_radii[idx(spec_a, prtcl_a), idx(spec_b, prtcl_b)] = fr
        separation_[idx(spec_a, prtcl_a), idx(spec_b, prtcl_b)] = sep
        intrprtcl_f[idx(spec_a, prtcl_a), idx(spec_b, prtcl_b)] = f
        repulsive_f[idx(spec_a, prtcl_a), idx(spec_b, prtcl_b)] = rep

@ti.kernel
def set_block_matrices():
    # parameter matrices
    for spec_a, spec_b in ti.ndrange(num_species, num_species):
        # sample once for this species pair direction

        fr = ti.abs(interaction_radius[None] + ti.randn() * (interaction_radius[None] * stdev[None]))
        sep = ti.abs(separation_radius[None] + ti.randn() * (separation_radius[None] * stdev[None]))
        f = ti.randn() * force_strength[None]
        rep_force = ti.abs(repelling_force[None] + (ti.randn() * stdev[None]) * repelling_force[None])

        particle_assignment_loop(spec_a, spec_b, fr, sep, f, rep_force)

# helper funcs

@ti.func
def wrap_borders(x: ti.template(), y: ti.template()):
    x_ = x % width
    y_ = y % height
    if DEBUG:
        assert x_ <= width
        assert y_ <= height
        assert 0 <= x_
        assert 0 <= y_
    return x_, y_

@ti.kernel
def compute_complexity(t: ti.i32):
    # just macro temperature for now; FIXME make grid out of this
    # params R, V, F
    for prtcl in range(N):
        vx = V[t, prtcl][0]
        vy = V[t, prtcl][1]
        vsquare = vx * vx + vy * vy

        # ti.atomic_add(complexity[None], float(vsquare / Ndim))
        complexity[None] += vsquare/float(Ndim)

@ti.kernel
def apply_grad():
    # gradient ascent on scalar parameters
    # params[i][None] -= lr * params[i].grad[None] # for loop doesnt seem to work
    repelling_force[None] += lr * repelling_force.grad[None]
    temperature[None] += lr * temperature.grad[None]
    friction_coef[None] += lr * friction_coef.grad[None]
    separation_radius[None] += lr * separation_radius.grad[None]
    interaction_radius[None] += lr * interaction_radius.grad[None]
    force_strength[None] += lr * force_strength.grad[None]
    close_range_factor[None] += lr * close_range_factor.grad[None]
    dist_range_factor[None] += lr * dist_range_factor.grad[None]
    stdev[None] += lr * stdev.grad[None]


@ti.func
def _incr_force(force_array: ti.template(), magnitude: float, dy: float, dx: float):
    # increment force_array (which is/may be row in a [? x 2]) vector field)
    theta = ti.atan2(dy, dx)
    ti.atomic_add(force_array[0], magnitude * ti.cos(theta))
    ti.atomic_add(force_array[1], magnitude * ti.sin(theta))

# physical kernels: attract, friction, update_system

@ti.kernel
def attract(t: ti.i32):

    # params R (read from t-1)
    # updates F (write to t)
    for a, b in ti.ndrange(N, N): # <-- gets vectorized <3
        # for b in range(N): # <--- gets serialized :-(

        ra, rb = R[t-1, a], R[t-1, b]

        dx, dy = rb[0] - ra[0], rb[1] - ra[1]
        dx, dy = wrap_borders(dx, dy)

        f = 0.
        if a != b:
            eps = 1e-5
            xdoty = dx * dx + dy * dy
            dr = ti.sqrt(xdoty) # euclidean distance

            if dr < separation_[a, b]:
                f = repulsive_f[a, b] * (separation_[a, b]-dr)
            elif (dr > separation_[a,b]) & (dr < (separation_[a,b] + force_radii[a,b]/2)):
                f = intrprtcl_f[a,b] * (dr - separation_[a,b])
            elif (dr > separation_[a,b] + force_radii[a, b]/2) & (dr < separation_[a,b] + force_radii[a,b]):
                f = -intrprtcl_f[a,b] * (dr - (separation_[a,b] + force_radii[a,b]))

            # inv_dr_2 = 1/(xdoty+eps)

            f *= close_range_factor[None] + (dist_range_factor[None]-close_range_factor[None]) * (dr * DY)
            # increase force at long range again
            # f += (dist_range_factor[None] - close_range_factor[None]) * (dr * DY)
            # inter = intrprtcl_f[a, b]
            # f *= inter # weight by randomly sampled interparticle force (depends on species)
            # print("f=",f, "inter=", inter)
            # print("Ra=",ra[0], ra[1])
            # print("Rb=",rb[0], rb[1])
            # print("inter=", inter)
            # print("CRF=", close_range_factor[None]*inv_dr_2)
            # print("F=", f)
            # print("="*20)

        # add contributions from each particle b to entry a
        _incr_force(F[t, a], f, dy, dx)

@ti.kernel
def friction(t: ti.i32):
    # params V, R (reads from t-1)
    # updates F (writes to t)

    # this loop gets vectorized:
    for prtcl in range(N):
        vx, vy = V[t-1, prtcl]
        v = ti.sqrt(vx * vx + vy * vy)
        theta = ti.atan2(vy, vx)

        # random doesnt seem to work in tape scope
        ffx = friction_coef[None] * v * ti.cos(theta) + ti.randn() * temperature[None]
        ffy = friction_coef[None] * v * ti.sin(theta) + ti.randn() * temperature[None]

        ti.atomic_add(F[t, prtcl][0], ffx)
        ti.atomic_add(F[t, prtcl][1], ffy)


@ti.kernel
def update_system(t: ti.i32):
    # params R, V (reads from t-1), F (reads from t); dt
    # updates R, V (writes to t)

    # this loop gets vectorized:
    for prtcl in range(N):
        x = R[t-1, prtcl][0]
        y = R[t-1, prtcl][1]

        dx = V[t-1, prtcl][0] * dt[None]
        dy = V[t-1, prtcl][1] * dt[None]

        xnew, ynew = x + dx, y + dy
        xnew, ynew = wrap_borders(xnew, ynew)
        R[t, prtcl][0] = xnew
        R[t, prtcl][1] = ynew

        fx = F[t, prtcl][0]
        fy = F[t, prtcl][1]

        V[t, prtcl][0] = V[t-1, prtcl][0] - fx * dt[None]
        V[t, prtcl][1] = V[t-1, prtcl][1] - fy * dt[None]


        if prtcl <= 5:
            # print("prtcl", prtcl)
            print("dx", dx)
            print("dy", dy)
            # print("vx", vx)
            # print("vy", vy)
            print("fx", fx)
            print("fy", fy)
            print("x", xnew)
            print("y", ynew)
            print("="*20)


def step(t: ti.i32):
    # simulation

    # calculate particle forces
    attract(t)
    # calculate friction
    friction(t)
    # update positions and forces
    update_system(t)

    if GUI:
        # TODO add color channels,  see colorfolor examples
        grid = np.zeros((width, height, 4))
        for spec, p in ti.ndrange(num_species, particles_per_species):
            prtcl = idx(spec, p)
            x = int(R[t, prtcl][0])
            y = int(R[t, prtcl][1])

            if prtcl <= 5:
                print("prtcl", prtcl)
                print("gui x", R[t, prtcl][0])
                print("gui y", R[t, prtcl][1])

            channels = 3 # RGB only
            saturation_step = 100
            color = spec % channels
            saturation = ((spec // channels) +1) * saturation_step
            grid[x, y, color] += saturation

        gui.set_image(grid)
        gui.show()
    else:
        particles_display = np.zeros(shape=(N, dim), dtype=np.float32)
        for prtcl in range(N):
            particles_display[prtcl, 0] = R[t, prtcl][0]
            particles_display[prtcl, 1] = R[t, prtcl][1]

        plt.xlim([0,width])
        plt.ylim([0,height])
        plt.scatter(particles_display[:,0], particles_display[:,1], marker=".", c="b")
        plt.show() # show each frame, comment to show only final frame

# main animate/forward functions that call into ti.kernels
GUI = 1
if GUI:
    gui = ti.GUI('Diff', res=(width, height))

def run():
    # do simulation for many steps
    print(f"Simulating {steps} steps ...")
    if GUI:
        t = 1
        while gui.running and not gui.get_event(gui.ESCAPE) and t < steps:
            step(t)
            t += 1
    else:
        for t in range(1,steps+1):
            step(t)

def report():
    for name, p in params.items():
        # grad = p.grad
        # grad = p.grad.to_numpy().nonzero() if p.grad else None
        print(f"Param {name} grad: {grad}")
    for name, arr in arrays.items():
        # grad = arr.grad.to_numpy().nonzero() if arr.grad is not None else None
        print(f"{name}\t grad: {grad}")
        print(f"{name}\t     : {arr.to_numpy().nonzero()}")
    print(f"Complexity: {complexity.to_numpy().nonzero()}")

# this function calls into the ti.kernels:
def update_params():
    # place_ti_vars()
    set_ti_scalars()
    set_ti_vectors()
    set_block_matrices()

    print("force_radii=",force_radii.to_numpy())
    print("seperation_=",separation_.to_numpy())
    print("intrprtcl_f=",intrprtcl_f.to_numpy())
    print("repulsive_f=",repulsive_f.to_numpy())

    # report()
    # within this context: update param.grad using autodiff of complexity w.r.t each param

    # run()
    with ti.Tape(complexity):
        run()
        compute_complexity(steps)
    # ^ setting complexity score
    # v and then exiting ti.Tape scope sets param.grad

    # apply_grad() # gradient ascent update
    # report()

if __name__ == "__main__":
    update_params()
