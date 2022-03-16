import taichi as ti

ti.init(
    random_seed=42,
    arch=ti.cpu,
    debug=0,
    advanced_optimization=0,
    excepthook=True,
    cpu_max_num_threads=1
)

real = ti.f32
scalar = lambda **kwargs: ti.field(dtype=real, shape=1, **kwargs)
vector = lambda **kwargs: ti.Vector.field(dim, dtype=real, **kwargs)

# ti.reset() # reset ti kernel to be allowed to init variables

dt = scalar(needs_grad=False)
