from itertools import product
import numpy as np
import hppfcl as fcl

import jax.numpy as jnp
from jax.random import split, normal, PRNGKey

from fdmfilaments.parameters_from_ellipsoidal_collapse import (
    cylinder_length_physical_Mpc,
    cylinder_scale_radius_physical_Mpc,
    cylinder_sqrt_v_dispersion_physical_kms,
    sample_mass_from_powerlaw_dn_dM,
)
from fuzzylli.domain import UniformHypercube
from fuzzylli.poisson_disk import PeriodicPoissonDisk


def effify(non_f_str: str):
    return eval(f'f"""{non_f_str}"""')


def sample_from_poisson_disk(N, d, domain, seed):
    sampler = PeriodicPoissonDisk(*domain.L, d)
    return jnp.asarray(sampler.sample(N))


def sample_from_S(N, d, key):
    xyz = normal(key, shape=(N, d))
    directions = xyz / jnp.linalg.norm(xyz)
    return directions


def R_from(a, b):
    v = jnp.cross(a, b)
    s = jnp.linalg.norm(v)
    c = jnp.dot(a, b)
    M = jnp.cross(v, jnp.eye(3) * -1)
    return jnp.eye(3) + M + (1 - c) / s**2 * M @ M


def init_finite_straight_filament_spine_gas(N, r, l, seed, domain, lengths):
    origins = []
    directions = []
    cylinders = []
    translations = UniformHypercube(
        [3, 3, 3], jnp.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    )

    key = PRNGKey(seed)

    i = 0
    new_cylinder = fcl.Cylinder(float(r), float(l))
    sampler = PeriodicPoissonDisk(*domain.L, 2 * jnp.max(domain.L) / N, seed=seed)
    while len(origins) < N and i < 1000:
        key_dir, key_orient, key = split(key, 3)
        origin = sampler.sample(1)
        direction = sample_from_S(1, domain.dim, key_dir).squeeze()
        R = np.asarray(R_from(jnp.array([0, 0, 1]), direction))
        collide = False
        for c in cylinders:
            for v in domain.L * translations.cell_centers_cartesian:
                new_cylinder_t = fcl.Transform3f(R, np.asarray(origin + v))

                new_coll_obj = fcl.CollisionObject(new_cylinder, new_cylinder_t)

                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()

                collide = fcl.collide(c, new_coll_obj, request, result)
            if collide:
                break

        if not collide:
            new_cylinder_t = fcl.Transform3f(R, np.asarray(origin))
            new_coll_obj = fcl.CollisionObject(new_cylinder, new_cylinder_t)
            origins.append(origin)
            directions.append(direction)
            cylinders.append(new_coll_obj)
        i = i + 1
    origins = jnp.asarray(origins)
    directions = jnp.asarray(directions)

    N = origins.shape[0]

    # Orientation of cylinders
    v = sample_from_S(N, 2, key)
    v1 = jnp.cross(directions, origins)
    v2 = jnp.cross(directions, v1)

    # Basis vectors in plane perpendicular to directions
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    orientations = v[:, 0, jnp.newaxis] * v1 + v[:, 1, jnp.newaxis] * v2
    orientations = orientations / jnp.linalg.norm(orientations)
    return origins, directions, orientations


np.set_printoptions(threshold=np.inf)
jnp.set_printoptions(threshold=jnp.inf)

# Constant parameters
prefix = "./data/cylinder_gas/"
cache = "./data"
load_if_cached = False
save_cdm = True
save_fdm = True
save_wdm = False
seed = 20
problem = "aLASSO"
mass_per_particle = 1e5

N = 1024
L = 2.0
domain = UniformHypercube(
    [1, 1, 1],
    jnp.array([[0.0, L], [0.0, L], [0.0, L]]),
)

z = 4.0
beta = 0.0
Ncyl = 8
mass = sample_mass_from_powerlaw_dn_dM(Ncyl, 3e9, seed=seed)
sigma = cylinder_sqrt_v_dispersion_physical_kms(mass, beta)
r0 = cylinder_scale_radius_physical_Mpc(mass, beta)
length = cylinder_length_physical_Mpc(mass) * (1 + z)
print(length)
epochs = Ncyl * [20000]
R_min = Ncyl * [1e-4]
Nlib = Ncyl * [2**10]
beta = Ncyl * [beta]

d = np.max(length)
r = 10 * np.max(r0)

origin, direction, orientation = init_finite_straight_filament_spine_gas(
    Ncyl, r, d, 42, domain, length
)
center = jnp.sum(origin, axis=0) / Ncyl
origin = (origin - center + jnp.array([L / 2, L / 2, L / 2])) / L
origin = origin - jnp.array([0.0, 0.0, 0.1])

mass = list(np.asarray(mass))
sigma = list(np.asarray(sigma))
r0 = list(np.asarray(r0))
length = list(np.asarray(length))
orientation = np.array2string(orientation, separator=",")
origin = np.array2string(origin, separator=",")
direction = np.array2string(direction, separator=",")

m22s = [1.0]
print(jnp.array(mass) / 1e9)

for (m22,) in product(m22s):
    config_name = f"L_{L:.2f}_N_{N}_Ncyl_{Ncyl}_m22_{m22:.2f}_{problem}_seed_{seed}"
    output_file = f"{prefix}{config_name}.h5"

    with open("../parameters/template.yaml", "r") as file:
        template = file.read()
        config = effify(template)

    config_file = f"../parameters/{config_name}.yaml"
    print(f"Save {config_file}")
    with open(config_file, "w") as file:
        file.write(config)
