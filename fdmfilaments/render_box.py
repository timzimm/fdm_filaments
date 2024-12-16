import pathlib
import argparse
import jax.numpy as np
from jax.config import config
import numpy as onp
import h5py

import yt
from yt.visualization.volume_rendering.api import (
    LineSource,
)
from fuzzylli.domain import UniformHypercube
from fuzzylli.units import set_schroedinger_units as set_units
from fuzzylli.cosmology import h

yt.enable_parallelism()
config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser(description="")

parser.add_argument("filename", type=str, help="path to HDF5 file")
args = parser.parse_args()
stem = pathlib.Path(args.filename).stem

file = h5py.File(args.filename, "r")
m22 = file.attrs["m22_1e-22eV"]
u = set_units(m22)
z = file.attrs["z"]
L = file.attrs["L_Mpc_div_h"] / h
M = file.attrs["M"]
file.close()
domain = UniformHypercube([M, M, M], np.array([[0, L], [0, L], [0, L]]))

ds = yt.loaders.load_hdf5_file(
    args.filename,
    root_node="density/fdm",
    bbox=onp.array(domain.extends) * u.from_Mpc,
    dataset_arguments={
        "length_unit": 1.0 * u.to_cm,
        "mass_unit": 1.0 * u.to_g,
        "time_unit": 1.0 * u.to_s,
    },
)
for i in [0]:
    sc = yt.create_scene(ds.all_data(), ("stream", f"{i}"), "perspective")
    source = sc.get_source()
    source.set_log(True)
    bounds = (2e-2, 1e2)

    phi = onp.linspace(0, 2 * onp.pi, 1000)
    for R in onp.array([0.05, 0.1]) * u.from_Mpc * h:
        vertex = onp.array(
            [
                [
                    [
                        0.75 * L,
                        L / 2 + R * onp.sin(phi[i]),
                        L / 2 + R * onp.cos(phi[i]),
                    ],
                    [
                        0.75 * L,
                        L / 2 + R * onp.sin(phi[i + 1]),
                        L / 2 + R * onp.cos(phi[i + 1]),
                    ],
                ]
                for i in range(0, len(phi) - 1)
            ]
        )

        c = onp.ones([vertex.shape[0], 4])
        c[:, 3] = 0.01

        lines = LineSource(vertex, c)
        lines.set_zbuffer(0)
        sc.add_source(lines)

    tf = yt.ColorTransferFunction(onp.log10(bounds), grey_opacity=False)

    def quadramp(vals, minval, maxval):
        return ((vals - vals.min()) / (vals.max() - vals.min())) ** 2

    tf.map_to_colormap(
        onp.log10(bounds[0]),
        onp.log10(bounds[1]),
        colormap="gist_heat",
        scale_func=quadramp,
    )
    tf.sample_colormap(onp.log10(40), 0.004, colormap="gist_heat", alpha=6)
    tf.sample_colormap(onp.log10(19), 0.004, colormap="gist_heat", alpha=1.5)
    tf.sample_colormap(onp.log10(7), 0.025, colormap="gist_heat", alpha=0.7)
    tf.sample_colormap(onp.log10(1), 0.07, colormap="gist_heat", alpha=0.28)
    source.tfh.tf = tf
    source.tfh.bounds = bounds

    camera = sc.camera
    camera.position = [1.6, 1.9, 1.9]
    camera.focus = [L / 2 + L / 3, L / 2, L / 2]
    camera.north_vector = [1, 0, 0]
    camera.resolution = (2 * 500, 2 * 840)
    camera.zoom(3.5)
    camera.switch_orientation()
    sc.save(f"../img/volume_rendering_{stem}_{i}.png", sigma_clip=2.5, render=True)
