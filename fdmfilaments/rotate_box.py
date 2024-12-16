import pathlib
import argparse
import h5py
import jax.numpy as jnp
from jax.config import config
import numpy as onp

import yt

from yt.visualization.volume_rendering.api import (
    Scene,
    Camera,
    TransferFunctionHelper,
    create_volume_source,
)

from unyt import unyt_array

from fuzzylli.domain import UniformHypercube
from fuzzylli.units import set_schroedinger_units as set_units
from fuzzylli.cosmology import h

yt.enable_parallelism()
# config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser(description="")

parser.add_argument("filename", type=str, help="path to HDF5 file")
args = parser.parse_args()

filename = args.filename
file = h5py.File(filename, "r")
stem = pathlib.Path(filename).stem


z = file.attrs["z"]
L = file.attrs["L_Mpc_div_h"] / h
m22 = file.attrs["m22_1e-22eV"]
M = file.attrs["M"]
file.close()

u = set_units(m22)
domain = UniformHypercube([M, M, M], jnp.array([[0, L], [0, L], [0, L]]))

ds = yt.loaders.load_hdf5_file(
    filename,
    root_node="density/fdm",
    bbox=onp.array(domain.extends) * u.from_Mpc,
    dataset_arguments={
        "length_unit": 1.0 * u.to_cm,
        "mass_unit": 1.0 * u.to_g,
        "time_unit": 1.0 * u.to_s,
    },
)

sc = yt.create_scene(ds.all_data(), ("stream", "0"), "perspective")
source = sc.get_source()
source.set_log(True)

bounds = (2e-2, 1e2)

tf = yt.ColorTransferFunction(onp.log10(bounds), grey_opacity=False)


def quadramp(vals, minval, maxval):
    return 4 * ((vals - vals.min()) / (vals.max() - vals.min())) ** 2


tf.map_to_colormap(
    onp.log10(bounds[0]),
    onp.log10(bounds[1]),
    colormap="gist_heat",
    scale_func=quadramp,
)
tf.add_layers(10, colormap="gist_heat", alpha=onp.geomspace(0.015, 10, 10))
source.tfh.tf = tf
source.tfh.bounds = bounds

camera = sc.camera
camera.position = [0.7, 1.8, 1.8]
camera.focus = [0.5, 0.5, 0.5]
camera.resolution = (800, 800)
sc.annotate_domain(ds, color=[1, 1, 1, 0.015])
frame = 0
for _ in camera.iter_rotate(
    2 * onp.pi,
    360,
    rot_vector=[1, 0, 0],
    rot_center=unyt_array([0.5, 0.5, 0.5], camera.focus.units),
):
    sc.save(f"../img/rotate_box_{stem}_{frame}.png", sigma_clip=2.2)
    frame += 1
