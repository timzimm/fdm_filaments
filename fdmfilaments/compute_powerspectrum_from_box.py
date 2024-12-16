import gc
import socket
from pathlib import Path

import logging
import argparse
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import h5py

from jax import config

from fuzzylli.domain import UniformHypercube
from fuzzylli.io_utils import create_ds

from correlation_functions_from_box import powerspectrum, density_contrast

config.update("jax_enable_x64", True)


logging.basicConfig(
    level=logging.INFO,
    format="\x1b[33;20m%(asctime)s {}\x1b[0m: %(message)s".format(socket.gethostname()),
)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

f = h5py.File(args.filename, "r+", driver="mpio", libver="latest", comm=MPI.COMM_WORLD)
f.atomic = False
n = f["/"].attrs["M"]
L = f["/"].attrs["L_Mpc_div_h"]
domain = UniformHypercube([n, n, n], np.array([[0, L], [0, L], [0, L]]))
if rank == 0:
    logger.info(f"Domain: [{n}, {n}, {n}], [{L:.2f}, {L:.2f}, {L:.2f}] Mpc/h")


density_datasets = []


def get_ds_names(name, node):
    if isinstance(node, h5py.Dataset):
        density_datasets.append(node)


P_grp = f.create_group("powerspectrum")

f["density"].visititems(get_ds_names)
if rank == 0:
    logger.info(f"Found {density_datasets}")
for dset in density_datasets:
    if rank == 0:
        logger.info(f"Compute P(k) for {dset.name}")

    N = np.array([n, n, n], dtype=int)
    fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float64)
    delta = newDistArray(fft, False)
    cart_comm = comm.Create_cart(
        dims=delta.commsizes, periods=[False, False, False], reorder=False
    )
    ij = cart_comm.Get_coords(rank)
    N_loc = delta.shape
    if rank == 0:
        logger.info(
            f"Local pencil shape = {N_loc} ({delta.size * delta.itemsize/ 1e9} GByte)"
        )
    with dset.collective:
        delta[:] = dset[
            ij[0] * N_loc[0] : (ij[0] + 1) * N_loc[0],
            ij[1] * N_loc[1] : (ij[1] + 1) * N_loc[1],
            :,
        ]
    comm.Barrier()

    delta = density_contrast(delta, comm)
    comm.Barrier()

    if rank == 0:
        logger.info("Compute FFT")
    delta_k = fft.forward(delta)
    del delta
    del fft
    gc.collect()
    comm.Barrier()

    if rank == 0:
        logger.info("Compute P(k)")
    k, P_k = powerspectrum(delta_k, L, cart_comm)
    k_P_k = np.vstack([k, P_k]).T
    dset_P = create_ds(P_grp, Path(dset.name).stem, k_P_k.shape)

    if rank == 0:
        logger.info(f"Save P(k) in {dset_P.name}")
        dset_P[:, :] = k_P_k
    del delta_k
    gc.collect()
    comm.Barrier()

f.close()
