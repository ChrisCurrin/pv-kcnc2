import os
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from neuron import h

pvParams = namedtuple("pvParams", "target_myelinated_L node_spacing node_length ais_L")


@lru_cache()
def get_pv(
    name="default",
    target_myelinated_L=1000.0,
    node_spacing=30.0,
    node_length=1.0,
    ais_L=60.0,
):
    """Create a parvalbumin-positive interneuron neuron.

    Note that this function is cached to reduce the number of neurons created by repeated calls.

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    morph_dir = str(Path(f"{dir_path}/morphologies").absolute()).replace("\\", "/")
    try:
        pv = h.pv(
            morph_dir,
            "C210401C.asc",
            target_myelinated_L,
            node_spacing,
            node_length,
            ais_L,
        )
        pv.name = (
            f"{name}({target_myelinated_L}, {node_spacing}, {node_length}, {ais_L})"
        )
    except AttributeError:
        pv_template = Path("PV_template.hoc").absolute()
        if not pv_template.exists():
            pv_template = Path(f"{dir_path}/PV_template.hoc").absolute()
        h.load_file(str(pv_template).replace("\\", "/"))

        pv = get_pv(name, target_myelinated_L, node_spacing, node_length, ais_L)

    return pv


def get_pv_params(pv):
    pv_full_name = pv.name
    p0 = pv_full_name.index("(")
    pv_name = pv_full_name[:p0]
    pv_params = pvParams(
        *[float(param) for param in pv_full_name[p0 + 1 : -1].split(", ")]
    )
    return pv_name, pv_params


if __name__ == "__main__":
    from src.nrn_helpers import init_nrn

    init_nrn()
    pv1 = get_pv()
    pv_same = get_pv()
    pv_diff = get_pv("test_pv_dif")
    assert pv1 == pv_same, "not the same object as excepted for the same call to pv()"
    assert (
        pv1 != pv_diff
    ), "expected different argument calls to produce different objects!"
