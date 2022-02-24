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


# different biophysical properties (by default, reset_biophys uses what is in PV_template.hoc)


def reset_biophys(pv, display=False):
    """Change biophysical properties of a "pv" cell"""
    pv.biophys()

    base_nav = {
        "somatic": pv.soma[0].gNav11bar_Nav11,
        "ais": pv.axon[0].gNav11bar_Nav11,
        "nodes": pv.node[0].gNav11bar_Nav11,
    }
    return base_nav


def reset_biophys_alt1(pv):
    pv.biophys()

    NaK_ratio_ais = 2.2
    NaK_ratio_nodes = 3.0
    nav_ais_nodes_ratio = NaK_ratio_nodes / NaK_ratio_ais

    nav_ais = 1.0
    kv3_ais = nav_ais / NaK_ratio_ais

    nav_nodes = nav_ais * nav_ais_nodes_ratio
    kv3_nodes = nav_nodes / NaK_ratio_nodes

    print(f"{nav_ais:.2f} | {kv3_ais:.2f} | {nav_nodes:.2f} | {kv3_nodes:.2f}")

    for sec in pv.somatic:
        sec.gNaTs2_tbar_NaTs2_t = 0.95585841724208476
        sec.gNav11bar_Nav11 = 0.11504623309972959
        sec.gSKv3_1bar_SKv3_1 = 0.18497365407533689

    for sec in pv.ais:
        sec.gNav11bar_Nav11 = nav_ais
        sec.gSKv3_1bar_SKv3_1 = kv3_ais

    for sec in pv.nodes:
        sec.gNav11bar_Nav11 = nav_nodes
        sec.gSKv3_1bar_SKv3_1 = kv3_nodes

    base_nav = {
        "somatic": pv.soma[0].gNav11bar_Nav11,
        "ais": pv.axon[0].gNav11bar_Nav11,
        "nodes": pv.node[0].gNav11bar_Nav11,
    }
    #     print(f"base gnav1.1 (soma, ais, node)={base_nav11_soma:.5f}, {base_nav11_ais:.5f}, {base_nav11_node:.5f}")
    return base_nav


def reset_biophys_alt2(pv):
    pv.biophys()

    NaK_ratio_ais = 2.5
    NaK_ratio_nodes = 2.5
    nav_ais_nodes_ratio = NaK_ratio_nodes / NaK_ratio_ais

    nav_ais = 0.8
    kv3_ais = nav_ais / NaK_ratio_ais

    nav_nodes = nav_ais * nav_ais_nodes_ratio
    kv3_nodes = nav_nodes / NaK_ratio_nodes

    print(f"{nav_ais:.2f} | {kv3_ais:.2f} | {nav_nodes:.2f} | {kv3_nodes:.2f}")

    for sec in pv.somatic:
        sec.gNaTs2_tbar_NaTs2_t = 0.95585841724208476
        sec.gNav11bar_Nav11 = 0.21504623309972959
        sec.gNap_Et2bar_Nap_Et2 = 7.9295968986726376e-07
        sec.gSKv3_1bar_SKv3_1 = 0.18497365407533689
    for sec in pv.ais:
        sec.gNav11bar_Nav11 = nav_ais
        sec.gSKv3_1bar_SKv3_1 = kv3_ais

    for sec in pv.nodes:
        sec.gNav11bar_Nav11 = nav_nodes
        sec.gSKv3_1bar_SKv3_1 = kv3_nodes

    base_nav = {
        "somatic": pv.soma[0].gNav11bar_Nav11,
        "ais": pv.axon[0].gNav11bar_Nav11,
        "nodes": pv.node[0].gNav11bar_Nav11,
    }
    #     print(f"base gnav1.1 (soma, ais, node)={base_nav11_soma:.5f}, {base_nav11_ais:.5f}, {base_nav11_node:.5f}")
    return base_nav


def reset_biophys_orig(pv):
    pv.biophys()

    NaK_ratio_ais = 8.39
    NaK_ratio_nodes = NaK_ratio_ais
    nav_ais_nodes_ratio = NaK_ratio_nodes / NaK_ratio_ais

    nav_ais = 2.99
    kv3_ais = nav_ais / NaK_ratio_ais

    nav_nodes = nav_ais * nav_ais_nodes_ratio
    kv3_nodes = nav_nodes / NaK_ratio_nodes

    print(f"{nav_ais:.2f} | {kv3_ais:.2f} | {nav_nodes:.2f} | {kv3_nodes:.2f}")

    for sec in pv.somatic:
        sec.gNaTs2_tbar_NaTs2_t = 0.95585841724208476
        sec.gNav11bar_Nav11 = 0.21504623309972959
        sec.gNap_Et2bar_Nap_Et2 = 7.9295968986726376e-07
        sec.gSKv3_1bar_SKv3_1 = 0.018497365407533689

    for sec in pv.ais:
        sec.gNav11bar_Nav11 = nav_ais
        sec.gSKv3_1bar_SKv3_1 = kv3_ais

    for sec in pv.nodes:
        sec.gNav11bar_Nav11 = nav_nodes
        sec.gSKv3_1bar_SKv3_1 = kv3_nodes

    base_nav = {
        "somatic": pv.soma[0].gNav11bar_Nav11,
        "ais": pv.axon[0].gNav11bar_Nav11,
        "nodes": pv.node[0].gNav11bar_Nav11,
    }
    #     print(f"base gnav1.1 (soma, ais, node)={base_nav11_soma:.5f}, {base_nav11_ais:.5f}, {base_nav11_node:.5f}")
    return base_nav


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
