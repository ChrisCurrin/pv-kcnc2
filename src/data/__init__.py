from .dataframes import *
from .files import *

if __name__ == "__main__":
    import os
    from pathlib import Path

    import numpy as np

    from src.data.dataframes import get_cached_df

    try:
        from src.cells.pv_nrn import get_pv
    except ImportError:
        print("must be run from `pv-scn1a` directory")
    pv = get_pv()
    amp = 0.1
    dur = 10
    test_name = "test"
    test_path = Path(".cache") / f"{test_name}.h5"

    if test_path.exists():
        os.remove(test_path)
        assert not test_path.exists()

    AP, _df = get_cached_df("test", pv, amp, dur, shape_plot=True)
    assert test_path.exists()

    AP, loaded_df = get_cached_df(test_path)
    assert np.all(loaded_df == _df), "values weren't stored/loaded properly!"
