import functools
import glob
import hashlib
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src import settings
from src.data.files import delete_files, cd

logger = logging.getLogger("nrn_helpers")


initialised = False
__KWARGS__ = {}


def init_nrn(celsius=37, v_init=-65, reinit=False):
    global initialised, __KWARGS__
    if initialised and not reinit:
        return True
    else:
        initialised = True

    # import neuron without loading mechanisms (yet)
    old_cwd = os.getcwd()
    logger.debug(old_cwd)
    try:
        os.chdir("/tmp/")
    except FileNotFoundError:
        pass
    finally:
        from neuron import h

        os.chdir(old_cwd)
        logger.debug(os.getcwd())

    # compile mod files
    if __mod_files_changed(settings.MOD_PATH) or settings.NEURON_RECOMPILE or reinit:
        with cd(settings.MOD_PATH, with_logs=False):
            output = nrnivmodl(clean_after=True)
        if "Error" in str(output):
            raise Exception("MOD FILES not compiled successfully")

    # load mod files
    h.nrn_load_dll(settings.NRNMECH_PATH)

    # load hoc files
    for hoc_file in glob.glob(settings.HOC_PATH + "/*.hoc"):
        h.load_file(hoc_file.replace("\\", "/"))

    # show GUI
    if settings.NEURON_GUI:
        from neuron import gui

        # h.showV()
        h.showRunControl()
        # h.topology()

    # general properties
    h.celsius = celsius
    h.v_init = v_init
    logger.info("celsius={} and v_init={}".format(h.celsius, h.v_init))
    np.random.seed(settings.RANDOM_SEED)

    __KWARGS__ = {}
    env_var(celsius=h.celsius, v_init=h.v_init)


def env_var(**kwargs):
    if kwargs:
        for k, v in kwargs.items():
            __KWARGS__[k] = v
    return __KWARGS__


def nrnivmodl(path="", clean_after=True) -> Tuple[str, Optional[str]]:
    """
    Compile mod files in path

    Note that the compiled file (nrnmech.dll or x86_64/libnrnmech.so) is created in the 
    current working directory.

    :param path: path to mod files or use current path if empty
    :param clean_after: clean the mod files after compilation

    :return: output of nrnivmodl and error message if any
    """
    err = None

    result = os.system(f"nrnivmodl {path}")

    if result == 0:
        compiler_output = "nrnivmodl: compilation successful"
    else:
        compiler_output = "nrnivmodl: compilation failed"
        err = "nrnivmodl: compilation failed"

    if clean_after:
        compiled_files = functools.reduce(
            lambda l1, l2: l1 + l2,
            [glob.glob("*.{}".format(file_type)) for file_type in ["o", "c"]],
        )
        delete_files(compiled_files)

    return compiler_output, err


def __mod_files_changed(path=settings.MOD_PATH):
    md5_files = glob.glob(path + "/hash.md5")
    if len(md5_files) == 0:
        old_md5 = ""
    elif len(md5_files) == 1:
        with open(md5_files[0]) as f:
            old_md5 = f.read()
    else:
        raise BaseException("Too many hash files")

    new_md5_hash = hashlib.md5()
    for mod_file in glob.glob(path + "/*.mod"):
        new_md5_hash.update(__md5(mod_file).encode("utf-8"))
    new_md5 = new_md5_hash.hexdigest()
    if new_md5 == old_md5:
        return False
    else:
        # there are changes
        with open(path + "/hash.md5", "w") as hash_file:
            hash_file.write(new_md5)
        logger.info("there were changes in the mod file directory")
        return True


def __md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    init_nrn(celsius=34, v_init=-80, reinit=True)  # as in BBP optimisation
