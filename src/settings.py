import os
import platform
import seaborn as sns

STIM_ONSET = 20
STIM_PULSE_DUR = 1.5
# ----------------------------------------------------------------------------------------------------------------------
# COLOR
# ----------------------------------------------------------------------------------------------------------------------
GROUP_COLOR_A = "#F6911E"
GROUP_COLOR_B = "#00ACED"
GROUP_COLOR_C = "#2B69AD"
GROUP_COLOR_D = "#919395"
GROUP_COLOR_E = "#232020"
# _spec_pal = sns.color_palette("Spectral", n_colors=20)
# SECTION_PALETTE = sns.color_palette([_spec_pal[0], _spec_pal[1], _spec_pal[-1]])
# SECTION_PALETTE = sns.color_palette("Set2", n_colors=3)[:2][::-1]
SECTION_PALETTE = "Set2"

# ----------------------------------------------------------------------------------------------------------------------
# NEURON
# ----------------------------------------------------------------------------------------------------------------------
NEURON_GUI = False
NEURON_RECOMPILE = False
NEURON_QUICK_DEFAULT = True  # use cvode where implicit
HOC_PATH = "cells/"
MOD_PATH = "cells/mechanisms/"
NRNMECH_PATH = ""

if not os.path.isdir(MOD_PATH):
    # find the dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MOD_PATH = os.path.join(dir_path, MOD_PATH)
    HOC_PATH = os.path.join(dir_path, HOC_PATH)
if platform.system() == "Linux" or platform.system() == "Darwin":
    NRNMECH_PATH = MOD_PATH + "x86_64/.libs/libnrnmech.so"
elif platform.system() == "Windows":
    NRNMECH_PATH = MOD_PATH + "nrnmech.dll"
else:
    print("unknown system")
    exit(-1)
NRNMECH_PATH = NRNMECH_PATH.replace("\\", "/")

# ----------------------------------------------------------------------------------------------------------------------
# RANDOM
# ----------------------------------------------------------------------------------------------------------------------
RANDOM_SEED = 0
