# Activate the environment for noob (No adaptatiOn withOut oBservation).
#   usage:  source env.sh
#
# The Compute Canada gentoo stack injects a `_manylinux.py` (via PYTHONPATH) that
# makes pip reject every PyPI wheel, and a pip.conf that forces the CC wheelhouse.
# Both must be cleared for this conda env, whose packages all come from PyPI.
source /project/rrg-yangw/linlian/miniconda3/etc/profile.d/conda.sh
conda activate obs
unset PYTHONPATH PIP_CONFIG_FILE

export NOOB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${NOOB_ROOT}"

# Keep dataloader workers from oversubscribing the node.
export OMP_NUM_THREADS=4
