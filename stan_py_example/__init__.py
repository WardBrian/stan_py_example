import shutil
import warnings
from pathlib import Path

import cmdstanpy

STAN_FILES_FOLDER = Path(__file__).parent / "stan"
CMDSTAN_VERSION = "2.30.1"


# on Windows specifically, we should point cmdstanpy to the repackaged
# CmdStan if it exists. This lets cmdstanpy handle the TBB path for us.
local_cmdstan = STAN_FILES_FOLDER / f"cmdstan-{CMDSTAN_VERSION}"
if local_cmdstan.exists():
    cmdstanpy.set_cmdstan_path(str(local_cmdstan.resolve()))

# Try to load the pre-compiled models. If that fails, compile them
try:

    BERNOULLI = cmdstanpy.CmdStanModel(
        exe_file=STAN_FILES_FOLDER / "bernoulli.exe",
        stan_file=STAN_FILES_FOLDER / "bernoulli.stan",
        compile=False,
    )

except ValueError:
    warnings.warn("Failed to load pre-built models, compiling")

    BERNOULLI = cmdstanpy.CmdStanModel(
        stan_file=STAN_FILES_FOLDER / "bernoulli.stan",
        stanc_options={"O1": True},
    )
    shutil.copy(
        BERNOULLI.exe_file,  # type: ignore
        STAN_FILES_FOLDER / "bernoulli.exe",
    )


def run_my_model():
    data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    # obtain a posterior sample from the model conditioned on the data
    fit = BERNOULLI.sample(chains=4, data=data)

    # summarize the results (wraps CmdStan `bin/stansummary`):
    return fit.summary()
