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

def load_stan_model(name: str) -> cmdstanpy.CmdStanModel:
    """
    Try to load precompiled Stan models. If that fails,
    compile them.
    """
    try:
        model = cmdstanpy.CmdStanModel(
            exe_file=STAN_FILES_FOLDER / f"{name}.exe",
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
            compile=False,
        )
    except ValueError:
        warnings.warn(f"Failed to load pre-built model '{name}.exe', compiling")
        model = cmdstanpy.CmdStanModel(
            stan_file=STAN_FILES_FOLDER / f"{name}.stan",
            stanc_options={"O1": True},
        )
        shutil.copy(
            model.exe_file,  # type: ignore
            STAN_FILES_FOLDER / f"{name}.exe",
        )

    return model


BERNOULLI = load_stan_model("bernoulli")


def run_my_model():
    data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    # obtain a posterior sample from the model conditioned on the data
    fit = BERNOULLI.sample(chains=4, data=data)

    # summarize the results (wraps CmdStan `bin/stansummary`):
    return fit.summary()
