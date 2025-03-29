def sweep_name(choices):
    if "experiment" in choices and choices.experiment is not None:
        return choices.experiment
    elif "hparams_search" in choices and choices.hparams_search is not None:
        n = choices.hparams_search
        if "model" in choices:
            n += f"_{choices.model}"
        return n
    return "sweep"
