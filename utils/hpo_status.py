from optuna.trial import TrialState

def filter_study(study):

    completed = study.get_trials(
        deepcopy=False,
        states=(TrialState.COMPLETE,),
    )
    failed = study.get_trials(
        deepcopy=False,
        states=(TrialState.FAIL,),
    )
    pruned = study.get_trials(
        deepcopy=False,
        states=(TrialState.PRUNED,),
    )

    return completed, failed, pruned
