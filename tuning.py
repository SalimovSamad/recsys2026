import time

import optuna

from common import SEED, evaluate, recommend

PRIMARY_METRIC = "NDCG@10"
MIN_IMPROVEMENT = 1e-5
PRUNER_STARTUP_TRIALS = 4
PRUNER_WARMUP_EPOCHS = 3

optuna.logging.set_verbosity(optuna.logging.WARNING)


def val_score(model, task):
    return evaluate(recommend(model, task), task)[0][PRIMARY_METRIC]


def suggest(trial, space):
    """Sample one configuration from {name: {"type": "int" | "float" | "cat", "range" | "values", "log"}}"""
    params = {}

    for name, spec in space.items():
        if spec["type"] == "cat":
            params[name] = trial.suggest_categorical(name, spec["values"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, *spec["range"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, *spec["range"], log=spec.get("log", False))

    return params


def search(build, space, task, n_trials, iterative=False, max_epochs=None, eval_every=1, patience=3, warm_start=()):
    """Optuna TPE search; for iterative models the best epoch is returned as the `epochs` parameter."""

    def objective(trial):
        params = suggest(trial, space)
        start = time.time()

        if not iterative:
            score = val_score(build(params).fit(task), task)
        else:
            state = {"score": -1.0, "epoch": 0, "bad": 0}

            def on_epoch(model, epoch):
                if epoch % eval_every and epoch != max_epochs:
                    return False

                score = val_score(model, task)
                trial.report(score, epoch)

                if score > state["score"] + MIN_IMPROVEMENT:
                    state.update(score=score, epoch=epoch, bad=0)
                else:
                    state["bad"] += 1

                if state["bad"] >= patience:
                    return True
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return False

            build({**params, "epochs": max_epochs}).fit(task, on_epoch=on_epoch)
            trial.set_user_attr("best_epoch", state["epoch"])
            score = state["score"]

        trial.set_user_attr("time_s", round(time.time() - start, 1))
        return score

    pruner = optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=PRUNER_WARMUP_EPOCHS)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner)
    for params in warm_start:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_trials)

    best = dict(study.best_trial.params)
    if iterative:
        best["epochs"] = study.best_trial.user_attrs["best_epoch"]
    return study, best
