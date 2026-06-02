from src.experiments.ablation_experiment import AblationExperiment
from src.experiments.base import BaseExperiment
from src.experiments.forward_selection_experiment import ForwardSelectionExperiment
from src.experiments.train_experiment import TrainExperiment


EXPERIMENTS: dict[str, type[BaseExperiment]] = {
    TrainExperiment.name: TrainExperiment,
    AblationExperiment.name: AblationExperiment,
    ForwardSelectionExperiment.name: ForwardSelectionExperiment,
}


def get_experiment(mode: str) -> BaseExperiment:
    experiment_class = EXPERIMENTS.get(mode)
    if experiment_class is None:
        available_modes = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment mode '{mode}'. Available modes: {available_modes}")
    return experiment_class()
