import argparse

from src.experiments import EXPERIMENTS, ExperimentResult, get_experiment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Запуск ML-экспериментов для задержек транспорта.")
    parser.add_argument(
        "--mode",
        choices=sorted(EXPERIMENTS),
        default="train",
        help="Какой эксперимент запустить.",
    )
    return parser.parse_args(argv)


def run_pipeline(mode: str = "train") -> ExperimentResult:
    experiment = get_experiment(mode)
    return experiment.run()


def main(argv=None) -> ExperimentResult:
    args = parse_args(argv)
    return run_pipeline(mode=args.mode)


if __name__ == "__main__":
    main()
