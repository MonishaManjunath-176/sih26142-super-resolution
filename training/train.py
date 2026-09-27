import argparse
import yaml

from training.trainer import SRMTrainingPipeline


def main():

    parser = argparse.ArgumentParser(
        description="Train CNN-Attention Hybrid Model for Satellite Super Resolution"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit training batches for a quick benchmark"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = SRMTrainingPipeline(
        config,
        max_batches=args.max_batches
    )

    trainer.run()


if __name__ == "__main__":
    main()