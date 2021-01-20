import argparse
import json
import os

from trainer import model

import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--train_data_path",
        help="GCS location of training data",
        required=True
    )
    parser.add_argument(
        "--eval_data_path",
        help="GCS location of evaluation data",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        help="Number of examples to compute gradient over.",
        type=int,
        default=512
    )

    # TODO: Add nnsize argument
    parser.add_argument(
        "--nnsize",
        help="Size of the neural network.",
        nargs="+",
        type=int,
        default=[64, 32]
    )

    # TODO: Add nembeds argument
    parser.add_argument(
        "--nembeds",
        help="Size of the embedding layer.",
        type=int,
        default=3
    )

    # TODO: Add num_epochs argument
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs for training.",
        type=int,
        default=5
    )

    # TODO: Add train_examples argument
    parser.add_argument(
        "--train_examples",
        help="Number of examples to train over.",
        type=int,
        default=5000
    )

    # TODO: Add eval_steps argument
    parser.add_argument(
        "--eval_steps",
        help="Number of examples to train over.",
        type=int,
        default=None
    )

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Modify some arguments
    arguments["train_examples"] *= 1000

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    arguments["output_dir"] = os.path.join(
        arguments["output_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    model.train_and_evaluate(arguments)
