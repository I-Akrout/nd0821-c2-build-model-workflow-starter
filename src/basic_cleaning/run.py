#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Initiating basic cleaning run")
    run = wandb.init(job_type="basic_cleaning")

    logger.info("Uploading argument values into the run config for tracking purpose")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Download csv input artifact to local machine") 
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Loading data into dataframe")
    df = pd.read_csv(artifact_local_path)

    logging.info("Testing loaded data")

    try:
        assert df.shape[0]*df.shape[1] > 0 
    except AssertionError as err:
        logger.error(f"ERROR: The loaded data seems empty. \
            The data shape is {df.shape}")
        raise err
    
    logging.info("SUCCESS: Data seems valid.")

    logging.info("Starting the cleaning procedure.")
    logging.info(f"-- Dropping data entries with prices \
        outside the [{args.min_price}, {args.max_price}]")

    idx = df['price'].between(args.min_price, args.max_price)
    try:
        assert len(idx) > 0
    except AssertionError as err:
        logger.error("ERROR: No valid data left")
        raise err

    df = df[idx].copy()

    logging.info("SUCCESS: Only data entries with adequate prices \
        are left.")

    logging.info('-- Transforming last_review column from str type \
        to datetime type.')
    
    df['last_review'] = pd.to_datetime(df['last_review'])

    logging.info('Saving clean data into csv')
    file_name = 'clean_sample.csv'
    df.to_csv(file_name, index=False)

    logging.info('Uploading the clean data artifact into wandb servers')
    logging.info("-- Creating artifact")
    clean_data_artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    clean_data_artifact.add_file(file_name)

    logging.info("-- Uploading artifact")
    run.log_artifact(clean_data_artifact)

    logging.info("SUCCESS: Data cleaning process ended successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='wandb artifact to raw data',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="wandb artifact to cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="wandb artifact type, should be cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='Description of the output artifact',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Price lower bound, drop data entries with price smaller \
            than this value",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Price upper bound, drop data entries with price bigger \
            than this value",
        required=True
    )


    args = parser.parse_args()

    go(args)
