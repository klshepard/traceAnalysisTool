import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--time",
    type=int,
    default=3600,
    help="minutes to allocate python wait to open slurm threads.",
)
args = parser.parse_args()

# By default, you'll get 5 days of wait time.

time.sleep(60.0 * args.time)
