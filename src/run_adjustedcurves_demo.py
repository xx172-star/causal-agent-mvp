# src/run_adjustedcurves_demo.py
import argparse
from rpy2 import robjects as ro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--group", default="group", help="Treatment/group column name")
    parser.add_argument("--time", default="time", help="Time-to-event column name")
    parser.add_argument("--event", default="event", help="Event indicator column name (0/1)")
    parser.add_argument(
        "--covariates",
        default="",
        help="Comma-separated covariates for propensity model, e.g. X1,X2",
    )
    args = parser.parse_args()

    csv_path = args.csv.replace("\\", "/")
    group = args.group.strip()
    time_col = args.time.strip()
    event_col = args.event.strip()
    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]

    # Build treatment model formula: group ~ 1  OR group ~ X1 + X2 + ...
    rhs = "1" if len(covariates) == 0 else " + ".join(covariates)
    tm_formula = f"{group} ~ {rhs}"

    R_CODE = f"""
suppressPackageStartupMessages({{
  library(adjustedCurves)
}})

dat <- read.csv("{csv_path}")

# basic type safety
dat${group} <- as.factor(dat${group})
dat${time_col} <- as.numeric(dat${time_col})
dat${event_col} <- as.numeric(dat${event_col})

adj <- adjustedsurv(
  data = dat,
  variable = "{group}",
  ev_time = "{time_col}",
  event = "{event_col}",
  method = "iptw_km",
  treatment_model = as.formula("{tm_formula}"),
  weight_method = "ps"
)

capture.output(print(adj))
"""

    out = ro.r(R_CODE)
    print("\n".join(list(out)))


if __name__ == "__main__":
    main()
