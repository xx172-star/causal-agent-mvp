"""
src/run_causalmodels_demo.py

Robust CausalModels demo via rpy2.

Key improvements:
- Print warnings immediately in R (options(warn=1)).
- Capture warnings via withCallingHandlers and ALSO print them (message), then list unique warnings at end.
- Robustly normalize treatment to 0/1 even if it is "True"/"False" strings in CSV.
- Safer default behavior: optionally limit covariate count to reduce rank-deficient fits (--max_covariates, default 15).
- Optional: attempt to override complex default formulas by assigning simpler formulas (may or may not be honored by package).
- Provide diagnostics when ATE is NA (treatment balance, NAs, propensity extremes).
- Optional machine-readable JSON summary via --out_json.

Usage:
  python src/run_causalmodels_demo.py --csv data/ihdp_data.csv --treatment treatment --outcome y_factual
  python src/run_causalmodels_demo.py --csv data/ihdp_data.csv --treatment treatment --outcome y_factual --covariates x1,x2,...,x25
  python src/run_causalmodels_demo.py --csv data/ihdp_data.csv --treatment treatment --outcome y_factual --complex_formulas 1
  python src/run_causalmodels_demo.py --csv data/ihdp_data.csv --treatment treatment --outcome y_factual --out_json out/demo/ihdp.summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from rpy2 import robjects as ro


def _rx2(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get an element from an rpy2 ListVector using rx2()."""
    try:
        return obj.rx2(key)
    except Exception:
        return default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--outcome", required=True)
    p.add_argument("--covariates", default="")
    p.add_argument(
        "--complex_formulas",
        type=int,
        default=0,
        help="0 safer default; 1 use package default (may warn/NA).",
    )
    p.add_argument(
        "--out_json",
        default="",
        help="If provided, write a machine-readable JSON summary to this path.",
    )
    p.add_argument(
        "--max_covariates",
        type=int,
        default=15,
        help="Limit number of covariates used (0 = no limit). Default 15 for stability.",
    )
    args = p.parse_args()

    csv_path = str(Path(args.csv)).replace("\\", "/")
    yvar = args.outcome.strip()
    avar = args.treatment.strip()
    covs = [c.strip() for c in args.covariates.split(",") if c.strip()]
    cov_r = "c(" + ",".join(f'"{c}"' for c in covs) + ")" if covs else "character(0)"
    complex_flag = int(args.complex_formulas)
    out_json = args.out_json.strip()
    max_covs = int(args.max_covariates)

    R_CODE = f"""
options(warn=1)  # print warnings as they occur

suppressPackageStartupMessages({{
  library(CausalModels)
}})

.warnings <- character(0)

# Capture warnings and also print them deterministically
.capture_warning <- function(w) {{
  msg <- conditionMessage(w)
  .warnings <<- c(.warnings, msg)
  message("WARNING: ", msg)
  invokeRestart("muffleWarning")
}}

dat <- tryCatch({{
  read.csv("{csv_path}", stringsAsFactors = FALSE, check.names = FALSE)
}}, error = function(e) {{
  stop(paste0("Failed to read CSV: ", conditionMessage(e)))
}})

if (!("{yvar}" %in% names(dat))) stop(paste0("Outcome column not found: {yvar}"))
if (!("{avar}" %in% names(dat))) stop(paste0("Treatment column not found: {avar}"))

# outcome numeric
suppressWarnings({{
  dat[["{yvar}"]] <- as.numeric(dat[["{yvar}"]])
}})
if (all(is.na(dat[["{yvar}"]]))) stop("Outcome became all NA after numeric coercion.")

# ---------------- Robust treatment normalization to 0/1 ----------------
a_raw <- dat[["{avar}"]]

# If character, normalize common encodings: "True"/"False", "1"/"0"
if (is.character(a_raw)) {{
  a_low <- tolower(trimws(a_raw))
  if (all(a_low %in% c("true","false"))) {{
    dat[["{avar}"]] <- ifelse(a_low == "true", 1, 0)
  }} else if (all(a_low %in% c("1","0"))) {{
    dat[["{avar}"]] <- as.integer(a_low)
  }} else {{
    stop(paste0("Unsupported character treatment encoding. Example values: ",
                paste(head(unique(a_raw), 5), collapse=", ")))
  }}
}}

# If logical, convert
if (is.logical(dat[["{avar}"]])) {{
  dat[["{avar}"]] <- ifelse(dat[["{avar}"]], 1, 0)
}}

# If factor, try to coerce via as.character then handle like above
if (is.factor(dat[["{avar}"]])) {{
  tmp <- as.character(dat[["{avar}"]])
  tmp_low <- tolower(trimws(tmp))
  if (all(tmp_low %in% c("true","false"))) {{
    dat[["{avar}"]] <- ifelse(tmp_low == "true", 1, 0)
  }} else if (all(tmp_low %in% c("1","0"))) {{
    dat[["{avar}"]] <- as.integer(tmp_low)
  }} else {{
    suppressWarnings({{
      dat[["{avar}"]] <- as.integer(tmp)
    }})
  }}
}}

# Finally enforce integer 0/1 and fixed factor level order
suppressWarnings({{
  dat[["{avar}"]] <- as.integer(dat[["{avar}"]])
}})
dat[["{avar}"]] <- factor(dat[["{avar}"]], levels=c(0,1))
dat[["{avar}"]] <- droplevels(dat[["{avar}"]])

if (any(is.na(dat[["{avar}"]]))) {{
  stop("Treatment contains NA after normalization (expected 0/1, TRUE/FALSE, or True/False).")
}}
if (nlevels(dat[["{avar}"]]) != 2) {{
  stop(paste0("Treatment must be binary after normalization. Levels: ",
              paste(levels(dat[["{avar}"]]), collapse=", ")))
}}
# ----------------------------------------------------------------------

# choose covariates
confounders_user <- {cov_r}
use_user <- length(confounders_user) > 0

if (!use_user) {{
  # default: prefer x1.. pattern if exists, else numeric columns excluding roles
  allc <- setdiff(names(dat), c("{yvar}", "{avar}"))
  x_like <- allc[tolower(substr(allc,1,1))=="x" & grepl("^[xX][0-9]+$", allc)]
  if (length(x_like) > 0) {{
    suf <- suppressWarnings(as.integer(sub("^[xX]", "", x_like)))
    x_like <- x_like[order(suf)]
    confounders <- x_like
  }} else {{
    confounders <- allc
  }}
}} else {{
  confounders <- confounders_user
}}

confounders <- confounders[confounders %in% names(dat)]

# keep numeric-ish only + drop constants
keep <- c()
for (cc in confounders) {{
  v <- dat[[cc]]
  if (is.logical(v)) {{
    dat[[cc]] <- ifelse(v, 1, 0); v <- dat[[cc]]
  }}
  if (is.character(v)) {{
    suppressWarnings({{ tmp <- as.numeric(v) }})
    if (!all(is.na(tmp))) {{ dat[[cc]] <- tmp; v <- dat[[cc]] }}
  }}
  if (is.factor(v)) {{
    suppressWarnings({{ tmp <- as.numeric(as.character(v)) }})
    if (!all(is.na(tmp))) {{ dat[[cc]] <- tmp; v <- dat[[cc]] }}
  }}
  if (is.numeric(v) || is.integer(v)) {{
    if (all(is.na(v))) next
    if (length(unique(v[!is.na(v)])) < 2) next
    keep <- c(keep, cc)
  }}
}}
confounders <- unique(keep)
if (length(confounders) == 0) stop("No usable numeric confounders found.")

# Limit covariate count for stability (MVP default)
if ({max_covs} > 0 && length(confounders) > {max_covs}) {{
  confounders <- head(confounders, {max_covs})
}}

# init_params (capture warnings)
withCallingHandlers({{
  init_params("{yvar}", "{avar}", covariates = confounders, data = dat)
}}, warning = .capture_warning)

# Optional: safer formulas to avoid separation/overfit (package may or may not honor these globals)
if ({complex_flag} == 0) {{
  try({{
    outcome_formula <- as.formula(paste("{yvar}", "~", paste(c("{avar}", confounders), collapse=" + ")))
    propensity_formula <- as.formula(paste("{avar}", "~", paste(confounders, collapse=" + ")))
    assign("outcome.formula", outcome_formula, envir = .GlobalEnv)
    assign("propensity.formula", propensity_formula, envir = .GlobalEnv)
  }}, silent=TRUE)
}}

# run DR (capture warnings)
model <- withCallingHandlers({{
  doubly_robust(data = dat)
}}, warning = .capture_warning)

# Build output lines (human-readable)
out_lines <- c()
out_lines <- c(out_lines, "=== CausalModels doubly_robust demo ===")
out_lines <- c(out_lines, paste0("Outcome: {yvar}"))
out_lines <- c(out_lines, paste0("Treatment: {avar} (levels: ", paste(levels(dat[["{avar}"]]), collapse=", "), ")"))
out_lines <- c(out_lines, paste0("N: ", nrow(dat)))
out_lines <- c(out_lines, paste0("Using ", length(confounders), " numeric confounders."))
out_lines <- c(out_lines, paste0("Confounders (first 15): ", paste(head(confounders, 15), collapse=", ")))
if (length(confounders) > 15) out_lines <- c(out_lines, paste0("... (+", length(confounders)-15, " more)"))
out_lines <- c(out_lines, "")

ate_tbl <- NULL
if (!is.null(model$ATE.summary)) {{
  out_lines <- c(out_lines, capture.output(print(model$ATE.summary)))
  ate_tbl <- model$ATE.summary
}} else {{
  out_lines <- c(out_lines, "ATE.summary not found; printing model:")
  out_lines <- c(out_lines, capture.output(print(model)))
}}

# If ATE is NA, print diagnostics
ate_is_na <- FALSE
try({{
  if (!is.null(ate_tbl) && all(is.na(ate_tbl$ATE))) ate_is_na <- TRUE
}}, silent=TRUE)

if (ate_is_na) {{
  out_lines <- c(out_lines, "")
  out_lines <- c(out_lines, "=== Diagnostics (ATE is NA) ===")
  out_lines <- c(out_lines, paste0("Outcome NA count: ", sum(is.na(dat[["{yvar}"]]))))
  out_lines <- c(out_lines, paste0("Treatment counts: ", paste(capture.output(print(table(dat[["{avar}"]]))), collapse=" ")))

  # Propensity extremes check via quick glm
  ps <- tryCatch({{
    f <- as.formula(paste("{avar}", "~", paste(head(confounders, min(10, length(confounders))), collapse=" + ")))
    mps <- glm(f, data=dat, family=binomial())
    p <- suppressWarnings(predict(mps, type="response"))
    p
  }}, error=function(e) NULL)

  if (!is.null(ps)) {{
    out_lines <- c(out_lines, paste0("Approx propensity min/max (glm with up to 10 covs): ",
                                    sprintf("%.4f", min(ps, na.rm=TRUE)), " / ",
                                    sprintf("%.4f", max(ps, na.rm=TRUE))))
  }}

  # near-zero variance (note: binary vars will have unique=2; this just reports small unique count)
  nzv <- c()
  for (cc in confounders) {{
    v <- dat[[cc]]
    if (is.numeric(v) || is.integer(v)) {{
      u <- unique(v[!is.na(v)])
      if (length(u) < 3) nzv <- c(nzv, cc)
    }}
  }}
  if (length(nzv) > 0) {{
    out_lines <- c(out_lines, paste0("Near-zero-variance covariates (unique<3): ", paste(head(nzv, 20), collapse=", ")))
    if (length(nzv) > 20) out_lines <- c(out_lines, paste0("... (+", length(nzv)-20, " more)"))
  }}
}}

# Print captured warnings at end
if (length(.warnings) > 0) {{
  out_lines <- c(out_lines, "")
  out_lines <- c(out_lines, "=== Captured warnings (first 50) ===")
  out_lines <- c(out_lines, head(unique(.warnings), 50))
}}

# Prepare a structured result for Python
res <- list(
  status = ifelse(ate_is_na, "ok_with_na", "ok"),
  outcome = "{yvar}",
  treatment = "{avar}",
  treatment_levels = levels(dat[["{avar}"]]),
  n = nrow(dat),
  p_confounders = length(confounders),
  confounders = confounders,
  ate = NA_real_,
  se = NA_real_,
  ci95 = c(NA_real_, NA_real_),
  warnings = unique(.warnings),
  log_lines = out_lines
)

try({{
  if (!is.null(ate_tbl) && "ATE" %in% names(ate_tbl)) {{
    res$ate <- as.numeric(ate_tbl$ATE[1])
  }}
  if (!is.null(ate_tbl) && "SE" %in% names(ate_tbl)) {{
    res$se <- as.numeric(ate_tbl$SE[1])
  }}
  if (!is.null(ate_tbl) && "2.5 %" %in% names(ate_tbl) && "97.5 %" %in% names(ate_tbl)) {{
    res$ci95 <- c(as.numeric(ate_tbl$`2.5 %`[1]), as.numeric(ate_tbl$`97.5 %`[1]))
  }}
}}, silent=TRUE)

res
"""

    try:
        out = ro.r(R_CODE)
    except Exception as e:
        raise SystemExit(f"[R ERROR] {e}") from e

    # Print human-readable log
    log_lines = list(_rx2(out, "log_lines", []))
    print("\n".join([str(x) for x in log_lines]))

    # Helpers to coerce rpy2 vectors
    def _as_list(x: Any) -> list:
        try:
            return list(x)
        except Exception:
            return []

    def _first_scalar(x: Any, default: Any) -> Any:
        lst = _as_list(x)
        return lst[0] if len(lst) > 0 else default

    n_val = int(_first_scalar(_rx2(out, "n", [0]), 0))
    p_val = int(_first_scalar(_rx2(out, "p_confounders", [0]), 0))

    summary: Dict[str, Any] = {
        "tool": "CausalModels::doubly_robust",
        "input_csv": csv_path,
        # IMPORTANT: pull scalar from R vectors to avoid "[1] \"ok\"" artifacts
        "outcome": str(_first_scalar(_rx2(out, "outcome", [""]), "")),
        "treatment": str(_first_scalar(_rx2(out, "treatment", [""]), "")),
        "treatment_levels": [str(x) for x in _as_list(_rx2(out, "treatment_levels", []))],
        "n": n_val,
        "p_confounders": p_val,
        "ate": float(_first_scalar(_rx2(out, "ate", [float("nan")]), float("nan"))),
        "se": float(_first_scalar(_rx2(out, "se", [float("nan")]), float("nan"))),
        "ci95": [float(x) for x in _as_list(_rx2(out, "ci95", [float("nan"), float("nan")]))],
        "warnings": [str(x) for x in _as_list(_rx2(out, "warnings", []))],
        "status": str(_first_scalar(_rx2(out, "status", ["ok"]), "ok")),
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if out_json:
        print(f"\n[WROTE_JSON] {out_json}")


if __name__ == "__main__":
    main()
