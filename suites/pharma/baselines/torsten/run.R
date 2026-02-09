#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default_value = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default_value)
  if (idx == length(args)) return(default_value)
  return(args[idx + 1])
}

out_path <- get_arg("--out", "")
if (is.null(out_path) || out_path == "") {
  stop("missing required --out")
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to write baseline JSON (install.packages('jsonlite'))")
}

write_json <- function(path, obj) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  json <- jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE)
  writeLines(paste0(json, "\n"), con = path)
}

meta <- list(
  r_version = as.character(getRversion()),
  platform = paste0(R.version$platform, " (", R.version$os, ")")
)

if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "torsten",
    case = "pk_fit",
    status = "skipped",
    reason = "cmdstanr not installed (Torsten baseline requires a pinned CmdStan toolchain)",
    meta = meta
  ))
  quit(status = 0)
}

# Torsten presence requires a Torsten-enabled CmdStan build; we treat this as not configured
# until we have a pinned setup in the benchmarks repo.
write_json(out_path, list(
  schema_version = "nextstat.pharma_baseline_result.v1",
  baseline = "torsten",
  case = "pk_fit",
  status = "skipped",
  reason = "runner template: Torsten-enabled CmdStan not configured/pinned yet",
  meta = meta,
  packages = list(
    cmdstanr = as.character(utils::packageVersion("cmdstanr"))
  )
))

