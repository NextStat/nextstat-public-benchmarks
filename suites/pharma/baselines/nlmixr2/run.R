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

write_json <- function(path, obj) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  json <- jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE)
  writeLines(paste0(json, "\n"), con = path)
}

schema_version <- "nextstat.pharma_baseline_result.v1"

meta <- list(
  r_version = as.character(getRversion()),
  platform = paste0(R.version$platform, " (", R.version$os, ")")
)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to write baseline JSON (install.packages('jsonlite'))")
}

if (!requireNamespace("nlmixr2", quietly = TRUE)) {
  write_json(out_path, list(
    schema_version = schema_version,
    baseline = "nlmixr2",
    case = "theoph_fit",
    status = "skipped",
    reason = "nlmixr2 not installed",
    meta = meta
  ))
  quit(status = 0)
}

# If nlmixr2 is installed, this is where we would run a pinned fit protocol.
# We intentionally keep this as a placeholder until we can pin the full R env (renv.lock/Docker).
write_json(out_path, list(
  schema_version = schema_version,
  baseline = "nlmixr2",
  case = "theoph_fit",
  status = "skipped",
  reason = "runner template: pinning + fit protocol not implemented yet",
  meta = meta,
  packages = list(
    nlmixr2 = as.character(utils::packageVersion("nlmixr2"))
  )
))

