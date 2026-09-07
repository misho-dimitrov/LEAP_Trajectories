#!/usr/bin/env Rscript

# Follow-up moderation analysis for PLS-SVD outputs from
# run_pls_trajectories.py (brain trajectory slopes vs behavioural trajectory
# slopes).
#
# Fits (per mode, per PLS model):
#  1) Age moderation:   v ~ u * age + group + sex + site
#  2) Group moderation: v ~ u * group + age + sex + site
#  3) Sex moderation:   v ~ u * sex + age + group + site
#
# Where u = brain variate score (u1..uK) and v = behaviour variate score (v1..vK)
# from a scores.csv written by run_pls_trajectories.py. Age/group/sex/site are
# baseline (T1) covariates pulled independently from df.csv.
#
# Batch mode (default): discovers every {model}/scores.csv under --pls-dir
# (one subfolder per run_pls_trajectories.py --models entry, e.g.
# sdq/prl/autism/ashq/tas) and runs the moderation models on each, writing to
# {pls-dir}/{model}/moderation/.
#
# Single-run mode: pass --scores explicitly (optionally with --outdir) to
# bypass discovery and process one scores.csv directly (e.g. for
# run_pls.py's T1-only outputs).

options(stringsAsFactors = FALSE)

msg <- function(...) cat(..., "\n", sep = "")
die <- function(...) {
  msg("ERROR: ", ...)
  quit(status = 1, save = "no")
}

get_script_dir <- function() {
  # Works when invoked via Rscript path/to/script.R
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd()))
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[1])))
}

parse_kv_args <- function(args) {
  out <- list()
  for (a in args) {
    if (!startsWith(a, "--")) next
    keyval <- sub("^--", "", a)
    if (grepl("=", keyval, fixed = TRUE)) {
      parts <- strsplit(keyval, "=", fixed = TRUE)[[1]]
      key <- parts[1]
      val <- paste(parts[-1], collapse = "=")
      out[[key]] <- val
    } else {
      out[[keyval]] <- TRUE
    }
  }
  out
}

as_int <- function(x, default = NA_integer_) {
  if (is.null(x) || isTRUE(x) || identical(x, "")) return(default)
  suppressWarnings(as.integer(x))
}

as_path <- function(x) {
  if (is.null(x) || isTRUE(x) || identical(x, "")) return(NULL)
  normalizePath(x, mustWork = FALSE)
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# helpers for standardized variables
zscore <- function(x) as.numeric(scale(as.numeric(x)))

# Function to save diagnostic plots for a linear model
save_diagnostic_plots <- function(model, filename, outdir) {
  pdf(file.path(outdir, filename), width = 10, height = 8)
  par(mfrow = c(2, 2))
  plot(model)
  dev.off()
}

fisher_z_test <- function(r1, n1, r2, n2) {
  if (n1 <= 3 || n2 <= 3) {
    msg("  Warning: group too small for Fisher z-test (n1=", n1, ", n2=", n2, ") — skipping")
    return(list(z = NA_real_, p = NA_real_))
  }
  z1 <- atanh(r1)
  z2 <- atanh(r2)
  se <- sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
  z_diff <- (z1 - z2) / se
  p <- 2 * pnorm(-abs(z_diff))
  list(z = z_diff, p = p)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("ggpubr", quietly = TRUE)) {
  install.packages("ggpubr", repos = "https://cloud.r-project.org")
}
library(ggplot2)
library(ggpubr)

script_dir <- get_script_dir()
rs_root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)

# Defaults that match rs_paths.py logic (run_pls_trajectories.py writes
# {pls-dir}/{model}/scores.csv).
pls_dir_default <- file.path(rs_root, "reports", "pls_trajectories")
behav_default <- normalizePath(file.path(rs_root, "..", "Behaviour", "df.csv"), mustWork = FALSE)

args <- parse_kv_args(commandArgs(trailingOnly = TRUE))

behav_path <- as_path(args[["behav"]])
if (is.null(behav_path)) behav_path <- behav_default

subject_col <- if (!is.null(args[["subject-col"]])) args[["subject-col"]] else "subject"
behav_subject_col <- if (!is.null(args[["behav-subject-col"]])) args[["behav-subject-col"]] else "subjects"

age_col <- if (!is.null(args[["age-col"]])) args[["age-col"]] else "t1_ageyrs"
group_col <- if (!is.null(args[["group-col"]])) args[["group-col"]] else "t1_group"
sex_col <- if (!is.null(args[["sex-col"]])) args[["sex-col"]] else "t1_sex"
site_col <- if (!is.null(args[["site-col"]])) args[["site-col"]] else "t1_site"

modes_arg <- if (!is.null(args[["modes"]])) args[["modes"]] else "auto"
if (identical(tolower(as.character(modes_arg)), "auto")) {
  modes_setting <- "auto"  # detect per-model from scores.csv columns
} else {
  modes_setting <- as_int(modes_arg)
  if (is.na(modes_setting) || modes_setting < 1) die("Invalid --modes (expected a positive integer or 'auto')")
}

if (!file.exists(behav_path)) die("Behaviour df.csv not found at: ", behav_path)

msg("Reading behaviour: ", behav_path)
behav_master <- tryCatch(read.csv(behav_path, check.names = FALSE), error = function(e) die(e$message))

if (!(behav_subject_col %in% names(behav_master))) {
  die("Subject column '", behav_subject_col, "' not present in df.csv")
}
if (!(age_col %in% names(behav_master))) {
  die("Age column '", age_col, "' not present in df.csv")
}
if (!(group_col %in% names(behav_master))) {
  die("Group column '", group_col, "' not present in df.csv")
}
if (!(sex_col %in% names(behav_master))) {
  die("Sex column '", sex_col, "' not present in df.csv")
}
if (!(site_col %in% names(behav_master))) {
  die("Site column '", site_col, "' not present in df.csv")
}
behav_master[[behav_subject_col]] <- as.character(behav_master[[behav_subject_col]])

# ============================================================================
# Per-model moderation analysis (one PLS run_pls_trajectories.py --models
# entry per call)
# ============================================================================

run_moderation_for_model <- function(model_name, scores_path, outdir) {
  label <- if (is.na(model_name)) "(single-run)" else model_name
  msg("")
  msg(strrep("=", 64))
  msg("Model: ", label)
  msg("Scores: ", scores_path)
  msg("Output: ", outdir)
  msg(strrep("=", 64))

  if (!file.exists(scores_path)) {
    msg("SKIP: scores.csv not found at: ", scores_path)
    return(invisible(FALSE))
  }

  ensure_dir(outdir)

  msg("Reading scores: ", scores_path)
  scores <- tryCatch(read.csv(scores_path, check.names = FALSE), error = function(e) {
    msg("SKIP: could not read scores.csv (", e$message, ")")
    NULL
  })
  if (is.null(scores)) return(invisible(FALSE))

  if (!(subject_col %in% names(scores))) {
    msg("SKIP: subject column '", subject_col, "' not present in scores.csv")
    return(invisible(FALSE))
  }

  scores[[subject_col]] <- as.character(scores[[subject_col]])
  behav <- behav_master

  # Trajectories PLS truncates subject IDs to 6 chars; align df.csv to match.
  if (all(nchar(scores[[subject_col]]) <= 6)) {
    behav[[behav_subject_col]] <- substr(behav[[behav_subject_col]], 1, 6)
  }

  merged <- merge(
    x = scores,
    y = behav[, c(behav_subject_col, age_col, group_col, sex_col, site_col)],
    by.x = subject_col,
    by.y = behav_subject_col,
    all.x = TRUE,
    all.y = FALSE
  )

  interaction_rows <- list()
  coef_rows <- list()

  # Detect how many modes this model actually has (u1..uK / v1..vK present),
  # unless the caller pinned an explicit --modes count.
  if (identical(modes_setting, "auto")) {
    u_mode_nums <- as.integer(sub("^u", "", grep("^u[0-9]+$", names(merged), value = TRUE)))
    modes <- if (length(u_mode_nums) == 0) 0L else max(u_mode_nums)
    msg("Detected ", modes, " mode(s) in scores.csv (auto).")
  } else {
    modes <- modes_setting
  }

  for (m in seq_len(modes)) {
  u_col <- paste0("u", m)
  v_col <- paste0("v", m)

  if (!(u_col %in% names(merged))) {
    msg("Mode ", m, ": column '", u_col, "' not in scores.csv — skipping (only ",
        sum(grepl("^u[0-9]+$", names(merged))), " mode(s) available)")
    next
  }
  if (!(v_col %in% names(merged))) {
    msg("Mode ", m, ": column '", v_col, "' not in scores.csv — skipping")
    next
  }

  dat <- merged[, c(subject_col, u_col, v_col, age_col, group_col, sex_col, site_col)]
  names(dat) <- c("subject", "u_raw", "v_raw", "age_raw", "group_raw", "sex_raw", "site_raw")

  # drop missing rows for this mode
  dat <- dat[complete.cases(dat), ]
  if (nrow(dat) < 10) {
    msg("Mode ", m, ": too few complete cases (n=", nrow(dat), ") - skipping")
    next
  }

  dat$u_z <- zscore(dat$u_raw)
  dat$v_z <- zscore(dat$v_raw)
  dat$age_z <- zscore(dat$age_raw)
  dat$group_f <- as.factor(dat$group_raw)
  dat$sex_f <- as.factor(dat$sex_raw)
  dat$site_f <- as.factor(dat$site_raw)

  # Model 1: age moderation, accounting for group + sex + site
  mod_age <- lm(v_z ~ u_z * age_z + group_f + sex_f + site_f, data = dat)
  save_diagnostic_plots(mod_age, paste0("mode", m, "_age_moderation_diagnostics.pdf"), outdir)
  d1_age <- drop1(mod_age, test = "F")
  # row name is interaction term
  age_term <- "u_z:age_z"
  if (!(age_term %in% rownames(d1_age))) die("Unexpected: drop1 missing term ", age_term)

  interaction_rows[[length(interaction_rows) + 1]] <- data.frame(
    mode = m,
    model = "age_moderation",
    interaction = age_term,
    n = nrow(dat),
    df = d1_age[age_term, "Df"],
    f_value = unname(d1_age[age_term, "F value"]),
    p_value = unname(d1_age[age_term, "Pr(>F)"]),
    stringsAsFactors = FALSE
  )

  ctab_age <- coef(summary(mod_age))
  coef_rows[[length(coef_rows) + 1]] <- data.frame(
    mode = m,
    model = "age_moderation",
    term = rownames(ctab_age),
    estimate = ctab_age[, "Estimate"],
    std_error = ctab_age[, "Std. Error"],
    statistic = ctab_age[, "t value"],
    p_value = ctab_age[, "Pr(>|t|)"],
    n = nrow(dat),
    stringsAsFactors = FALSE
  )

  # Model 2: group moderation, accounting for age + sex + site
  mod_group <- lm(v_z ~ u_z * group_f + age_z + sex_f + site_f, data = dat)
  save_diagnostic_plots(mod_group, paste0("mode", m, "_group_moderation_diagnostics.pdf"), outdir)
  d1_group <- drop1(mod_group, test = "F")
  group_term <- "u_z:group_f"
  if (!(group_term %in% rownames(d1_group))) die("Unexpected: drop1 missing term ", group_term)

  interaction_rows[[length(interaction_rows) + 1]] <- data.frame(
    mode = m,
    model = "group_moderation",
    interaction = group_term,
    n = nrow(dat),
    df = d1_group[group_term, "Df"],
    f_value = unname(d1_group[group_term, "F value"]),
    p_value = unname(d1_group[group_term, "Pr(>F)"]),
    stringsAsFactors = FALSE
  )

  ctab_group <- coef(summary(mod_group))
  coef_rows[[length(coef_rows) + 1]] <- data.frame(
    mode = m,
    model = "group_moderation",
    term = rownames(ctab_group),
    estimate = ctab_group[, "Estimate"],
    std_error = ctab_group[, "Std. Error"],
    statistic = ctab_group[, "t value"],
    p_value = ctab_group[, "Pr(>|t|)"],
    n = nrow(dat),
    stringsAsFactors = FALSE
  )

  # Model 3: sex moderation, accounting for age + group + site
  mod_sex <- lm(v_z ~ u_z * sex_f + age_z + group_f + site_f, data = dat)
  save_diagnostic_plots(mod_sex, paste0("mode", m, "_sex_moderation_diagnostics.pdf"), outdir)
  d1_sex <- drop1(mod_sex, test = "F")
  sex_term <- "u_z:sex_f"
  if (!(sex_term %in% rownames(d1_sex))) die("Unexpected: drop1 missing term ", sex_term)

  interaction_rows[[length(interaction_rows) + 1]] <- data.frame(
    mode = m,
    model = "sex_moderation",
    interaction = sex_term,
    n = nrow(dat),
    df = d1_sex[sex_term, "Df"],
    f_value = unname(d1_sex[sex_term, "F value"]),
    p_value = unname(d1_sex[sex_term, "Pr(>F)"]),
    stringsAsFactors = FALSE
  )

  ctab_sex <- coef(summary(mod_sex))
  coef_rows[[length(coef_rows) + 1]] <- data.frame(
    mode = m,
    model = "sex_moderation",
    term = rownames(ctab_sex),
    estimate = ctab_sex[, "Estimate"],
    std_error = ctab_sex[, "Std. Error"],
    statistic = ctab_sex[, "t value"],
    p_value = ctab_sex[, "Pr(>|t|)"],
    n = nrow(dat),
    stringsAsFactors = FALSE
  )
}

if (length(interaction_rows) == 0) {
  msg("SKIP: no moderation models were run for '", label, "' (check missingness / columns / modes)")
  return(invisible(FALSE))
}

interaction_df <- do.call(rbind, interaction_rows)
coef_df <- do.call(rbind, coef_rows)

# Multiple testing correction for the interaction terms
# - p_fdr_all: FDR across all interaction tests (modes x moderator)
# - p_fdr_within_model: FDR within each moderator family (4 modes)
interaction_df$p_fdr_all <- p.adjust(interaction_df$p_value, method = "fdr")
interaction_df$p_fdr_within_model <- ave(
  interaction_df$p_value,
  interaction_df$model,
  FUN = function(p) p.adjust(p, method = "fdr")
)

interaction_out <- file.path(outdir, "moderation_interactions.csv")
coef_out <- file.path(outdir, "moderation_coefficients.csv")

write.csv(interaction_df, interaction_out, row.names = FALSE)
write.csv(coef_df, coef_out, row.names = FALSE)

msg("Wrote: ", interaction_out)
msg("Wrote: ", coef_out)
msg("Diagnostic plots saved to: ", outdir)

# ============================================================================
# FOLLOW-UP: Brain–behaviour correlations by group, for every mode
# (significant group × brain score interaction)
# ============================================================================

# Only attempt follow-up for modes that actually exist in the scores table
followup_modes <- seq_len(modes)
followup_modes <- followup_modes[paste0("u", followup_modes) %in% names(merged) &
                                  paste0("v", followup_modes) %in% names(merged)]
cor_rows <- list()
fisher_rows <- list()
plots <- list()

if (length(followup_modes) == 0) {
  msg("No follow-up modes available — skipping follow-up.")
}

for (m in followup_modes) {
  u_col <- paste0("u", m)
  v_col <- paste0("v", m)

  dat <- merged[, c(subject_col, u_col, v_col, age_col, group_col)]
  names(dat) <- c("subject", "u_raw", "v_raw", "age_raw", "group_raw")
  dat <- dat[complete.cases(dat), ]

  dat$u_z <- zscore(dat$u_raw)
  dat$v_z <- zscore(dat$v_raw)
  dat$group_f <- as.factor(dat$group_raw)

  group_levels <- levels(dat$group_f)

  msg("\n========== Mode ", m, ": Brain–Behaviour Correlations by Group ==========")
  for (g in group_levels) {
    idx <- dat$group_f == g
    ct <- cor.test(dat$u_z[idx], dat$v_z[idx])
    msg(sprintf(
      "  %s: r = %.3f, p = %.4f, 95%% CI [%.3f, %.3f], n = %d",
      g, ct$estimate, ct$p.value, ct$conf.int[1], ct$conf.int[2], sum(idx)
    ))
    cor_rows[[length(cor_rows) + 1]] <- data.frame(
      mode = m, group = g, r = ct$estimate, p = ct$p.value,
      ci_lo = ct$conf.int[1], ci_hi = ct$conf.int[2], n = sum(idx),
      stringsAsFactors = FALSE
    )
  }

  # Fisher z-tests: pairwise comparison of correlations between groups
  msg("\n  --- Fisher's z-test (pairwise) ---")
  for (i in seq_len(length(group_levels) - 1)) {
    for (j in (i + 1):length(group_levels)) {
      g1 <- group_levels[i]; g2 <- group_levels[j]
      r1 <- cor(dat$u_z[dat$group_f == g1], dat$v_z[dat$group_f == g1])
      r2 <- cor(dat$u_z[dat$group_f == g2], dat$v_z[dat$group_f == g2])
      n1 <- sum(dat$group_f == g1); n2 <- sum(dat$group_f == g2)
      fz <- fisher_z_test(r1, n1, r2, n2)
      msg(sprintf(
        "  %s (r=%.3f, n=%d) vs %s (r=%.3f, n=%d): z = %.3f, p = %.4f",
        g1, r1, n1, g2, r2, n2, fz$z, fz$p
      ))
      fisher_rows[[length(fisher_rows) + 1]] <- data.frame(
        mode = m, group1 = g1, r1 = r1, n1 = n1,
        group2 = g2, r2 = r2, n2 = n2,
        z = fz$z, p = fz$p,
        stringsAsFactors = FALSE
      )
    }
  }

  # Scatter plot: brain score vs behaviour score, coloured by group
  p <- ggplot(dat, aes(x = u_z, y = v_z, colour = group_f)) +
    geom_point(alpha = 0.5, size = 1.8) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
    stat_cor(aes(label = paste(after_stat(r.label), after_stat(p.label), sep = "~~~")),
             label.x.npc = "left", label.y.npc = "top", size = 3.5) +
    labs(
      title = paste0("PLS Mode ", m, ": Brain–Behaviour Association by Group"),
      x = paste0("Brain Score (u", m, ")"),
      y = paste0("Behaviour Score (v", m, ")"),
      colour = "Group"
    ) +
    theme_bw(base_size = 14) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    scale_colour_brewer(palette = "Set1")

  plots[[as.character(m)]] <- p

  # Save individual mode plot
  pdf_path <- file.path(outdir, paste0("mode", m, "_brain_behav_by_group.pdf"))
  ggsave(pdf_path, p, width = 7, height = 6)
  msg("Saved: ", pdf_path)
}

# Combined figure with one panel per follow-up mode
if (length(plots) >= 2) {
  ncol_combined <- min(3, length(plots))
  nrow_combined <- ceiling(length(plots) / ncol_combined)
  p_combined <- ggarrange(
    plotlist = plots,
    ncol = ncol_combined, nrow = nrow_combined,
    common.legend = TRUE, legend = "bottom",
    labels = LETTERS[seq_along(plots)]
  )
  combined_path <- file.path(outdir, "all_modes_brain_behav_by_group.pdf")
  ggsave(combined_path, p_combined, width = 6.5 * ncol_combined, height = 6 * nrow_combined)
  msg("Saved: ", combined_path)
} else {
  msg("Skipping combined figure (fewer than 2 follow-up mode plots available).")
}

# Save correlation and Fisher z tables
if (length(cor_rows) > 0) {
  cor_df <- do.call(rbind, cor_rows)
  fisher_df <- do.call(rbind, fisher_rows)

  cor_out <- file.path(outdir, "group_correlations_by_mode.csv")
  fisher_out <- file.path(outdir, "fisher_z_tests_by_mode.csv")
  write.csv(cor_df, cor_out, row.names = FALSE)
  write.csv(fisher_df, fisher_out, row.names = FALSE)
  msg("Wrote: ", cor_out)
  msg("Wrote: ", fisher_out)
}

msg("Done. Outputs saved to ", outdir)
invisible(TRUE)
}

# ============================================================================
# Batch driver: run every PLS model's scores.csv through the moderation
# analysis, unless --scores is passed explicitly (single-run mode).
# ============================================================================

pls_dir <- as_path(args[["pls-dir"]])
if (is.null(pls_dir)) pls_dir <- pls_dir_default

models_arg <- if (!is.null(args[["models"]])) args[["models"]] else "all"

scores_override <- as_path(args[["scores"]])
outdir_override <- as_path(args[["outdir"]])

jobs <- list()

if (!is.null(scores_override)) {
  # Single-run mode, e.g. run_pls.py's T1-only outputs:
  #   --scores reports/pls/<model>/scores.csv
  #   --outdir reports/pls/<model>/moderation
  outdir_single <- if (!is.null(outdir_override)) outdir_override else file.path(dirname(scores_override), "moderation")
  jobs[[1]] <- list(model_name = NA_character_, scores = scores_override, outdir = outdir_single)
} else {
  if (!dir.exists(pls_dir)) die("PLS trajectories output directory not found: ", pls_dir, ". Run run_pls_trajectories.py first, or pass --pls-dir / --scores.")

  if (identical(tolower(models_arg), "all")) {
    subdirs <- list.dirs(pls_dir, full.names = FALSE, recursive = FALSE)
    model_names <- subdirs[file.exists(file.path(pls_dir, subdirs, "scores.csv"))]
    if (length(model_names) == 0) {
      die("No model subfolders with scores.csv found under: ", pls_dir)
    }
  } else {
    model_names <- trimws(strsplit(models_arg, ",", fixed = TRUE)[[1]])
    model_names <- model_names[nzchar(model_names)]
  }

  for (mn in model_names) {
    jobs[[length(jobs) + 1]] <- list(
      model_name = mn,
      scores = file.path(pls_dir, mn, "scores.csv"),
      outdir = if (!is.null(outdir_override)) file.path(outdir_override, mn) else file.path(pls_dir, mn, "moderation")
    )
  }
}

job_labels <- vapply(jobs, function(j) if (is.na(j$model_name)) "(single-run)" else j$model_name, character(1))
msg("Models to run: ", paste(job_labels, collapse = ", "))

n_ok <- 0
for (job in jobs) {
  label <- if (is.na(job$model_name)) "(single-run)" else job$model_name
  ok <- tryCatch(
    run_moderation_for_model(job$model_name, job$scores, job$outdir),
    error = function(e) {
      msg("ERROR running model '", label, "': ", conditionMessage(e))
      msg("Skipping this model.")
      FALSE
    }
  )
  if (isTRUE(ok)) n_ok <- n_ok + 1
}

msg("")
msg("Completed ", n_ok, " / ", length(jobs), " model(s).")
