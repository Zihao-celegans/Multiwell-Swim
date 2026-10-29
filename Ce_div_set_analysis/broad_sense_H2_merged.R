library(lme4)
library(tidyverse)

# ---- edit these ----
base <- "C:/Users/jl200/Dropbox/JHU_2026_spring/Multiwell_swim/Data_Backup/My_Passport_6TB_Andersen_Lab_1"
trials <- c(test02 = "10102026_CeDiv_Leva_test02", test03 = "10212026_CeDiv_Leva_test03")
out_dir <- file.path(base, "Merged_CeDiv_Leva_test02_test03")
raw_traits <- c("A0", "auc", "auc_norm", "slope_init", "slope_final")
# Normalized to each strain's own control, so control wells carry no strain signal
normalized_traits <- c("range_delta", "range_ratio", "slope_rec_delta", "slope_rec_ratio",
                       "t_p10_delta", "t_p50_delta", "t_p90_delta",
                       "t_p10_ratio", "t_p50_ratio", "t_p90_ratio",
                       "A0_delta", "auc_delta", "slope_init_delta", "slope_final_delta",
                       "A0_ratio", "auc_ratio", "slope_init_ratio", "slope_final_ratio")
trait_cols <- c(raw_traits, normalized_traits)
nreps <- 100
set.seed(1)
# --------------------

# Broad-sense heritability with trial as a fixed effect: H2 = Vg / (Vg + Ve)
H2 <- function(d) {
  f <- if (n_distinct(d$trial) > 1) value ~ trial + (1 | strain) else value ~ 1 + (1 | strain)
  fit <- suppressWarnings(suppressMessages(lme4::lmer(f, data = d)))
  vc <- as.data.frame(lme4::VarCorr(fit))
  Vg <- vc$vcov[vc$grp == "strain"]
  Ve <- vc$vcov[vc$grp == "Residual"]
  Vg / (Vg + Ve)
}

# Resample wells with replacement within each strain x trial
boot_once <- function(d) {
  d %>% group_by(strain, trial) %>% slice_sample(prop = 1, replace = TRUE) %>% ungroup()
}

H2_with_ci <- function(d, nreps) {
  d <- d %>% group_by(strain) %>% filter(n() >= 2) %>% ungroup()
  n_strains <- n_distinct(d$strain)
  if (n_strains < 2) {
    return(tibble(H2 = NA_real_, lower_5 = NA_real_, upper_95 = NA_real_,
                  n_strains = n_strains, n_wells = nrow(d)))
  }
  boots <- replicate(nreps, H2(boot_once(d)))
  tibble(
    H2        = H2(d),
    lower_5   = quantile(boots, 0.05, na.rm = TRUE, names = FALSE),
    upper_95  = quantile(boots, 0.95, na.rm = TRUE, names = FALSE),
    n_strains = n_strains,
    n_wells   = nrow(d)
  )
}

dat <- imap_dfr(trials, function(folder, trial) {
  read_csv(file.path(base, folder, "traits", "curve_traits_ActValS_per_well.csv"),
           show_col_types = FALSE) %>%
    mutate(trial = trial)
}) %>%
  mutate(strain = as.factor(strain), trial = as.factor(trial)) %>%
  pivot_longer(all_of(trait_cols), names_to = "trait", values_to = "value") %>%
  filter(!is.na(value), !(trait %in% normalized_traits & dose_mM == 0))

shared <- dat %>% distinct(trial, strain) %>% count(strain) %>% filter(n == n_distinct(dat$trial)) %>% pull(strain)
strain_sets <- list(all_strains = levels(droplevels(dat$strain)), shared_strains = as.character(shared))
message("All strains: ", length(strain_sets$all_strains), "; shared: ", length(strain_sets$shared_strains))

results <- imap_dfr(strain_sets, function(strains, set_name) {
  dat %>%
    filter(strain %in% strains) %>%
    group_by(dose, dose_mM, trait) %>%
    group_modify(~ H2_with_ci(.x, nreps)) %>%
    ungroup() %>%
    mutate(strain_set = set_name)
}) %>%
  arrange(strain_set, trait, dose_mM)

print(results, n = Inf)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_csv(results, file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2.csv"))

# Plot: merged (all strains) next to each trial's own H2 from broad_sense_H2.R
trial_results <- imap_dfr(trials, function(folder, trial) {
  read_csv(file.path(base, folder, "traits", "curve_traits_ActValS_broad_sense_H2.csv"),
           show_col_types = FALSE) %>%
    mutate(group = trial)
})
plot_data <- results %>%
  filter(strain_set == "all_strains") %>%
  mutate(group = "merged") %>%
  bind_rows(trial_results) %>%
  mutate(group = factor(group, levels = c("merged", names(trials))))

trait_order <- c("A0", "A0_delta", "A0_ratio",
                 "auc", "auc_delta", "auc_ratio",
                 "slope_init", "slope_init_delta", "slope_init_ratio",
                 "slope_final", "slope_final_delta", "slope_final_ratio",
                 "auc_norm", "range_delta", "range_ratio",
                 "t_p10_delta", "t_p50_delta", "t_p90_delta",
                 "t_p10_ratio", "t_p50_ratio", "t_p90_ratio",
                 "slope_rec_delta", "slope_rec_ratio")
stopifnot(setequal(trait_order, c(raw_traits, normalized_traits)))
p <- plot_data %>%
  mutate(trait = factor(trait, levels = trait_order),
         dose_label = factor(dose_mM, levels = sort(unique(dose_mM)))) %>%
  ggplot(aes(dose_label, H2, colour = group, group = group)) +
  geom_line(position = position_dodge(width = 0.6), alpha = 0.5) +
  geom_errorbar(aes(ymin = lower_5, ymax = upper_95), width = 0.2,
                position = position_dodge(width = 0.6)) +
  geom_point(size = 2, position = position_dodge(width = 0.6)) +
  facet_wrap(~ trait, ncol = 3) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Broad-sense heritability (H2)",
       x = "Dose (mM)", y = "H2", colour = "Data") +
  theme_bw() +
  theme(strip.text = element_text(size = 8), legend.position = "bottom")
png_path <- file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2.png")
unlink(png_path)  # agg fails to overwrite an existing PNG in this synced folder
ggsave(png_path, p, width = 11, height = 14, dpi = 200)
message("Saved to: ", out_dir)
