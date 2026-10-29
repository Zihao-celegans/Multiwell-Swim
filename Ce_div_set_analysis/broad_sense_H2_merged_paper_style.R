library(lme4)
library(tidyverse)

# Two-step trial adjustment (AndersenLab anthelmintic_dose_responses_manuscript, cleanData + H2 functions),
# without any well filtering and without any control subtraction, compared with the one-step
# model of broad_sense_H2_merged.R. The two methods differ only in how the trial effect is removed:
#   one-step: lmer(value ~ trial + (1|strain)), trial and strain estimated together
#   two-step: 1. lm(value ~ trial) without strain; keep the residuals
#             2. lmer(residual ~ 1 + (1|strain)); bootstrap within strain
# Both run per dose and trait on the same wells (control dose excluded for the control-normalized traits).

# ---- edit these ----
base <- "C:/Users/jl200/Dropbox/JHU_2026_spring/Multiwell_swim/Data_Backup/My_Passport_6TB_Andersen_Lab_1"
trials <- c(test02 = "10102026_CeDiv_Leva_test02", test03 = "10212026_CeDiv_Leva_test03")
out_dir <- file.path(base, "Merged_CeDiv_Leva_test02_test03")
raw_traits <- c("A0", "auc", "auc_norm", "slope_init", "slope_final")
normalized_traits <- c("range_delta", "range_ratio", "slope_rec_delta", "slope_rec_ratio",
                       "t_p10_delta", "t_p50_delta", "t_p90_delta",
                       "t_p10_ratio", "t_p50_ratio", "t_p90_ratio",
                       "A0_delta", "auc_delta", "slope_init_delta", "slope_final_delta",
                       "A0_ratio", "auc_ratio", "slope_init_ratio", "slope_final_ratio")
trait_cols <- c(raw_traits, normalized_traits)
nreps <- 100
set.seed(1)
# --------------------

# Same H2 function as the paper: strain as the only random effect
H2 <- function(d) {
  fit <- suppressWarnings(suppressMessages(lme4::lmer(value ~ 1 + (1 | strain), data = d)))
  vc <- as.data.frame(lme4::VarCorr(fit))
  Vg <- vc$vcov[vc$grp == "strain"]
  Ve <- vc$vcov[vc$grp == "Residual"]
  Vg / (Vg + Ve)
}

# Paper's bootstrap: resample wells within each strain
boot_once <- function(d) d %>% group_by(strain) %>% slice_sample(prop = 1, replace = TRUE) %>% ungroup()

H2_with_ci <- function(d, nreps) {
  d <- d %>% group_by(strain) %>% filter(n() >= 2) %>% ungroup()
  n_strains <- n_distinct(d$strain)
  if (n_strains < 2) {
    return(tibble(H2 = NA_real_, lower_5 = NA_real_, upper_95 = NA_real_,
                  n_strains = n_strains, n_wells = nrow(d)))
  }
  boots <- replicate(nreps, H2(boot_once(d)))
  tibble(H2 = H2(d),
         lower_5 = quantile(boots, 0.05, na.rm = TRUE, names = FALSE),
         upper_95 = quantile(boots, 0.95, na.rm = TRUE, names = FALSE),
         n_strains = n_strains, n_wells = nrow(d))
}

# Step 1 (trial regression without strain), then step 2 (H2 of the residuals), for one dose and trait
H2_two_step <- function(d, nreps) {
  d <- d %>% group_by(strain) %>% filter(n() >= 2) %>% ungroup()
  if (n_distinct(d$strain) >= 2) d$value <- residuals(lm(value ~ trial, data = d))
  H2_with_ci(d, nreps)
}

dat <- imap_dfr(trials, function(folder, trial) {
  read_csv(file.path(base, folder, "traits", "curve_traits_ActValS_per_well.csv"),
           show_col_types = FALSE) %>% mutate(trial = trial)
}) %>%
  mutate(strain = as.factor(strain), trial = as.factor(trial)) %>%
  pivot_longer(all_of(trait_cols), names_to = "trait", values_to = "value") %>%
  filter(!is.na(value), !(trait %in% normalized_traits & dose_mM == 0))

paper <- dat %>%
  group_by(dose, dose_mM, trait) %>%
  group_modify(~ H2_two_step(.x, nreps)) %>%
  ungroup() %>%
  mutate(method = "two-step (paper)") %>%
  arrange(trait, dose_mM)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_csv(paper, file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2_paper_two_step.csv"))

ours <- read_csv(file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2.csv"),
                 show_col_types = FALSE) %>%
  filter(strain_set == "all_strains") %>%
  mutate(method = "one-step (trial + strain)") %>%
  select(-strain_set)

both <- bind_rows(ours, paper)

comparison <- ours %>%
  select(trait, dose, dose_mM, H2_one_step = H2) %>%
  inner_join(paper %>% select(trait, dose, dose_mM, H2_paper = H2), by = c("trait", "dose", "dose_mM")) %>%
  mutate(difference = H2_paper - H2_one_step) %>%
  arrange(trait, dose_mM)
write_csv(comparison, file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2_method_comparison.csv"))
print(comparison, n = Inf)
message("Mean |difference|: ", round(mean(abs(comparison$difference)), 3),
        "; correlation: ", round(cor(comparison$H2_one_step, comparison$H2_paper), 3))

trait_order <- c("A0", "A0_delta", "A0_ratio",
                 "auc", "auc_delta", "auc_ratio",
                 "slope_init", "slope_init_delta", "slope_init_ratio",
                 "slope_final", "slope_final_delta", "slope_final_ratio",
                 "auc_norm", "range_delta", "range_ratio",
                 "t_p10_delta", "t_p50_delta", "t_p90_delta",
                 "t_p10_ratio", "t_p50_ratio", "t_p90_ratio",
                 "slope_rec_delta", "slope_rec_ratio")
stopifnot(setequal(trait_order, trait_cols))

p <- both %>%
  mutate(trait = factor(trait, levels = trait_order),
         dose_label = factor(dose_mM, levels = sort(unique(dose_mM))),
         method = factor(method, levels = c("one-step (trial + strain)", "two-step (paper)"))) %>%
  ggplot(aes(dose_label, H2, colour = method, group = method)) +
  geom_line(position = position_dodge(width = 0.4), alpha = 0.5) +
  geom_errorbar(aes(ymin = lower_5, ymax = upper_95), width = 0.2,
                position = position_dodge(width = 0.4)) +
  geom_point(size = 2, position = position_dodge(width = 0.4)) +
  facet_wrap(~ trait, ncol = 3) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Merged broad-sense heritability (H2): one-step model vs two-step trial adjustment",
       x = "Dose (mM)", y = "H2", colour = "Method") +
  theme_bw() +
  theme(strip.text = element_text(size = 8), legend.position = "bottom")

png_path <- file.path(out_dir, "curve_traits_ActValS_merged_broad_sense_H2_method_comparison.png")
unlink(png_path)  # agg fails to overwrite an existing PNG in this synced folder
ggsave(png_path, p, width = 11, height = 14, dpi = 200)
message("Saved to: ", out_dir)
