library(lme4)
library(tidyverse)

# ---- edit these ----
args <- commandArgs(trailingOnly = TRUE)
traits_csv <- if (length(args) >= 1) args[1] else "C:/Users/jl200/Dropbox/JHU_2026_spring/Multiwell_swim/Data_Backup/My_Passport_6TB_Andersen_Lab_1/10102026_CeDiv_Leva_test02/traits/curve_traits_ActValS_per_well.csv"
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

# Broad-sense heritability: H2 = Vg / (Vg + Ve), strain as a random effect
H2 <- function(d) {
  fit <- suppressWarnings(suppressMessages(
    lme4::lmer(value ~ 1 + (1 | strain), data = d)
  ))
  vc <- as.data.frame(lme4::VarCorr(fit))
  Vg <- vc$vcov[vc$grp == "strain"]
  Ve <- vc$vcov[vc$grp == "Residual"]
  Vg / (Vg + Ve)
}

# Resample wells with replacement within each strain
boot_once <- function(d) {
  d %>% group_by(strain) %>% slice_sample(prop = 1, replace = TRUE) %>% ungroup()
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

dat <- read_csv(traits_csv, show_col_types = FALSE) %>%
  mutate(strain = as.factor(strain)) %>%
  pivot_longer(all_of(trait_cols), names_to = "trait", values_to = "value") %>%
  filter(!is.na(value), !(trait %in% normalized_traits & dose_mM == 0))

results <- dat %>%
  group_by(dose, dose_mM, trait) %>%
  group_modify(~ H2_with_ci(.x, nreps)) %>%
  ungroup() %>%
  arrange(trait, dose_mM)

print(results, n = Inf)
write_csv(results, sub("per_well\\.csv$", "broad_sense_H2.csv", traits_csv))
