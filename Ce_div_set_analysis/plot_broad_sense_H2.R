library(tidyverse)

# Usage: Rscript plot_broad_sense_H2.R <H2.csv> [<H2.csv> ...]
args <- commandArgs(trailingOnly = TRUE)
base <- "C:/Users/jl200/Dropbox/JHU_2026_spring/Multiwell_swim/Data_Backup/My_Passport_6TB_Andersen_Lab_1"
if (length(args) == 0) {
  args <- file.path(base, c("08292026_CeDiv_Leva_test01", "10102026_CeDiv_Leva_test02"),
                    "traits/curve_traits_ActValS_broad_sense_H2.csv")
}

trait_order <- c("A0", "auc", "auc_norm", "slope_init", "slope_final",
                 "range_delta", "range_ratio", "slope_rec_delta", "slope_rec_ratio",
                 "t_p10_delta", "t_p50_delta", "t_p90_delta",
                 "t_p10_ratio", "t_p50_ratio", "t_p90_ratio")

plot_H2 <- function(csv) {
  res <- read_csv(csv, show_col_types = FALSE) %>%
    mutate(trait = factor(trait, levels = trait_order),
           dose_label = factor(dose_mM, levels = sort(unique(dose_mM))))
  trial <- basename(dirname(dirname(csv)))
  n_strains <- max(res$n_strains)

  p <- ggplot(res, aes(dose_label, H2, group = 1)) +
    geom_line(colour = "grey60") +
    geom_errorbar(aes(ymin = lower_5, ymax = upper_95), width = 0.2) +
    geom_point(size = 2, colour = "firebrick") +
    facet_wrap(~ trait, ncol = 5) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = paste("Broad-sense heritability (H2):", trial),
         subtitle = paste0(n_strains, " strains"),
         x = "Dose (mM)", y = "H2") +
    theme_bw() +
    theme(strip.text = element_text(size = 8))

  out <- sub("\\.csv$", ".png", csv)
  ggsave(out, p, width = 12, height = 7, dpi = 200)
  message("Saved: ", out)
}

walk(args, plot_H2)
