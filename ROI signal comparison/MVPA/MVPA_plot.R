library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(nlme)
library(dplyr)

######### classification plot########
ROI='SC'

df <- read_csv(paste0('analysis/MVPA/trainConsequence_testSelection/',ROI,'_MVPA_classification_80folds.csv'))
perm_df <- read_csv(paste0('analysis/MVPA/trainConsequence_testSelection/',ROI,'_MVPA_classification_80folds_permutation_subjShuffle.csv'))

perm_plot_df <- perm_df %>%group_by(perm, test_type) %>%summarise(accuracy = mean(accuracy), .groups = "drop")
ggplot(df_tmp, aes(x = test_type, y = accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.6) +
  # Permutation distribution
  geom_violin(data = perm_plot_df, aes(x = test_type, y = accuracy), fill = 'gray50', color=NA, alpha = 0.25) +
  # Jittered dots for individual subjects, optionally colored
  geom_jitter(aes(color = test_subjects), width = 0.06, size = 1, alpha = 0.65) +
  # Mean ± SE error bars across all subjects
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.1, size = 0.8, color = 'black', aes(group = 1)) +
  # Overall mean points
  stat_summary(fun = mean, geom = "point", size = 1.8, color = 'black', aes(group = 1)) +
  # # Chance line at 0.5
  # Y-axis limits and ticks
  coord_cartesian(ylim = c(0, 1.05)) +
  scale_y_continuous(breaks = seq(0, 1.05, by = 0.2)) +
  theme(legend.position = "none") +
  theme(
    axis.line = element_line(colour = "black"),
    axis.text.x = element_blank(),        # Hide x-axis tick labels
    axis.text.y = element_text(size = 11),
    axis.title.x = element_blank(),       # Hide x-axis title
    axis.title.y = element_blank(),       # Hide y-axis title
    plot.title = element_blank(),         # Hide plot title
    strip.text.x = element_blank(),       # Hide facet strip x labels
    strip.text.y = element_blank(),       # Hide facet strip y labels
    panel.background = element_blank(),
    strip.background = element_rect(fill = "white"))



