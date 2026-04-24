library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(nlme)
library(dplyr)

df <- read_csv('mvpa_classification_results.csv')
perm_df <- read_csv('mvpa_classification_permutation_results.csv')

perm_plot_df <- perm_df %>% group_by(perm, test_type, train_type) %>%
  summarise(accuracy = mean(accuracy), .groups = "drop")

ggplot(df, aes(x = test_type, y = accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.6) +
  geom_violin(data = perm_plot_df, aes(x = test_type, y = accuracy), fill = 'gray50', color=NA, alpha = 0.25) +
  geom_jitter(aes(color = test_subjects), width = 0.06, size = 1, alpha = 0.65) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.1, size = 0.8, color = 'black', aes(group = 1)) +
  stat_summary(fun = mean, geom = "point", size = 1.8, color = 'black', aes(group = 1)) +
  theme(legend.position = "none") +
  theme(legend.position = "none",
        axis.line = element_line(colour = "black"),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 11),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_blank(),
        strip.text.x = element_blank(),
        strip.text.y = element_blank(),
        panel.background = element_blank(),
        strip.background = element_rect(fill = "white"))



