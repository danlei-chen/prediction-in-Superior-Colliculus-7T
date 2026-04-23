library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(nlme)
library(lmerTest)
library(pbkrtest)
library(emmeans)
library(dplyr)

base_dir <- 'analysis/'
df <- read_csv(paste0(base_dir, 'sc_signal_by_layer_stim.csv'))

df_model <- lmer(signal ~ (sc_layers + side) * task + (sc_layers + side | subject), data = df)
print(isSingular(df_model))
anova(df_model)

fig_df <- aggregate(x = df$signal, by = list(df$task, df$sc_layers, df$subject), FUN = mean)
colnames(fig_df) <- c('task', 'sc_layers', 'subject', 'signal')

ggplot(fig_df, aes(sc_layers, signal, colour = task, fill = task)) +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.25, size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_line(stat = "smooth", method = lm, size = 1, alpha = 0.5) +
  scale_colour_manual(values = c('vision' = "#E69F00", 'somatosensory' = "#0072B2")) +
  scale_fill_manual(values = c('vision' = "#E69F00", 'somatosensory' = "#0072B2")) +
  coord_cartesian(ylim = c(-10, 19)) +
  scale_y_continuous(breaks = round(seq(-10, 19, by = 4), 1)) +
  theme(axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 15, face = "bold"),
        strip.text.x = element_text(size = 16),
        strip.text.y = element_text(size = 16),
        panel.background = element_blank(),
        legend.position = "none",
        strip.background = element_rect(fill = "white"))