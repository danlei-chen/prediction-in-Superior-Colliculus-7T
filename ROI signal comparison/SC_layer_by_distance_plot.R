library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(nlme)
library(lmerTest)
library(pbkrtest)
library(emmeans)

######################################################################
######################################################################
# df <- read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_emo_RT_voxel_distance.csv')
# df <- rbind(df, read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_pain_RT_voxel_distance.csv'))
df <- read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_emo_voxel_distance.csv')
df <- rbind(df, read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_pain_voxel_distance.csv'))
# df$type <- factor(df$df$type, levels=c('3-back','1-back'))
df$subject <- as.factor(df$subject)
df$task <- as.factor(df$task)
df$distance <- gsub('\\[', '', df$distance)
df$distance <- gsub(']', '', df$distance)
df$distance <- as.numeric(df$distance)
df$quartile <- gsub('\\[', '', df$quartile)
df$quartile <- gsub(']', '', df$quartile)
df$quartile <- as.numeric(df$quartile)
# df$quartile = as.factor(df$quartile)
df <- df[df$type=='active_motor'|df$type=='passive',]

ggplot(df, aes(distance, signal, colour = type)) + 
  facet_grid(task~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  # stat_summary(fun = mean, geom = "line", size=.25, alpha=0.5) +
  geom_smooth(method=lm)+
  # geom_vline(xintercept=quantile(df$distance)['25%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['50%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['75%'],linetype = "dashed", alpha=0.5)+ 
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) +
  # scale_colour_manual(values = c('3-back' = "chartreuse3", '1-back' = "chartreuse4")) +
  theme(text=element_text(family="Times New Roman"))

# ggplot(df, aes(distance, signal, colour = distance)) + 
#   facet_grid(task~side)+
#   stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
#   # stat_summary(fun = mean, geom = "line", size=.25, alpha=0.5) +
#   geom_smooth(method=lm)+
#   # geom_vline(xintercept=quantile(df$distance)['25%'],linetype = "dashed", alpha=0.5)+ 
#   # geom_vline(xintercept=quantile(df$distance)['50%'],linetype = "dashed", alpha=0.5)+ 
#   # geom_vline(xintercept=quantile(df$distance)['75%'],linetype = "dashed", alpha=0.5)+ 
#   # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) +
#   # scale_colour_manual(values = c('3-back' = "chartreuse3", '1-back' = "chartreuse4")) +
#   theme(text=element_text(family="Times New Roman"))

model.distance <- lmer(signal ~ (type*side*distance*task) + (type*side*distance| subject), data=df)
save(model.distance,file="/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/model.distance.Rda")
anova(model.distance)
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/model.distance.Rda")


emmean_column1 <- emmeans(model.distance, ~ type*side*distance*task,pbkrtest.limit = 253706)
emmean_plot.df1 <- emmean_column1 %>% broom::tidy()
save(emmean_plot.df1,file="/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df1.distance.Rda")

emmean_plot.df1$std.error.high <- emmean_plot.df1$estimate+emmean_plot.df1$std.error
emmean_plot.df1$std.error.low <- emmean_plot.df1$estimate-emmean_plot.df1$std.error
colnames(emmean_plot.df1) <- c("block","condition","side","distance","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df1, aes(distance, estimates, colour = condition)) + 
  facet_grid(block~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  geom_smooth(method=lm)+
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) + 
  theme(text=element_text(family="Times New Roman"))
  
emmean_column2 <- emmeans(model.sc.distance.3, ~ file_type, pbkrtest.limit = 393588)
emmean_plot.df2 <- emmean_column2 %>% broom::tidy()
save(emmean_plot.df2,file="/Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/univariate/SC_signal/r_output/emmean_plot.df2.Rda")
emmean_plot.df2$std.error.high <- emmean_plot.df2$estimate+emmean_plot.df2$std.error
emmean_plot.df2$std.error.low <- emmean_plot.df2$estimate-emmean_plot.df2$std.error
colnames(emmean_plot.df2) <- c("condition","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df2, aes(condition, signal, colour = file_type)) + 
  # facet_grid(block~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  geom_smooth(method=lm)+
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) + 
  theme(text=element_text(family="Times New Roman"))
 
emmean_column3 <- emmeans(model.sc.distance.3, ~ file_type*side, pbkrtest.limit = 393588)
emmean_plot.df3 <- emmean_column3 %>% broom::tidy()
save(emmean_plot.df3,file="/Volumes/GoogleDrive/My\ Drive/U01/AffPainTask_connectivity/analysis/univariate/SC_signal/r_output/emmean_plot.df3.Rda")
emmean_plot.df3$std.error.high <- emmean_plot.df3$estimate+emmean_plot.df3$std.error
emmean_plot.df3$std.error.low <- emmean_plot.df3$estimate-emmean_plot.df3$std.error
colnames(emmean_plot.df3) <- c("condition","side","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df3, aes(condition, signal, colour = file_type)) + 
  facet_grid(~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  geom_smooth(method=lm)+
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) + 
  theme(text=element_text(family="Times New Roman"))




ggplot(df, aes(quartile, signal, colour = file_type)) + 
  # facet_grid(block~side)+
  facet_grid(~side)+
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun.data = mean_se, geom="errorbar",width=0.2, size = .5) +
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) + 
  theme(text=element_text(family="Times New Roman"))

ggplot(df, aes(y, distance, colour = quartile)) + 
  facet_grid(~side)+
  geom_point(size=3.5, alpha=0.5)

df_mean = aggregate(x = df$signal, by = list(df$y,df$distance,df$file_type,df$side), FUN = mean)
colnames(df_mean) <- c('y','distance','file_type','side','signal')
ggplot(df_mean, aes(y, distance, colour = signal)) + 
  facet_grid(file_type~side)+
  geom_point(size=3.5, alpha=0.7)+ 
  scale_colour_distiller(palette = "YlOrBr")




ggplot(df, aes(distance, signal, colour = file_type)) + 
  # geom_point()
  facet_grid(~y)+
  stat_summary(fun = mean, geom = "point", size=2, alpha=0.5)+
  stat_summary(fun = mean, geom = "line")



#test
# JT
df_mean = aggregate(x = df$signal, by = list(df$y, df$x, df$z, df$quartile, df$file_type,df$side), FUN = mean)
colnames(df_mean) <- c('y','x', 'z', 'quartile','file_type','side','signal')
df_mean$distance_group <- floor(df_mean$distance/2)

ggplot(df_mean[df_mean$file_type == '3-back' & df_mean$side == 'right',], 
       aes(x, y, colour = signal)) + 
  facet_grid(file_type~quartile)+
  geom_point(size=3.5, alpha=0.7) + 
  scale_colour_distiller(palette = "YlOrBr")


ggplot(df_mean[df_mean$file_type == '3-back' & df_mean$side == 'left',], 
       aes(x, y, colour = signal)) + 
  facet_grid(file_type~distance_group)+
  geom_point(size=3.5, alpha=0.7) + 
  scale_colour_distiller(palette = "YlOrBr")


ggplot(df_mean[df_mean$file_type == '1-back' & df_mean$side == 'right',], 
       aes(x, y, colour = signal)) + 
  facet_grid(file_type~distance_group)+
  geom_point(size=3.5, alpha=0.7) + 
  scale_colour_distiller(palette = "YlOrBr")

df <- read_csv('/Volumes/GoogleDrive/My\ Drive/U01/working_memory/roi/segment_SC/sc_segment_subj_physio_correlation.csv')
df <- df[df$condition!='all',]
df$condition <- factor(df$condition, levels=c('3-back','1-back'))
df$subject <- as.factor(df$subject)
# df$block <- as.numeric(df$block)
df$distance <- as.numeric(df$distance)
df$quartile <- as.numeric(df$quartile)
# df$quartile = as.factor(df$quartile)

ggplot(df, aes(distance, SC_EDA_DDA_ampsum_mean_corr, colour = condition)) + 
  facet_grid(~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  # stat_summary(fun = mean, geom = "line", size=.25, alpha=0.5) +
  geom_smooth(method=lm)+
  geom_hline(yintercept=0,linetype = "dashed", alpha=0.5)+
  # geom_vline(xintercept=quantile(df$distance)['25%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['50%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['75%'],linetype = "dashed", alpha=0.5)+ 
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) +
  # scale_colour_manual(values = c('3-back' = "chartreuse3", '1-back' = "chartreuse4")) +
  theme(text=element_text(family="Times New Roman"))+ggtitle('EDA')

ggplot(df, aes(distance, SC_resp_rate_smooth_mean_corr, colour = condition)) + 
  facet_grid(~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  # stat_summary(fun = mean, geom = "line", size=.25, alpha=0.5) +
  geom_smooth(method=lm)+
  geom_hline(yintercept=0,linetype = "dashed", alpha=0.5)+
  # geom_vline(xintercept=quantile(df$distance)['25%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['50%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['75%'],linetype = "dashed", alpha=0.5)+ 
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) +
  # scale_colour_manual(values = c('3-back' = "chartreuse3", '1-back' = "chartreuse4")) +
  theme(text=element_text(family="Times New Roman"))+ggtitle('respiration')

ggplot(df, aes(distance, SC_ibi_mean_corr, colour = condition)) + 
  facet_grid(~side)+
  stat_summary(fun = mean, geom = "point", size=1, alpha=0.5) +
  # stat_summary(fun = mean, geom = "line", size=.25, alpha=0.5) +
  geom_smooth(method=lm)+
  geom_hline(yintercept=0,linetype = "dashed", alpha=0.5)+
  # geom_vline(xintercept=quantile(df$distance)['25%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['50%'],linetype = "dashed", alpha=0.5)+ 
  # geom_vline(xintercept=quantile(df$distance)['75%'],linetype = "dashed", alpha=0.5)+ 
  # scale_colour_manual(values = c('3-back' = "gray40", '1-back' = "gray70")) +
  # scale_colour_manual(values = c('3-back' = "chartreuse3", '1-back' = "chartreuse4")) +
  theme(text=element_text(family="Times New Roman"))+ggtitle('IBI')
