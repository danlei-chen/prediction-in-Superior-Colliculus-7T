library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(nlme)
library(lmerTest)
library(pbkrtest)
library(emmeans)
library(dplyr)

######################################################################
######################################################################
df_all <- read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_emo_0.25_qua.csv')
df_all <- rbind(df_all, read_csv('/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/csv_output/SC_signal_pain_0.25_qua.csv'))
# df_all = df_all[df_all$roi!='all',]
df_all = df_all[df_all$roi_dim1!='all',]
# # get rid of subject iwth only 1 run to run the anova
# for (n in unique(df_all$subject)){
#   if (sum(df_all$subject==n)<6 ){
#     print(n)
#     df_all = df_all[!(df_all$subject==n),]
#   }
# }
df_all$run[df_all$run=='run-01']=1
df_all$run[df_all$run=='run-02']=2
df_all$run[df_all$run=='run-03']=3
df_all$run[df_all$run=='run-04']=4
df_all$run[df_all$run=='run-05']=5
# df_all$run = as.factor(df_all$run)
df_all$roi_dim1[df_all$roi_dim1=='left']='left SC'
df_all$roi_dim1[df_all$roi_dim1=='right']='right SC'
df_all$roi_dim1 = as.factor(df_all$roi_dim1)
df_all$roi_dim1 <- factor(df_all$roi_dim1, levels = c("left SC", "right SC"))
df_all$roi_dim3[df_all$roi_dim3=='upper']='dorsal SC'
df_all$roi_dim3[df_all$roi_dim3=='lower']='ventral SC'
df_all$roi_dim3 = as.factor(df_all$roi_dim3)
df_all$roi_dim3 <- factor(df_all$roi_dim3, levels = c("dorsal SC", "ventral SC"))
df_all$type[df_all$type=='active_motor']='Active_motor'
df_all$type[df_all$type=='active_no_motor']='Active_no_motor'
df_all$type[df_all$type=='passive']='Passive'
df_all$type = as.factor(df_all$type)
df_all$type <- factor(df_all$type, levels = c("Active_motor", "Active_no_motor", "Passive"))
df_all$task[df_all$task=='emo']='Visual task'
df_all$task[df_all$task=='pain']='Somatosensory task'
df_all$task <- factor(df_all$task, levels = c('Visual task', 'Somatosensory task'))
df_all <- df_all[((df_all$type=="Active_motor")|(df_all$type=="Passive")),]

#both
df_all$run <- as.factor(df_all$run)
# df_all$run = as.integer(df_all$run)
model.all <- lmer(mean_signal ~ (run + roi_dim1 + roi_dim3 + type + roi_dim1:type + roi_dim3:type + roi_dim1:roi_dim3:type)*task + (run + roi_dim1 + roi_dim3 + type + roi_dim1:type + roi_dim3:type + roi_dim1:roi_dim3:type| subject), data=df_all)
save(model.all,file="/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/model.all.SC.Rda")
anova(model.all)
load(paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/model.all.SC.Rda"))
print(anova(model.all))

#these models don't converge
# model.all.1 <- lmer(mean_signal ~ (run + roi + type + roi:type + run:type)*task + (run + roi + type + roi:type + run:type | subject), data=df_all)
# anova(model.all.1)
# model.all.2 <- lmer(mean_signal ~ (run + roi + type + roi:type + run:type)*task + (run + roi + type + roi:type | subject), data=df_all)
# anova(model.all.2)

#run and save model
emmean_column1 <- emmeans(model.all, ~ task*run*roi_dim1*roi_dim3*type)
emmean_plot.df1 <- emmean_column1 %>% broom::tidy()
save(emmean_plot.df1,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df1.SC.Rda"))
emmean_column2 <- emmeans(model.all, ~ task)
emmean_plot.df2 <- emmean_column2 %>% broom::tidy()
save(emmean_plot.df2,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df2.SC.Rda"))
emmean_column3 <- emmeans(model.all, ~ type)
emmean_plot.df3 <- emmean_column3 %>% broom::tidy()
save(emmean_plot.df3,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df3.SC.Rda"))
emmean_column4 <- emmeans(model.all, ~ roi_dim3)
emmean_plot.df4 <- emmean_column4 %>% broom::tidy()
save(emmean_plot.df4,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df4.SC.Rda"))
emmean_column5 <- emmeans(model.all, ~ roi_dim1*type)
emmean_plot.df5 <- emmean_column5 %>% broom::tidy()
save(emmean_plot.df5,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df5.SC.Rda"))
emmean_column6 <- emmeans(model.all, ~ roi_dim1*task)
emmean_plot.df6 <- emmean_column6 %>% broom::tidy()
save(emmean_plot.df6,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df6.SC.Rda"))
emmean_column7 <- emmeans(model.all, ~ roi_dim3*task)
emmean_plot.df7 <- emmean_column7 %>% broom::tidy()
save(emmean_plot.df7,file=paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/","Avd_CS+-ActPasUSnegneu_motor","/results/r_output/emmean_plot.df7.SC.Rda"))

load(paste0("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/model.all.SC.Rda"))
print(anova(model.all))

###############################################
#plot the emmean estimates of ALL effects
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df1.SC.Rda")
emmean_plot.df1$std.error.high <- emmean_plot.df1$estimate+emmean_plot.df1$std.error
emmean_plot.df1$std.error.low <- emmean_plot.df1$estimate-emmean_plot.df1$std.error
colnames(emmean_plot.df1) <- c("Task","Side","Layer","Decision","Run","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
#visual
ggplot(emmean_plot.df1[emmean_plot.df1$Task=='Visual task',], aes(Run, estimates, group=interaction(Decision,Side,Layer), colour = Decision)) + 
  facet_grid(Layer~Side) +
  geom_point(size=3, alpha=0.75)+
  # geom_line(size=1.1, aes(linetype=Decision)) +
  geom_line(size=1.2, alpha=0.75) +
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.4, alpha=0.75) + 
  scale_colour_manual(values = c('Active_motor' = "tan3", 'Passive' = "tan1"))+
  coord_cartesian(ylim = c(-6, 11)) + scale_y_continuous(breaks = round(seq(-6, 11, by = 2),1))
# theme(axis.line = element_line(colour = "black"),
  #       axis.text=element_text(size=12),
  #       axis.title=element_text(size=18),
  #       plot.title = element_text(size=16,face="bold")) + 
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
  # theme(panel.background = element_blank()) + 
  # theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))
#somatosensory
ggplot(emmean_plot.df1[emmean_plot.df1$Task=='Somatosensory task',], aes(Run, estimates, group=interaction(Decision,Side,Layer), colour = Decision)) + 
  facet_grid(Layer~Side) +
  geom_point(size=3, alpha=0.75)+
  # geom_line(size=1.1, aes(linetype=Decision)) +
  geom_line(size=1.2, alpha=0.75) +
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.4, alpha=0.75) + 
  scale_colour_manual(values = c('Active_motor' = "darkslategray3", 'Passive' = "darkslategray2"))+
  coord_cartesian(ylim = c(-6, 11)) + scale_y_continuous(breaks = round(seq(-6, 11, by = 2),1))
# theme(axis.line = element_line(colour = "black"),
#       axis.text=element_text(size=12),
#       axis.title=element_text(size=18),
#       plot.title = element_text(size=16,face="bold")) + 
# # ggtitle("mean signal in subejct-level superior colliculus mask") +
# theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
# theme(panel.background = element_blank()) + 
# theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))

emmean_plot.df1$std.error.df = emmean_plot.df1$std.error*sqrt(emmean_plot.df1$df)
###############################################
# layers
emmean_plot.df1_std.error <- emmean_plot.df1 %>% group_by(Layer) %>% summarise_at(vars("std.error.df"), sum)
emmean_plot.df1_df <- emmean_plot.df1 %>% group_by(Layer) %>% summarise_at(vars("df","estimates"), mean)
emmean_plot.df1_std.error$std.err <- sqrt(emmean_plot.df1_std.error$std.error.df)/sqrt(emmean_plot.df1_df$df)
emmean_plot.df1_std.error$std.error.high <- emmean_plot.df1_df$estimates + emmean_plot.df1_std.error$std.err
emmean_plot.df1_std.error$std.error.low <- emmean_plot.df1_df$estimates - emmean_plot.df1_std.error$std.err
emmean_plot.df1_std.error$estimates <- emmean_plot.df1_df$estimates
ggplot(emmean_plot.df1_std.error, aes(Layer, estimates)) +
  # stat_summary(fun.y = mean, geom = "bar") +
  # # stat_summary(fun.y = mean, geom = "line") +
  # stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .7) +
  geom_bar(stat = "identity", alpha=0.9) +
  # geom_point(size=3, alpha=0.75)+
  # geom_line(size=1.2, alpha=0.75) +
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, alpha=0.75) +
  theme(legend.position="none")+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# layers
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Layer) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Layer, estimates)) + 
  stat_summary(fun.y = mean, geom = "bar") +
  # stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .7) +
  # geom_bar(stat = "identity", alpha=0.9) +
  # # geom_point(size=3, alpha=0.75)+
  # # geom_line(size=1.2, alpha=0.75) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, alpha=0.75) +
  theme(legend.position="none")+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# decision
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Decision) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Decision, estimates, fill=Decision)) + 
  stat_summary(fun.y = mean, geom = "bar") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .7) +
  # geom_bar(stat = "identity", alpha=0.9) +
  # # geom_point(size=3, alpha=0.75)+
  # # geom_line(size=1.2, alpha=0.75) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, alpha=0.75) +
  theme(legend.position="none")+
  scale_fill_manual(values = c("gray40", "gray80"))+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# sensory modalities
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Task) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Task, estimates, fill=Task)) + 
  stat_summary(fun.y = mean, geom = "bar") +
  # stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .7, color='grey40') +
  # geom_bar(stat = "identity", alpha=0.9) +
  # # geom_point(size=3, alpha=0.75)+
  # # geom_line(size=1.2, alpha=0.75) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, alpha=0.75) +
  scale_fill_manual(values = c("tan2", "darkslategray3"))+
  theme(legend.position="none")+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# side x decision
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Side,Decision) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Side, estimates, group=Decision, colour=Decision)) + 
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .5) +
  # # geom_bar(stat = "identity", alpha=0.9) +
  # geom_point(size=3)+
  # geom_line(size=1.2) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1) +
  scale_color_manual(values = c("gray40", "gray80"))+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# side x Sensory modality
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Side,Task) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Side, estimates, group=Task, colour=Task)) + 
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .5) +
  # # geom_bar(stat = "identity", alpha=0.9) +
  # geom_point(size=3)+
  # geom_line(size=1.2) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1) +
  scale_color_manual(values = c("tan2", "darkslategray3"))+
  theme(legend.position="none")+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

###############################################
# layers x task
# emmean_plot.df1_sub <- emmean_plot.df1 %>% group_by(Layer,Task) %>% 
#   summarise_at(vars("estimates", "std.error.high", "std.error.low"), mean)
ggplot(emmean_plot.df1, aes(Layer, estimates, group=Task, colour=Task)) + 
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .5) +
  # # geom_bar(stat = "identity", alpha=0.9) +
  # geom_point(size=3)+
  # geom_line(size=1.2, alpha=0.8) +
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, alpha=0.8) +
  scale_color_manual(values = c("tan2", "darkslategray3"))+
  theme(legend.position="none")+
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))+
  coord_cartesian(ylim = c(-4, 6)) + scale_y_continuous(breaks = round(seq(-4, 6, by = 1),1))

df_all_sub_agg <- aggregate(df_all$mean_signal,by=(list(df_all$subject,df_all$roi_dim3,df_all$task)), FUN=mean)
colnames(df_all_sub_agg) <- c('subject', 'layers','task','mean_signal')
t.test(df_all_sub_agg[df_all_sub_agg$task=='emo',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='emo',]$layers=='upper'], df_all_sub_agg[df_all_sub_agg$task=='emo',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='emo',]$layers=='lower'], paired = TRUE,alternative = "greater")
t.test(df_all_sub_agg[df_all_sub_agg$task=='pain',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='pain',]$layers=='upper'], df_all_sub_agg[df_all_sub_agg$task=='pain',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='pain',]$layers=='lower'], paired = TRUE,alternative = "less")





#OLD
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df2.SC.Rda")
emmean_plot.df2$std.error.high <- emmean_plot.df2$estimate+emmean_plot.df2$std.error
emmean_plot.df2$std.error.low <- emmean_plot.df2$estimate-emmean_plot.df2$std.error
colnames(emmean_plot.df2) <- c("Sensory_modalities","Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df2, aes(Decision,estimates, fill=Sensory_modalities, group=1)) +
  facet_wrap(~Sensory_modalities)+
  geom_bar(stat = "identity")+
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1)+
  scale_fill_manual(values = c("tan3", "darkslategray2") ) +
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold"))
  # coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1))
  # # coord_cartesian(ylim = c(-1, 8)) +
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+
  # theme(panel.background = element_blank()) +
  # theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))

df_all_sub_agg <- aggregate(df_all$mean_signal,by=(list(df_all$subject,df_all$task)), FUN=mean)
colnames(df_all_sub_agg) <- c('subject', 'task','mean_signal')
t.test(df_all_sub_agg$mean_signal[df_all_sub_agg$task=='Visual task'], mu = 0, alternative = "greater")
t.test(df_all_sub_agg$mean_signal[df_all_sub_agg$task=='Somatosensory task'], mu = 0, alternative = "less")

###############################################
#plot the emmean estimates of aversive effect
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df3.SC.Rda")
emmean_plot.df3$std.error.high <- emmean_plot.df3$estimate+emmean_plot.df3$std.error
emmean_plot.df3$std.error.low <- emmean_plot.df3$estimate-emmean_plot.df3$std.error
colnames(emmean_plot.df3) <- c("Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df3, aes(Decision,estimates, fill=Decision, group=1)) + 
  geom_bar(stat = "identity")+
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1)+
  scale_fill_manual(values = c("gray40", "gray80") ) +
  theme(axis.line = element_line(colour = "black"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=18),
        plot.title = element_text(size=16,face="bold")) +
  # coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1)) +
  theme(legend.position="none")
  # # coord_cartesian(ylim = c(-1, 8)) +
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
  # theme(panel.background = element_blank()) + 
  #  + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))

df_all_sub_agg <- aggregate(df_all$mean_signal,by=(list(df_all$subject,df_all$type)), FUN=sum)
colnames(df_all_sub_agg) <- c('subject', 'type','mean_signal')
t.test(df_all_sub_agg$mean_signal[df_all_sub_agg$type=='Active_motor'], df_all_sub_agg$mean_signal[df_all_sub_agg$type=='Passive'], paired = TRUE,alternative = "greater")

###############################################
#plot the emmean estimates of layer effect
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df4.SC.Rda")
emmean_plot.df4$std.error.high <- emmean_plot.df4$estimate+emmean_plot.df4$std.error
emmean_plot.df4$std.error.low <- emmean_plot.df4$estimate-emmean_plot.df4$std.error
colnames(emmean_plot.df4) <- c("Layers","Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df4, aes(Layers,estimates,group = Decision)) + 
  geom_point(size=3, colour="gray50")+
  # geom_line(size=1.1, aes(linetype=Aversiveness)) +
  geom_line(size=1.2, colour="gray50")+
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1, colour="gray50")+
  coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1)) 
  # # coord_cartesian(ylim = c(-1, 8)) +
  # theme(axis.line = element_line(colour = "black"),
  #       axis.text=element_text(size=12),
  #       axis.title=element_text(size=18),
  #       plot.title = element_text(size=16,face="bold")) + 
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
  # theme(panel.background = element_blank()) + 
  # theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))

###############################################
#plot the emmean estimates of hemisphere*type effect
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df5.SC.Rda")
emmean_plot.df5$std.error.high <- emmean_plot.df5$estimate+emmean_plot.df5$std.error
emmean_plot.df5$std.error.low <- emmean_plot.df5$estimate-emmean_plot.df5$std.error
colnames(emmean_plot.df5) <- c("Side", "Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df5, aes(Side, estimates, group=Decision, colour=Decision)) + 
  geom_point(size=3)+
  # geom_line(size=1.1, aes(linetype=Aversiveness)) +
  geom_line(size=1.2)+
  geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1)+
  scale_colour_manual(values = c("gray40", "gray80"))
  # coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1)) 
  # # coord_cartesian(ylim = c(-1, 12)) +
  # theme(axis.line = element_line(colour = "black"),
  #       axis.text=element_text(size=12),
  #       axis.title=element_text(size=18),
  #       plot.title = element_text(size=16,face="bold")) + 
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
  # theme(panel.background = element_blank()) + 
  # theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))

###############################################
#plot the emmean estimates of layer*task effect
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df6.SC.Rda")
emmean_plot.df6$std.error.high <- emmean_plot.df6$estimate+emmean_plot.df6$std.error
emmean_plot.df6$std.error.low <- emmean_plot.df6$estimate-emmean_plot.df6$std.error
colnames(emmean_plot.df6) <- c("Side", "Task","Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
# emmean_plot.df6$Layers <- factor(emmean_plot.df6$Layers, levels = c("ventral SC", "dorsal SC"))
# emmean_plot.df6$Layers <- factor(emmean_plot.df6$Layers, levels = c("dorsal SC", "ventral SC"))
ggplot(emmean_plot.df6, aes(Side, estimates, group=Task, colour=Task)) + 
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .5) +
  # geom_point(size=3)+
  # # geom_line(size=1.1, aes(linetype=Aversiveness)) +
  # geom_line(size=1.2)+
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1)+
  scale_colour_manual(values = c("tan2", "darkslategray3"))
  # coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1)) 
  # theme(axis.line = element_line(colour = "black"),
  #       axis.text=element_text(size=12),
  #       axis.title=element_text(size=18),
  #       plot.title = element_text(size=16,face="bold")) + 
  # # ggtitle("mean signal in subejct-level superior colliculus mask") +
  # theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
  # theme(panel.background = element_blank()) + 
  # coord_flip() +
  # theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))


df_all_sub_agg <- aggregate(df_all$mean_signal,by=(list(df_all$subject,df_all$roi_dim3,df_all$task)), FUN=mean)
colnames(df_all_sub_agg) <- c('subject', 'layers','task','mean_signal')
t.test(df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$layers=='dorsal SC'], df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$layers=='ventral SC'], paired = TRUE,alternative = "greater")
t.test(df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$layers=='dorsal SC'], df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$layers=='ventral SC'], paired = TRUE,alternative = "less")

###############################################
#plot the emmean estimates of layer*task effect
load("/Volumes/GoogleDrive/My\ Drive/U01/decision_making/analysis/univariate/SC_signal/Avd_CS+-ActPasUSnegneu_motor/results/r_output/emmean_plot.df7.SC.Rda")
emmean_plot.df7$std.error.high <- emmean_plot.df7$estimate+emmean_plot.df7$std.error
emmean_plot.df7$std.error.low <- emmean_plot.df7$estimate-emmean_plot.df7$std.error
colnames(emmean_plot.df7) <- c("Layers", "Task","Decision","estimates","std.error","df","conf.low","conf.high","std.error.high","std.error.low")
ggplot(emmean_plot.df7, aes(Layers, estimates, group=Task, colour=Task)) + 
  stat_summary(fun.y = mean, geom = "line") +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom="errorbar", width=0.1, size = .5) +
  # geom_point(size=3)+
  # # geom_line(size=1.1, aes(linetype=Aversiveness)) +
  # geom_line(size=1.2)+
  # geom_errorbar(aes(ymin=std.error.low, ymax=std.error.high), width=0.1)+
  scale_colour_manual(values = c("tan2", "darkslategray3"))
# coord_cartesian(ylim = c(-4, 4)) + scale_y_continuous(breaks = round(seq(-4, 4, by = 1),1)) 
# theme(axis.line = element_line(colour = "black"),
#       axis.text=element_text(size=12),
#       axis.title=element_text(size=18),
#       plot.title = element_text(size=16,face="bold")) + 
# # ggtitle("mean signal in subejct-level superior colliculus mask") +
# theme(strip.text.x = element_text(size = 18),strip.text.y = element_text(size = 18))+ 
# theme(panel.background = element_blank()) + 
# coord_flip() +
# theme(legend.position="none") + theme(text=element_text(family="Times New Roman")) + theme(strip.background =element_rect(fill="white"))


df_all_sub_agg <- aggregate(df_all$mean_signal,by=(list(df_all$subject,df_all$roi_dim3,df_all$task)), FUN=mean)
colnames(df_all_sub_agg) <- c('subject', 'layers','task','mean_signal')
t.test(df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$layers=='dorsal SC'], df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Visual task',]$layers=='ventral SC'], paired = TRUE,alternative = "greater")
t.test(df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$layers=='dorsal SC'], df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$mean_signal[df_all_sub_agg[df_all_sub_agg$task=='Somatosensory task',]$layers=='ventral SC'], paired = TRUE,alternative = "less")
