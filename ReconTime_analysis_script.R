# Housekeeping
wd = 'C:/Users/micha.engeser/Documents/GitHub/Reconstructed_time_analysis'
setwd(wd)
dir()
if (!('tidyverse' %in% installed.packages()))
{install.packages("tidyverse")}
library(tidyverse)

if (!('dplyr' %in% installed.packages()))
{install.packages('dplyr')}
library(dplyr)

# specify subjet details
subNum = 102
ses = 1
Lab_ID = 'SX'
task = 'prp'

# loading file
sub_folder = paste0('sub-', Lab_ID, subNum)
ses_folder = paste0('ses-', ses)
file_name = paste0(sub_folder, '_', ses_folder, '_run-all_task-', task,'_events.csv')

setwd("..")
parent_dir = getwd()
setwd(wd)
file = file.path(parent_dir, 'Reconstructed_time_experiment', 'data', sub_folder, ses_folder, file_name)
event_table = read.csv2(file)

# remove target trials 
event_table = event_table[event_table$task_relevance != 'target', ]
relevant_event_table = event_table[event_table$task_relevance == 'non-target', ]
irrelevant_event_table = event_table[event_table$task_relevance == 'irrelevant', ]


# analysis of aud RT
RT_aud_mean = mean(event_table$RT_aud)

# split by SOA, SOA_lock and task relevance
task_relevance_lvl = c('non-target', 'irrelevant')
SOA_lock_lvl = c('onset', 'offset')
SOA_lvl = c(0,0.116,0.232,0.466)
duration_lvl = c(0.5,1,1.5)

# all task relevance

all_data = data.frame(matrix(ncol = 0, nrow = 1000))
counter = 0
for (i in SOA_lvl){
  
  counter = counter + 1 
  all_data$SOA[counter] = i
  all_data$mean_RT[counter] = mean(event_table$RT_aud[event_table$SOA == i])
  all_data$sd_RT[counter] = sd(event_table$RT_aud[event_table$SOA == i])
  
  for (ii in SOA_lock_lvl) {
    
    counter = counter + 1 
    all_data$SOA[counter] = i
    all_data$SOA_lock[counter] = ii
    all_data$mean_RT[counter] = mean(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii])
    all_data$sd_RT[counter] = sd(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii])
    
    for (iii in task_relevance_lvl) {
      
      counter = counter + 1 
      all_data$SOA[counter] = i
      all_data$SOA_lock[counter] = ii
      all_data$task_relevance[counter] = iii
      all_data$mean_RT[counter] = mean(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii & event_table$task_relevance == iii])
      all_data$sd_RT[counter] = sd(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii & event_table$task_relevance == iii])
      
      for (iv in duration_lvl) {
        
      counter = counter + 1 
      all_data$SOA[counter] = i
      all_data$SOA_lock[counter] = ii
      all_data$task_relevance[counter] = iii
      all_data$duration[counter] = iv
      all_data$mean_RT[counter] = mean(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii & event_table$task_relevance == iii & event_table$duration == iv])
      all_data$sd_RT[counter] = sd(event_table$RT_aud[event_table$SOA == i & event_table$SOA_lock == ii & event_table$task_relevance == iii & event_table$duration == iv])
}}}}


all_data = all_data[1:(counter), ]

# plotting 

ggplot(data=all_data, aes(x=SOA*1000, y=mean_RT, group=SOA_lock)) +
  geom_line(aes(color=SOA_lock))+
  geom_point()


