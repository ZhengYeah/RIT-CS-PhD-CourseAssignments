########
#Task 1
########

# Question 1

# install.packages("stargazer")
library(stargazer)
statedata <- read.csv("statedata.csv")
stargazer(statedata)

# Question 2

library(tidyverse)
state_life <- select(statedata, state.abb, Life.Exp)
sorted_data <- state_life %>%
  arrange(desc(Life.Exp))
print(sorted_data)

# Question 3

state_corr <- select(statedata, Population, Income, Illiteracy, Life.Exp, HS.Grad, Frost, Murder, Area)
round(cor(state_corr), 2)
head(state_corr)

library(ggplot2)
# install.packages("reshape2")
library(reshape2)
melted_corr <- melt(cor(state_corr))
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 45, hjust = 1))





########
#Task 2
########

# Question 1

president_data = read.csv("1976-2020-president.csv")
head(president_data)

library(dplyr)
president_data %>%
  group_by(year) %>%
  arrange(desc(candidatevotes)) %>%
  summarize(candidatevotes = first(candidatevotes), party_detailed = first(party_detailed) ) %>%
  print()
  
# Question 2

library(usmap)
library(ggplot2)
party_winners = president_data %>%
  filter(year == 2020) %>%
  group_by(state) %>%
  arrange(desc(candidatevotes)) %>%
  slice(1) %>%
  select(state, party_detailed) %>%
  print()

plot_usmap(data = party_winners, values = "party_detailed") + 
  scale_fill_manual(values = c("blue", "red", "green")) + 
  theme(legend.position = "right") + 
  labs(title = "2020 Presidential Election Results by State", fill = "Party")

# Question 3

party_winners_2004_2016 = president_data %>%
  filter(year %in% c(2004, 2008, 2012, 2016)) %>%
  group_by(year, state) %>%
  arrange(desc(candidatevotes)) %>%
  slice(1) %>%
  select(year, state, party_detailed) %>%
  mutate(year = as.factor(year))

plot_usmap(data = party_winners_2004_2016, values = "party_detailed") +
  scale_fill_manual(values = c("blue", "green", "red")) +
  theme(legend.position = "right") +
  labs(title = "2004, 2008, 2012, 2016 Presidential Election Results by State", fill = "Party") +
  facet_wrap(~year)


