
library(usmap)  
library(ggplot2) 


visited_states <- c() #Fill this in with the states you've visited (e.g., "CA", "TX", "NY")


states_df <- data.frame(state = state.abb, visited = "Not Visited")
states_df$visited[match(visited_states, states_df$state)] <- "Visited"


plot_usmap(data = states_df, values = "visited", color = "black") +
  scale_fill_manual(values = c("Visited" = "", "Not Visited" = "")) + #Choose your colors here
  labs(title = "States I've Visited", fill = "Status") +
  theme_minimal()