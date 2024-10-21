#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Presidential Election Results by State"),

    # Sidebar with a slider input for years 
    sidebarLayout(
        sidebarPanel(
            sliderInput("year",
                        "Year:",
                        # discrete slider with ticks every 4 year
                        min = 2004,
                        max = 2016,
                        value = 2004,
                        step = 4)
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("distPlot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    output$distPlot <- renderPlot({
        # generate maps for the selected year
        president_data <- read.csv("./1976-2020-president.csv")
        library(dplyr)
        party_winners_2004_2016 = president_data %>%
          filter(year %in% c(2004, 2008, 2012, 2016)) %>%
          group_by(year, state) %>%
          arrange(desc(candidatevotes)) %>%
          slice(1) %>%
          select(year, state, party_detailed) %>%
          mutate(year = as.factor(year))
        
      
        # draw the map by the selected year
        library(ggplot2)
        library(usmap)
        
        # create a map of the winning party by state
        data = party_winners_2004_2016 %>%
          filter(year == input$year)
        
        plot_usmap(data = data, values = "party_detailed") +
          scale_fill_manual(values = c("DEMOCRAT" = "blue", "REPUBLICAN" = "red", "other" = "grey")) +
          theme(legend.position = "right") +
          labs(title = "Presidential Election Results by State", fill = "Party") +
          facet_wrap(input$year)
        
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
