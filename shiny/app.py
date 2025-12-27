from shiny import App

# Import the UI and Server functions
# In the R code, this corresponds to source("ui.R") and source("server.R")
from app_ui import app_ui
from server import server

# Note: In Python, libraries and global configurations (global.R) 
# are handled by standard import statements within the modules themselves.

# Create the Shiny App instance
# In R, this corresponds to shinyApp(ui = ui, server = server)
app = App(app_ui, server)