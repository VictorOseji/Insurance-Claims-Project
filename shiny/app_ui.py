from shiny import ui

# Import the UI functions from the module files.
# Ensure that you have files like executive_mod.py, dashboard_mod.py, etc., 
# in a folder named 'modules', and that they export functions ending with _ui.
from modules.executive_mod import executive_mod_ui
from modules.dashboard_mod import dashboard_mod_ui
from modules.prediction_mod import prediction_mod_ui
from modules.data_mod import data_mod_ui
from modules.model_mod import model_mod_ui
from modules.explainability_mod import explainability_mod_ui
from modules.insights_mod import insights_mod_ui


def app_ui():
    return ui.page_navbar(
        # Custom Title with Logo and Text
        ui.div(
            {"class": "d-flex align-items-center fw-bold"},
            ui.img(
                src="Victor_Logo2.png", 
                height="50px", 
                style="margin-right:30px;", 
                class_="me-2"
            ),
            ui.div(
                {"style": "display: flex; flex-direction: column; line-height: 1.2;"},
                ui.span(
                    {"style": "color:#373737; font-size: 18px; font-weight: 600;"}, 
                    "FNOL Claims"
                ),
                ui.span(
                    {"style": "color:#373737; font-size: 18px; font-weight: 600;"}, 
                    "Intelligence"
                )
            )
        ),
        
        # Theme: Minty Bootswatch with custom primary color
        theme=ui.theme(
            bootswatch="minty",
            primary="#0062cc"
        ),
        
        # Header: Include custom CSS
        header=ui.TagList(
            ui.include_css("www/custom.css")
        ),
        
        # Executive Tab (Overview)
        ui.nav_panel(
            "Overview",
            executive_mod_ui("executive"),
            icon=ui.icon("speedometer2")  # Replaces icon("dashboard")
        ),
        
        # Dashboard Tab
        ui.nav_panel(
            "Dashboard",
            dashboard_mod_ui("dashboard"),
            icon=ui.icon("grid")  # Replaces icon("dashboard")
        ),
        
        # Prediction Tab
        ui.nav_panel(
            "Prediction",
            prediction_mod_ui("prediction"),
            icon=ui.icon("graph-up-arrow")  # Replaces icon("chart-line")
        ),
        
        # Data Management Tab
        ui.nav_panel(
            "Data Management",
            data_mod_ui("data"),
            icon=ui.icon("database")
        ),
        
        # Model Management Tab
        ui.nav_panel(
            "Model Management",
            model_mod_ui("model"),
            icon=ui.icon("gear-wide-connected")  # Replaces icon("cogs")
        ),
        
        # Explainability Tab
        ui.nav_panel(
            "Explainability",
            explainability_mod_ui("explainability"),
            icon=ui.icon("question-circle")
        ),
        
        # Business Insights Tab
        ui.nav_panel(
            "Business Insights",
            insights_mod_ui("insights"),
            icon=ui.icon("lightbulb")
        )
    )