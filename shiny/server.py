from shiny import App, reactive
import mlflow

# Import module servers
from modules.executive_mod import executive_mod_server
from modules.dashboard_mod import dashboard_mod_server
from modules.prediction_mod import prediction_mod_server
from modules.data_mod import data_mod_server
from modules.model_mod import model_mod_server
from modules.explainability_mod import explainability_mod_server
from modules.insights_mod import insights_mod_server

# Import utility functions
# Assuming these utils files exist in your 'utils' directory
from utils import data_processing, modeling, explainability, mlflow_utils


def server(input, output, session):
    
    # --- Helper Class to mimic R's reactiveValues ---
    class ReactiveValues:
        def __init__(self):
            self.claims_data = reactive.value(None)
            self.policy_data = reactive.value(None)
            self.processed_data = reactive.value(None)
            self.model = reactive.value(None)
            self.model_performance = reactive.value(None)
            self.predictions = reactive.value(None)

    # Initialize reactive values container
    rv = ReactiveValues()

    # --- Initialize Modules ---
    
    # Executive module (doesn't take data args)
    executive_mod_server("executive")
    
    # Initialize Data and Model modules first so we can capture their return values
    # Assuming these Python modules return objects containing 'data_updated', 'model_updated', etc.
    data_server = data_mod_server("data")
    model_server = model_mod_server("model", mlflow_utils.get_mlflow_client())

    # Initialize other modules passing the reactive values
    dashboard_mod_server("dashboard", rv.claims_data, rv.model_performance)
    prediction_mod_server("prediction", rv.model, rv.processed_data)
    explainability_mod_server("explainability", rv.model, rv.processed_data)
    insights_mod_server("insights", rv.claims_data, rv.policy_data, rv.model_performance)

    # --- Initial Data Loading ---
    
    @reactive.effect
    def _load_initial_data():
        # Load initial claims and policy data
        rv.claims_data.set(data_processing.load_initial_claims_data())
        rv.policy_data.set(data_processing.load_initial_policy_data())

        # Process data
        rv.processed_data.set(data_processing.process_data(rv.claims_data(), rv.policy_data()))

        # Load initial model
        rv.model.set(modeling.load_initial_model())

        # Calculate model performance
        rv.model_performance.set(modeling.evaluate_model(rv.model(), rv.processed_data()))

    # --- Handle Data Updates from Data Module ---
    
    # In R: observeEvent(input$data_updated, ...)
    # In Python: We observe the reactive returned by the module server
    @reactive.effect
    @reactive.event(data_server.data_updated)
    def _handle_data_updates():
        rv.claims_data.set(data_server.updated_claims_data())
        rv.policy_data.set(data_server.updated_policy_data())
        rv.processed_data.set(data_processing.process_data(rv.claims_data(), rv.policy_data()))

    # --- Handle Model Updates from Model Module ---
    
    @reactive.effect
    @reactive.event(model_server.model_updated)
    def _handle_model_updates():
        rv.model.set(model_server.updated_model())
        rv.model_performance.set(modeling.evaluate_model(rv.model(), rv.processed_data()))

    # End of server