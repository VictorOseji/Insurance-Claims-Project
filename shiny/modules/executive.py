from shiny import ui, render, reactive, module
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

# UI Module
def executive_mod_ui(id):
    return ui.TagList(
        # Executive Header
        ui.div(
            {"class": "bg-primary bg-gradient text-white p-4 mb-4 rounded-3",
             "style": "background: linear-gradient(135deg, #0062cc, #0056b3);"},
            ui.h1("Executive Dashboard", class_="display-5 fw-bold", style="color: white;"),
            ui.p("FNOL Claims Intelligence System - Strategic Business Insights", class_="lead mb-0")
        ),
        
        # Key Performance Indicators
        # layout_column_wrap width=3 corresponds to 1/4 of the row (12 cols)
        ui.layout_column_wrap(
            1/4,
            
            # Value Box 1: Annual Cost Savings
            # Using ui.card to mimic complex value_box structure with extra content
            ui.card(
                ui.card_header(
                    ui.div(
                        {"class": "d-flex justify-content-between align-items-center"},
                        "Annual Cost Savings",
                        ui.icon("currency-pound", class_="text-success fs-4")
                    )
                ),
                ui.card_body(
                    ui.h3("£2.4M"),
                    ui.p({"class": "text-muted"}, "15% reduction in claim leakage"),
                    ui.p({"class": "mb-0"}, 
                         ui.tags.small({"class": "text-success"}, ui.icon("arrow-up"), "12% vs last quarter")
                    )
                ),
                class_="border-success border-top border-3"
            ),
            
            # Value Box 2: Processing Time
            ui.card(
                ui.card_header(
                    ui.div(
                        {"class": "d-flex justify-content-between align-items-center"},
                        "Processing Time Reduction",
                        ui.icon("clock", class_="text-info fs-4")
                    )
                ),
                ui.card_body(
                    ui.h3("68%"),
                    ui.p({"class": "text-muted"}, "Average FNOL processing time"),
                    ui.p({"class": "mb-0"}, 
                         ui.tags.small({"class": "text-success"}, ui.icon("arrow-up"), "From 48hrs to 15hrs")
                    )
                ),
                class_="border-info border-top border-3"
            ),
            
            # Value Box 3: Customer Satisfaction
            ui.card(
                ui.card_header(
                    ui.div(
                        {"class": "d-flex justify-content-between align-items-center"},
                        "Customer Satisfaction",
                        ui.icon("emoji-smile", class_="text-primary fs-4")
                    )
                ),
                ui.card_body(
                    ui.h3("94%"),
                    ui.p({"class": "text-muted"}, "Net Promoter Score"),
                    ui.p({"class": "mb-0"}, 
                         ui.tags.small({"class": "text-success"}, ui.icon("arrow-up"), "+18 points YoY")
                    )
                ),
                class_="border-primary border-top border-3"
            ),
            
            # Value Box 4: Fraud Detection
            ui.card(
                ui.card_header(
                    ui.div(
                        {"class": "d-flex justify-content-between align-items-center"},
                        "Fraud Detection Rate",
                        ui.icon("shield-check", class_="text-warning fs-4")
                    )
                ),
                ui.card_body(
                    ui.h3("23%"),
                    ui.p({"class": "text-muted"}, "Improvement in fraud identification"),
                    ui.p({"class": "mb-0"}, 
                         ui.tags.small({"class": "text-success"}, ui.icon("arrow-up"), "£1.8M annual savings")
                    )
                ),
                class_="border-warning border-top border-3"
            )
        ),
        
        # Strategic Insights Section
        ui.layout_column_wrap(
            1/2,
            
            # Business Impact Summary
            ui.card(
                ui.card_header("Business Impact Summary"),
                ui.layout_column_wrap(1, 
                    # Key Metrics
                    ui.div(
                        {"class": "p-3"},
                        # Revenue Impact
                        ui.div(
                            {"class": "d-flex justify-content-between align-items-center mb-3 pb-3 border-bottom"},
                            ui.div(
                                ui.h6("Revenue Impact", class_="mb-1"),
                                ui.p("£3.2M additional revenue from improved retention", class_="text-muted small mb-0")
                            ),
                            ui.span({"class": "badge bg-success fs-6"}, "+8.5%")
                        ),
                        # Operational Efficiency
                        ui.div(
                            {"class": "d-flex justify-content-between align-items-center mb-3 pb-3 border-bottom"},
                            ui.div(
                                ui.h6("Operational Efficiency", class_="mb-1"),
                                ui.p("40% reduction in manual claim reviews", class_="text-muted small mb-0")
                            ),
                            ui.span({"class": "badge bg-info fs-6"}, "+40%")
                        ),
                        # Risk Management
                        ui.div(
                            {"class": "d-flex justify-content-between align-items-center mb-3 pb-3 border-bottom"},
                            ui.div(
                                ui.h6("Risk Management", class_="mb-1"),
                                ui.p("Enhanced reserve accuracy with AI predictions", class_="text-muted small mb-0")
                            ),
                            ui.span({"class": "badge bg-primary fs-6"}, "92%")
                        ),
                        # Market Position
                        ui.div(
                            {"class": "d-flex justify-content-between align-items-center"},
                            ui.div(
                                ui.h6("Market Position", class_="mb-1"),
                                ui.p("Industry-leading claims processing capability", class_="text-muted small mb-0")
                            ),
                            ui.span({"class": "badge bg-warning fs-6"}, "#1")
                        )
                    ),
                    # Visual Progress Indicators
                    ui.div(
                        {"class": "p-3 bg-light rounded"},
                        ui.h6("Strategic Initiative Progress", class_="mb-3"),
                        ui.div(
                            {"class": "mb-3"},
                            ui.h6("Digital Transformation", class_="small text-muted mb-1"),
                            ui.div({"class": "progress", "style": "height: 8px;"},
                                   ui.div({"class": "progress-bar bg-success", "role": "progressbar", 
                                           "style": "width: 85%;", "aria-valuenow": "85", "aria-valuemin": "0", "aria-valuemax": "100"}, 
                                          "85%")
                            )
                        ),
                        ui.div(
                            {"class": "mb-3"},
                            ui.h6("Customer Experience Enhancement", class_="small text-muted mb-1"),
                            ui.div({"class": "progress", "style": "height: 8px;"},
                                   ui.div({"class": "progress-bar bg-info", "role": "progressbar", 
                                           "style": "width: 92%;", "aria-valuenow": "92", "aria-valuemin": "0", "aria-valuemax": "100"}, 
                                          "92%")
                            )
                        ),
                        ui.div(
                            {"class": "mb-0"},
                            ui.h6("Cost Optimization", class_="small text-muted mb-1"),
                            ui.div({"class": "progress", "style": "height: 8px;"},
                                   ui.div({"class": "progress-bar bg-warning", "role": "progressbar", 
                                           "style": "width: 78%;", "aria-valuenow": "78", "aria-valuemin": "0", "aria-valuemax": "100"}, 
                                          "78%")
                            )
                        )
                    )
                ),
                ui.card_footer(
                    ui.navset_tab(
                        ui.nav("Quarterly", ui.icon("calendar-event")),
                        ui.nav("Annual", ui.icon("graph-up"))
                    )
                )
            ),
            
            # ROI Calculator
            ui.card(
                ui.card_header("Investment & ROI Analysis"),
                ui.div(
                    {"class": "p-3"},
                    # Investment Summary
                    ui.div(
                        {"class": "alert alert-info"},
                        ui.h6("Total Investment", class_="alert-heading"),
                        ui.p("£1.2M over 18 months including technology, implementation, and training", class_="mb-0")
                    ),
                    # ROI Metrics
                    ui.div(
                        {"class": "row mt-4"},
                        ui.div(
                            {"class": "col-6"},
                            ui.div({"class": "text-center p-3 bg-success bg-opacity-10 rounded"},
                                   ui.h3("218%", class_="text-success"),
                                   ui.p("3-Year ROI", class_="mb-0 text-muted")
                            )
                        ),
                        ui.div(
                            {"class": "col-6"},
                            ui.div({"class": "text-center p-3 bg-primary bg-opacity-10 rounded"},
                                   ui.h3("14", class_="text-primary"),
                                   ui.p("Months to Break-Even", class_="mb-0 text-muted")
                            )
                        )
                    ),
                    # Annual Benefits Breakdown
                    ui.div(
                        {"class": "mt-4"},
                        ui.h6("Annual Benefits Breakdown", class_="mb-3"),
                        
                        ui.div({"class": "mb-2 d-flex justify-content-between"},
                               ui.span("Cost Savings from Automation"),
                               ui.tags.strong("£1.1M")
                        ),
                        ui.div({"class": "mb-2 d-flex justify-content-between"},
                               ui.span("Fraud Prevention"),
                               ui.tags.strong("£1.8M")
                        ),
                        ui.div({"class": "mb-2 d-flex justify-content-between"},
                               ui.span("Improved Reserve Accuracy"),
                               ui.tags.strong("£0.8M")
                        ),
                        ui.div({"class": "mb-2 d-flex justify-content-between"},
                               ui.span("Customer Retention Value"),
                               ui.tags.strong("£3.2M")
                        ),
                        ui.tags.hr(),
                        ui.div({"class": "d-flex justify-content-between fw-bold"},
                               ui.span("Total Annual Benefits"),
                               ui.span({"class": "text-success"}, "£6.9M")
                        )
                    )
                )
            )
        ),
        
        # Competitive Advantage Section
        ui.layout_column_wrap(
            1,
            ui.card(
                ui.card_header("Competitive Advantage & Strategic Positioning"),
                ui.navset_card_pill(
                    ui.nav_panel(
                        "Market Leadership",
                        ui.div(
                            {"class": "p-3"},
                            ui.layout_column_wrap(
                                1/2,
                                # Competitive Comparison
                                ui.div(
                                    {"class": "col-md-6"},
                                    ui.h6("Industry Benchmarking", class_="mb-3"),
                                    output_widget("competitive_benchmark_plot")
                                ),
                                # Market Share Impact
                                ui.div(
                                    {"class": "col-md-6"},
                                    ui.h6("Market Share Impact", class_="mb-3"),
                                    ui.div(
                                        {"class": "alert alert-success"},
                                        ui.h6("3.2% Market Share Growth", class_="alert-heading"),
                                        ui.p("Outpacing industry average of 1.1% through superior claims experience", class_="mb-0")
                                    ),
                                    ui.div(
                                        {"class": "mt-3"},
                                        ui.h6("Key Differentiators", class_="mb-2"),
                                        ui.tags.ul(
                                            {"class": "list-unstyled"},
                                            ui.tags.li({"class": "mb-2"}, ui.icon("check-circle", class_="text-success me-2"), " Real-time claim cost prediction"),
                                            ui.tags.li({"class": "mb-2"}, ui.icon("check-circle", class_="text-success me-2"), " Industry-leading processing speed"),
                                            ui.tags.li({"class": "mb-2"}, ui.icon("check-circle", class_="text-success me-2"), " Superior fraud detection capabilities"),
                                            ui.tags.li({"class": "mb-0"}, ui.icon("check-circle", class_="text-success me-2"), " Enhanced customer experience")
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    ui.nav_panel(
                        "Strategic Roadmap",
                        ui.div(
                            {"class": "p-3"},
                            # Timeline Visualization
                            ui.div(
                                {"class": "timeline"},
                                # Phase 1
                                ui.div({"class": "timeline-item"},
                                       ui.div({"class": "timeline-marker bg-primary"}),
                                       ui.div({"class": "timeline-content"},
                                              ui.h6("Phase 1: Foundation", class_="text-primary"),
                                              ui.p("Core ML models and FNOL automation implemented", class_="text-muted small mb-0"),
                                              ui.span({"class": "badge bg-success"}, "Completed")
                                       )
                                ),
                                # Phase 2
                                ui.div({"class": "timeline-item"},
                                       ui.div({"class": "timeline-marker bg-info"}),
                                       ui.div({"class": "timeline-content"},
                                              ui.h6("Phase 2: Enhancement", class_="text-info"),
                                              ui.p("Advanced explainability and fraud detection integration", class_="text-muted small mb-0"),
                                              ui.span({"class": "badge bg-primary"}, "In Progress")
                                       )
                                ),
                                # Phase 3
                                ui.div({"class": "timeline-item"},
                                       ui.div({"class": "timeline-marker bg-warning"}),
                                       ui.div({"class": "timeline-content"},
                                              ui.h6("Phase 3: Expansion", class_="text-warning"),
                                              ui.p("Cross-product integration and predictive analytics", class_="text-muted small mb-0"),
                                              ui.span({"class": "badge bg-secondary"}, "Planned Q3 2023")
                                       )
                                ),
                                # Phase 4
                                ui.div({"class": "timeline-item"},
                                       ui.div({"class": "timeline-marker bg-secondary"}),
                                       ui.div({"class": "timeline-content"},
                                              ui.h6("Phase 4: Innovation", class_="text-secondary"),
                                              ui.p("AI-driven proactive risk management and prevention", class_="text-muted small mb-0"),
                                              ui.span({"class": "badge bg-light text-dark"}, "Planned 2024")
                                       )
                                )
                            )
                        )
                    ),
                    ui.nav_panel(
                        "Risk & Mitigation",
                        ui.div(
                            {"class": "p-3"},
                            ui.layout_column_wrap(
                                1/2,
                                # Risk Assessment
                                ui.div(
                                    {"class": "col-md-6"},
                                    ui.h6("Key Risk Factors", class_="mb-3"),
                                    ui.div(
                                        {"class": "risk-item mb-3"},
                                        ui.div({"class": "d-flex justify-content-between align-items-center"},
                                               ui.h6("Technology Adoption", class_="mb-0"),
                                               ui.span({"class": "badge bg-warning"}, "Medium")
                                        ),
                                        ui.p("Staff adaptation to new AI-driven processes", class_="text-muted small mb-2"),
                                        ui.div({"class": "progress", "style": "height: 6px;"},
                                               ui.div({"class": "progress-bar bg-warning", "role": "progressbar", 
                                                       "style": "width: 40%;", "aria-valuenow": "40", "aria-valuemin": "0", "aria-valuemax": "100"})
                                        )
                                    ),
                                    ui.div(
                                        {"class": "risk-item mb-3"},
                                        ui.div({"class": "d-flex justify-content-between align-items-center"},
                                               ui.h6("Model Accuracy", class_="mb-0"),
                                               ui.span({"class": "badge bg-success"}, "Low")
                                        ),
                                        ui.p("Continuous monitoring and retraining in place", class_="text-muted small mb-2"),
                                        ui.div({"class": "progress", "style": "height: 6px;"},
                                               ui.div({"class": "progress-bar bg-success", "role": "progressbar", 
                                                       "style": "width: 15%;", "aria-valuenow": "15", "aria-valuemin": "0", "aria-valuemax": "100"})
                                        )
                                    ),
                                    ui.div(
                                        {"class": "risk-item mb-0"},
                                        ui.div({"class": "d-flex justify-content-between align-items-center"},
                                               ui.h6("Regulatory Compliance", class_="mb-0"),
                                               ui.span({"class": "badge bg-success"}, "Low")
                                        ),
                                        ui.p("Full compliance with FCA guidelines", class_="text-muted small mb-2"),
                                        ui.div({"class": "progress", "style": "height: 6px;"},
                                               ui.div({"class": "progress-bar bg-success", "role": "progressbar", 
                                                       "style": "width: 10%;", "aria-valuenow": "10", "aria-valuemin": "0", "aria-valuemax": "100"})
                                        )
                                    )
                                ),
                                # Mitigation Strategies
                                ui.div(
                                    {"class": "col-md-6"},
                                    ui.h6("Mitigation Strategies", class_="mb-3"),
                                    ui.div(
                                        {"class": "alert alert-info"},
                                        ui.h6("Comprehensive Training Program", class_="alert-heading"),
                                        ui.p("Ongoing training and certification for all claims staff", class_="mb-0")
                                    ),
                                    ui.div(
                                        {"class": "alert alert-success"},
                                        ui.h6("Robust Governance Framework", class_="alert-heading"),
                                        ui.p("Regular model audits and compliance checks", class_="mb-0")
                                    ),
                                    ui.div(
                                        {"class": "alert alert-warning"},
                                        ui.h6("Phased Implementation", class_="alert-heading"),
                                        ui.p("Gradual rollout with continuous feedback loops", class_="mb-0")
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        
        # Executive Recommendations
        ui.layout_column_wrap(
            1,
            ui.card(
                ui.card_header("Executive Recommendations"),
                ui.div(
                    {"class": "p-3"},
                    ui.layout_column_wrap(
                        2/3, # col-md-8
                        ui.div(
                            {"class": "col-md-8"},
                            ui.h6("Strategic Recommendations", class_="mb-3"),
                            
                            ui.div(
                                {"class": "recommendation-item mb-3 p-3 bg-light rounded"},
                                ui.div(
                                    {"class": "d-flex align-items-start"},
                                    ui.div(
                                        {"class": "me-3"},
                                        ui.span({"class": "badge bg-primary fs-6"}, "1")
                                    ),
                                    ui.div(
                                        ui.h6("Scale to Commercial Lines", class_="mb-1"),
                                        ui.p("Extend the FNOL intelligence system to commercial insurance lines, projected to deliver additional £4.2M in annual savings", class_="text-muted mb-0")
                                    )
                                )
                            ),
                            
                            ui.div(
                                {"class": "recommendation-item mb-3 p-3 bg-light rounded"},
                                ui.div(
                                    {"class": "d-flex align-items-start"},
                                    ui.div(
                                        {"class": "me-3"},
                                        ui.span({"class": "badge bg-primary fs-6"}, "2")
                                    ),
                                    ui.div(
                                        ui.h6("Invest in Proactive Prevention", class_="mb-1"),
                                        ui.p("Leverage predictive insights to develop risk prevention services, creating new revenue streams", class_="text-muted mb-0")
                                    )
                                )
                            ),
                            
                            ui.div(
                                {"class": "recommendation-item mb-0 p-3 bg-light rounded"},
                                ui.div(
                                    {"class": "d-flex align-items-start"},
                                    ui.div(
                                        {"class": "me-3"},
                                        ui.span({"class": "badge bg-primary fs-6"}, "3")
                                    ),
                                    ui.div(
                                        ui.h6("Enhance Customer Portal", class_="mb-1"),
                                        ui.p("Integrate prediction capabilities directly into customer-facing portal for transparency and self-service", class_="text-muted mb-0")
                                    )
                                )
                            )
                        ),
                        
                        # Action Items
                        ui.div(
                            {"class": "col-md-4"},
                            ui.h6("Immediate Action Items", class_="mb-3"),
                            
                            ui.div(
                                {"class": "list-group"},
                                ui.div(
                                    {"class": "list-group-item d-flex justify-content-between align-items-center"},
                                    ui.span("Approve Phase 3 Funding"),
                                    ui.span({"class": "badge bg-danger"}, "Urgent")
                                ),
                                ui.div(
                                    {"class": "list-group-item d-flex justify-content-between align-items-center"},
                                    ui.span("Review Commercial Expansion Plan"),
                                    ui.span({"class": "badge bg-warning"}, "This Week")
                                ),
                                ui.div(
                                    {"class": "list-group-item d-flex justify-content-between align-items-center"},
                                    ui.span("Board Presentation Preparation"),
                                    ui.span({"class": "badge bg-info"}, "Next Week")
                                )
                            ),
                            
                            ui.div(
                                {"class": "mt-3"},
                                ui.input_action_button("schedule_review_btn", "Schedule Executive Review", class_="btn-primary w-100")
                            )
                        )
                    )
                )
            )
        )
    )

# Server Module Logic
@module.server
def executive_mod_server(input, output, session):
    
    @render_widget
    def competitive_benchmark_plot():
        metrics = ["Processing Speed", "Accuracy", "Customer Satisfaction", "Cost Efficiency"]
        xyz_values = [92, 94, 94, 88]
        industry_avg = [65, 72, 78, 70]
        best_competitor = [78, 85, 82, 75]
        
        fig = go.Figure()
        
        # Add XYZ Group data
        fig.add_trace(go.Scatterpolar(
            r=xyz_values,
            theta=metrics,
            fill='toself',
            name='XYZ Group',
            line=dict(color='#0062cc', width=3),
            marker=dict(color='#0062cc', size=8)
        ))
        
        # Add Industry Average
        fig.add_trace(go.Scatterpolar(
            r=industry_avg,
            theta=metrics,
            fill='toself',
            name='Industry Average',
            line=dict(color='#6c757d', width=2, dash='dash'),
            marker=dict(color='#6c757d', size=6)
        ))
        
        # Add Best Competitor
        fig.add_trace(go.Scatterpolar(
            r=best_competitor,
            theta=metrics,
            fill='toself',
            name='Best Competitor',
            line=dict(color='#dc3545', width=2),
            marker=dict(color='#dc3545', size=6)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.1
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @reactive.effect
    def _():
        # Triggered when schedule_review_btn is clicked
        if input.schedule_review_btn():
            ui.show_notification(
                "Executive review request sent. You will receive calendar invitation shortly.",
                type="success",
                duration=5
            )