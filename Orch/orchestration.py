import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from Agentic_AI.search_agent import SearchAgent
from Agentic_AI.rca_agent import RCAAgent
from Agentic_AI.recommendation_agent import RecommendationAgent
from Agentic_AI.decision_agent import DecisionAgent

# Define the shared state
class AlertState(TypedDict):
    alert_json: dict
    similar_alerts: Optional[list]
    rca_output: Optional[str]
    recommendation: Optional[str]
    decision: Optional[dict]

# Initialize agents
search_agent = SearchAgent()
rca_agent = RCAAgent()
recommendation_agent = RecommendationAgent()
decision_agent = DecisionAgent()

# Define LangGraph functions
def run_search(state: AlertState) -> AlertState:
    state["similar_alerts"] = search_agent.search(state["alert_json"])
    return state

def run_rca(state: AlertState) -> AlertState:
    state["rca_output"] = rca_agent.perform_root_cause_analysis(
        state["alert_json"], state.get("similar_alerts", [])
    )
    return state

def run_recommendation(state: AlertState) -> AlertState:
    state["recommendation"] = recommendation_agent.generate_suggestions(
        alert_json=state["alert_json"],
        rca_output=state["rca_output"],
        similar_incidents=state.get("similar_alerts", [])
    )
    return state

def run_decision(state: AlertState) -> AlertState:
    state["decision"] = decision_agent.decide_action(state["recommendation"])
    return state

# Build the LangGraph pipeline
builder = StateGraph(AlertState)
builder.add_node("search_node", run_search)
builder.add_node("rca_node", run_rca)
builder.add_node("recommendation_node", run_recommendation)
builder.add_node("decision_node", run_decision)

builder.set_entry_point("search_node")
builder.add_edge("search_node", "rca_node")
builder.add_edge("rca_node", "recommendation_node")
builder.add_edge("recommendation_node", "decision_node")
builder.add_edge("decision_node", END)

graph = builder.compile()

class Orchestrator:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.rca_agent = RCAAgent()
        self.recommendation_agent = RecommendationAgent()
        self.decision_agent = DecisionAgent()

    def process_alert(self, alert_json: dict) -> dict:
        # Step 1: Search similar incidents
        print("[STEP 1] Searching for Similar Alerts...\n")
        similar_alerts = self.search_agent.search(alert_json)
        print(json.dumps(similar_alerts, indent=2) if similar_alerts else "No similar alerts found.\n")

        # Step 2: Perform RCA
        print("\n[STEP 2] Performing Root Cause Analysis...\n")
        rca_output = self.rca_agent.perform_root_cause_analysis(alert_json, similar_alerts)
        print(rca_output)

        # Step 3: Generate Recommendations
        print("\n[STEP 3] Generating Remediation Recommendations...\n")
        recommendation = self.recommendation_agent.generate_suggestions(alert_json, rca_output, similar_alerts)
        print(recommendation)

        # Step 4: Make Decision
        print("\n[STEP 4] Decision Based on Recommendations...\n")
        decision = self.decision_agent.decide_action(recommendation)
        print(json.dumps(decision, indent=2))

        return {
            "similar_alerts": similar_alerts,
            "rca": rca_output,
            "remediation_steps": recommendation,
            "decision": decision
        }


if __name__ == "__main__":
    alert_path = "Alert_2/ALERT001.json"

    try:
        with open(alert_path, "r") as f:
            alert_data = json.load(f)

        orchestrator = Orchestrator()
        result = orchestrator.process_alert(alert_data)

    except FileNotFoundError:
        print(f"Error: Alert file '{alert_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{alert_path}'.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
