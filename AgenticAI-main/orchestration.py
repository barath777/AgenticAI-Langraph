import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from search_agent import SearchAgent
from rca_agent import RCAAgent
from recommendation_agent import RecommendationAgent
from decision_agent import DecisionAgent
from Remediation_Agent import RemediationAgent 

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
Remediation_agent = RemediationAgent()

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

def run_remediation(state: AlertState) -> AlertState:
    print("\n[STEP 5] Remediation Execution...\n")

    similar_alerts = state.get("similar_alerts", [])
    alert = state["alert_json"]
    decision = state.get("decision", {})

    if decision.get("decision") != "auto_remediate":
        print("Decision not suitable for automatic remediation.")
        return state

    for match in similar_alerts:
         if (
        match["similarity_score"] >= 0.75 and
        match["alert_type"] == alert.get("alert_type") and
        match["severity"] == alert.get("severity") and
        match.get("application") == alert.get("application") and
        match.get("alert_description", "").strip().lower() == alert.get("alert_description", "").strip().lower()
    ):
            success = Remediation_agent.execute_remediation(decision, alert, similar_alerts)
            if not success:
                print("Remediation failed.")
            return state

    print("Remediation skipped: No similar alert with matching conditions.")
    return state

# Build the LangGraph pipeline
builder = StateGraph(AlertState)
builder.add_node("search_node", run_search)
builder.add_node("rca_node", run_rca)
builder.add_node("recommendation_node", run_recommendation)
builder.add_node("decision_node", run_decision)
builder.add_node("remediation_node", run_remediation)

builder.set_entry_point("search_node")
builder.add_edge("search_node", "rca_node")
builder.add_edge("rca_node", "recommendation_node")
builder.add_edge("recommendation_node", "decision_node")
builder.add_edge("decision_node", "remediation_node")  
builder.add_edge("remediation_node", END)

graph = builder.compile()

class Orchestrator:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.rca_agent = RCAAgent()
        self.recommendation_agent = RecommendationAgent()
        self.decision_agent = DecisionAgent()
        self.Remediation_agent = RemediationAgent()

    def process_alert(self, alert_json: dict) -> dict:
        # Step 1: Search similar incidents
        print("[STEP 1] Searching for Similar Alerts...\n")
        print("\nIncoming Alert:")
        print(alert_json)
        print("\nSimilar Alerts are :")
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

        # Step 5: Execute Remediation
        print("\n[STEP 5] Remediation Execution...\n")
        remediation_performed = False
        for match in similar_alerts or []:
            if (
                match["similarity_score"] >= 0.75 and
                match["alert_type"] == alert_json.get("alert_type") and
                alert_json.get("application") == match.get("application", "") and
                match.get("alert_description", "").strip().lower() == alert_json.get("alert_description", "").strip().lower()
            ):
                remediation_performed = self.Remediation_agent.execute_remediation(decision, alert_json, similar_alerts)
                break

        if not remediation_performed:
            print("Remediation skipped: No matching historical alert met the criteria.")

        return {
            "similar_alerts": similar_alerts,
            "rca": rca_output,
            "remediation_steps": recommendation,
            "decision": decision
        }


if __name__ == "__main__":
    alert_path = "Alert_2/ALERT005.json"

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
