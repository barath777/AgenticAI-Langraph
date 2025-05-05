import os
import json
from Agentic_AI.search_agent import SearchAgent
from Agentic_AI.rca_agent import RCAAgent
from Agentic_AI.recommendation_agent import RecommendationAgent
from Agentic_AI.decision_agent import DecisionAgent
from loggers import log_alert_results_to_csv

class Orchestrator:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.rca_agent = RCAAgent()
        self.recommendation_agent = RecommendationAgent()
        self.decision_agent = DecisionAgent()

    def process_alert(self, alert_json: dict) -> dict:
        print("[STEP 1] Searching for Similar Alerts...\n")
        similar_alerts = self.search_agent.search(alert_json)
        print(json.dumps(similar_alerts, indent=2) if similar_alerts else "No similar alerts found.\n")

        print("\n[STEP 2] Performing Root Cause Analysis...\n")
        rca_output = self.rca_agent.perform_root_cause_analysis(alert_json, similar_alerts)
        print(rca_output)

        print("\n[STEP 3] Generating Remediation Recommendations...\n")
        recommendation = self.recommendation_agent.generate_suggestions(alert_json, rca_output, similar_alerts)
        print(recommendation)

        print("\n[STEP 4] Decision Based on Recommendations...\n")
        decision = self.decision_agent.decide_action(recommendation)
        print(json.dumps(decision, indent=2))

        return {
            "similar_alerts": similar_alerts,
            "rca": rca_output,
            "remediation_steps": recommendation,
            "decision": decision
        }

def run_batch(alert_folder: str = "Alert_2", output_folder: str = "alert_answers", csv_log_path: str = "alert_analysis_log.csv"):
    os.makedirs(output_folder, exist_ok=True)

    alert_files = [f for f in os.listdir(alert_folder) if f.endswith(".json")]

    if not alert_files:
        print(f"No JSON files found in folder '{alert_folder}'.")
        return

    orchestrator = Orchestrator()

    for filename in alert_files:
        alert_path = os.path.join(alert_folder, filename)
        print(f"\n========== Processing {filename} ==========\n")

        try:
            with open(alert_path, "r") as f:
                alert_data = json.load(f)

            result = orchestrator.process_alert(alert_data)

            result_filename = f"result_{filename}"
            result_path = os.path.join(output_folder, result_filename)
            with open(result_path, "w") as out_f:
                json.dump(result, out_f, indent=2)

            log_alert_results_to_csv(filename, alert_data, result, csv_log_path)

        except FileNotFoundError:
            print(f"Error: File '{alert_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{alert_path}'.")
        except Exception as e:
            print(f"Unexpected error in '{filename}': {str(e)}")

# Optional main entry point
if __name__ == "__main__":
    run_batch()
