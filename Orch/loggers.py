import pandas as pd
import os

def log_alert_results_to_csv(alert_file_name: str, alert_json: dict, result: dict, csv_path: str = "alert_analysis_log.csv"):
    alert_type = alert_json.get("alert_type", "Unknown")
    similar_alerts_summary = result["similar_alerts"] if result["similar_alerts"] else "None"
    rca_output = result["rca"]
    recommendation = result["remediation_steps"]
    decision = result["decision"]

    decision_label = decision.get("decision", "N/A")
    confidence = decision.get("average_confidence", "N/A")

    if isinstance(similar_alerts_summary, list):
        similar_alerts_summary = "\n---\n".join(
            [f"Type: {a['alert_type']}, Score: {a['similarity_score']:.2f}, RC: {a.get('root_cause', '')}" for a in similar_alerts_summary]
        )

    row_data = {
        "Alert File": alert_file_name,
        "Alert Type": alert_type,
        "Similar Incidents": similar_alerts_summary,
        "Search Agent Output": str(result["similar_alerts"]),
        "RCA Output": rca_output,
        "Recommendation Output": recommendation,
        "Decision Output": decision_label,
        "Confidence Score": confidence
    }

    # Append to CSV or create new one
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row_data])

    df_updated.to_csv(csv_path, index=False)
    print(f"Logged to CSV: {csv_path}")
