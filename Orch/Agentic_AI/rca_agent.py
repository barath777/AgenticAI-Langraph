import os
import json
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# ---------- RCA Agent ----------
class RCAAgent:
    def __init__(self):
        self.logs_df = pd.read_csv("Data_2/application_Logs.csv")
        self.metrics_df = pd.read_csv("Data_2/kpi_metrics.csv")
        self.deployments_df = pd.read_csv("Data_2/production_deployment.csv")

        self.logs_df["timestamp"] = pd.to_datetime(self.logs_df["timestamp"], utc=True)
        self.metrics_df["Timestamp"] = pd.to_datetime(self.metrics_df["Timestamp"], utc=True)
        # self.deployments_df["deployment_timestamp"] = pd.to_datetime(self.deployments_df["deployment_timestamp"], utc=True)
        self.deployments_df["deployment_timestamp"] = pd.to_datetime(self.deployments_df["deployment_timestamp"], utc=True, format='mixed')


        llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
        self.chain = llm | StrOutputParser()

    def perform_root_cause_analysis(self, alert_json, similar_alerts):
        try:
            alert_time = pd.to_datetime(alert_json["timestamp"], utc=True)
            time_window = pd.Timedelta(minutes=30)

            logs_slice = self.logs_df[
                (self.logs_df["timestamp"] >= alert_time - time_window) &
                (self.logs_df["timestamp"] <= alert_time + time_window)
            ]

            metrics_slice = self.metrics_df[
                (self.metrics_df["Timestamp"] >= alert_time - time_window) &
                (self.metrics_df["Timestamp"] <= alert_time + time_window)
            ]

            deployments_slice = self.deployments_df[
                (self.deployments_df["deployment_timestamp"] >= alert_time - pd.Timedelta(days=3)) &
                (self.deployments_df["deployment_timestamp"] <= alert_time)
            ]
            # Only keep first 15 rows from each to reduce size
            logs_text = logs_slice.head(15).to_string(index=False)
            metrics_text = metrics_slice.head(15).to_string(index=False)
            deployments_text = deployments_slice.head(15).to_string(index=False)
            
            prompt = f"""
            You are an expert Site Reliability Engineer (SRE).

            Your task is to perform a Root Cause Analysis (RCA) for the given production alert. You **must identify correlations between application logs, KPI metrics, and production deployments** to derive a meaningful root cause — avoid generic conclusions.

            ### Requirements:
            - **Compare the timing** of errors in logs, anomalies in metrics, and recent deployments.
            - **Highlight any sequence** (e.g., deployment → metric spike → error) that suggests causal impact.
            - Your reasoning must **tie the root cause directly** to what is observed in the data.
            - Do **not** rely on assumptions or external knowledge — only what is observable.

            ### FORMAT (strict):
            - Correlation Observed: <A short paragraph explaining how logs, metrics, and deployments are connected>
            - Root Cause: <One-line root cause based on the correlation>
            - Diagnosis: <Concise remediation or fix>


            ALERT:
            {json.dumps(alert_json, indent=2)}

            SIMILAR HISTORICAL ALERTS:
            {json.dumps(similar_alerts, indent=2)}

            LOGS (first 15):
            {logs_text}

            METRICS (first 15):
            {metrics_text}

            DEPLOYMENTS (first 15):
            {deployments_text}

            Based on the above, identify the most likely root cause and suggest a high-level diagnosis.
            """

            result = self.chain.invoke(prompt)
            return result
        
        except Exception as e:
            return f"RCA failed: {str(e)}"
