import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# -------- Setup LLM --------
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama3-8b-8192")

# -------- Load Alert JSON --------
with open("Alerts/ALERT001.json", "r") as file:
    alert_json = json.load(file)

# -------- Preprocessing Historical Alerts and FAISS Setup --------
df = pd.read_csv("Data/Historical_Alerts_Resolutions.csv").fillna("")
df["alert_timestamp"] = df["alert_timestamp"].astype(str)
df["text"] = (
    df["alert_timestamp"] + " " + df["alert_type"] + " " + df["resolution_steps"] + " " 
    + df["application"] + " " + df["severity"] + " " 
    + df["root_cause"] + " " + df["change_implemented"] + " " + df["post_resolution_status"]
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

# -------- SearchAgent Class --------
class SearchAgent:
    def __init__(self, model, index, df):
        self.model = model
        self.index = index
        self.df = df

    def search(self, alert_json, threshold=0.3):
        alert_text = (
            alert_json.get("timestamp", "") + " " +
            alert_json.get("description", "") + " " +
            alert_json.get("alert_type", "") + " " +
            alert_json.get("application", "") + " " +
            " ".join(alert_json.get("affected_services", [])) + " " +
            alert_json.get("severity", "")
        )
        alert_embedding = self.model.encode([alert_text]).astype("float32")
        alert_embedding /= np.linalg.norm(alert_embedding)

        distances, indices = self.index.search(alert_embedding, k=3)
        similar_alerts = []
        
        for i in range(len(indices[0])):
            if distances[0][i] > threshold:
                alert = self.df.iloc[indices[0][i]]
                similar_alerts.append({
                    "alert_id": alert["alert_id"],
                    "alert_type": alert["alert_type"],
                    "resolution_steps": alert["resolution_steps"],
                    "similarity_score": float(distances[0][i]),
                })
        
        return similar_alerts if similar_alerts else []

# -------- RCAAgent Class --------
class RCAAgent:
    def __init__(self):
        self.logs_df = pd.read_csv("Data1/Application_Logs.csv")
        self.metrics_df = pd.read_csv("Data1/kpi_metrics_data.csv")
        self.deployments_df = pd.read_csv("Data/Production_Deployments.csv")

        self.logs_df["timestamp"] = pd.to_datetime(self.logs_df["timestamp"], utc=True)
        self.metrics_df["Timestamp"] = pd.to_datetime(self.metrics_df["Timestamp"], utc=True)
        self.deployments_df["deployment_timestamp"] = pd.to_datetime(self.deployments_df["deployment_timestamp"], utc=True)

    def perform_root_cause_analysis(self, alert_json, similar_alerts):
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

        prompt = f"""
You are an expert SRE.

Your task: Perform a **short and clear root cause analysis** for the given production alert using the provided context.

Strictly follow this format:
- Root Cause: <one-line reason>
- Diagnosis: <brief fix or recommendation>

DO NOT provide any additional explanation or narrative.

ALERT:
{json.dumps(alert_json, indent=2)}

SIMILAR HISTORICAL ALERTS:
{json.dumps(similar_alerts, indent=2)}

LOGS AROUND ALERT TIME:
{logs_slice.to_string(index=False)}

METRICS AROUND ALERT TIME:
{metrics_slice.to_string(index=False)}

PRODUCTION DEPLOYMENTS AROUND ALERT TIME:
{deployments_slice.to_string(index=False)}

Based on the above, identify the most likely root cause and suggest a high-level diagnosis.
"""

        chain = llm | StrOutputParser()
        result = chain.invoke(prompt)
        return result

# -------- LangGraph Nodes --------
def search_node(state):
    alert_json = state["alert"]
    search_agent = SearchAgent(embedding_model, faiss_index, df)
    similar_alerts = search_agent.search(alert_json)

    print("\n🔍 Similar Alerts Found:\n")
    for alert in similar_alerts:
        print(json.dumps(alert, indent=2))

    return {"alert": alert_json, "similar_alerts": similar_alerts}

def rca_node(state):
    alert_json = state["alert"]
    similar_alerts = state["similar_alerts"]

    rca_agent = RCAAgent()
    root_cause = rca_agent.perform_root_cause_analysis(alert_json, similar_alerts)
    return {"alert": alert_json, "similar_alerts": similar_alerts, "root_cause": root_cause}

# -------- Define LangGraph Workflow --------
class AgentState(TypedDict):
    alert: dict
    similar_alerts: List[dict]
    root_cause: str

graph_builder = StateGraph(AgentState)
graph_builder.add_node("SearchAgent", search_node)
graph_builder.add_node("RCAAgent", rca_node)

graph_builder.set_entry_point("SearchAgent")
graph_builder.add_edge("SearchAgent", "RCAAgent")
graph_builder.add_edge("RCAAgent", END)

app = graph_builder.compile()

# -------- Run the Workflow --------
final_result = app.invoke({"alert": alert_json})

print("\n🧠 Root Cause Analysis Result:\n")
print(final_result["root_cause"])
