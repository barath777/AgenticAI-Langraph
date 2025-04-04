import langgraph as lg
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

# Define the Groq API Key directly
GROQ_API_KEY = "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"  # Replace with your actual API key

# Load historical alerts dataset
df = pd.read_csv("Data/Historical_Alerts_Resolutions.csv").fillna("")
df["alert_timestamp"] = df["alert_timestamp"].astype(str)
df["text"] = (
    df["alert_timestamp"] + " " + df["alert_type"] + " " + df["resolution_steps"] + " " 
    + df["application"] + " " + df["severity"] + " " 
    + df["root_cause"] + " " + df["change_implemented"] + " " + df["post_resolution_status"]
)

# Create embeddings using SentenceTransformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Define Search Agent
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
        
        return similar_alerts if similar_alerts else None

# Define RCA Agent (Root Cause Analysis)
class RCAAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def fetch_logs(self):
        """Mock function to fetch logs from the past 2 days."""
        return [
            "Log1: CPU spike detected on server X.",
            "Log2: High memory usage on service Y.",
            "Log3: Frequent database connection resets."
        ]

    def analyze_root_cause(self, alert_json, similar_alerts):
        logs = self.fetch_logs()
        prompt = f"""
        A new alert has been received:
        {json.dumps(alert_json, indent=2)}

        The following past alerts are found to be similar:
        {json.dumps(similar_alerts, indent=2)}

        The following logs from the past 2 days have been collected:
        {json.dumps(logs, indent=2)}

        Identify the root cause based on:
        - Common alert patterns
        - System logs
        - Application logs
        - Past resolutions

        Provide a structured root cause analysis.
        """

        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content

# Define Analysis Agent
class AnalysisAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate_analysis(self, rca_output):
        prompt = f"""
        Given the following Root Cause Analysis output:
        {rca_output}

        Provide a structured analysis with:
        - Risk assessment
        - Suggested actions
        - Possible preventive measures
        """

        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content

# Define Decision Agent
class DecisionAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def decide_remediation(self, analysis_output):
        prompt = f"""
        Based on the following analysis:
        {analysis_output}

        Decide if the remediation should be:
        - **Automated** (if resolution steps are well-defined and repeatable)
        - **Manual** (if it requires human intervention, business decisions, or unknown variables)

        Respond with only one sentence in this format:
        "The remediation should be automated." 
        OR 
        "The remediation should be manual."
        """

        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20  # Limit to a single sentence
        )

        return response.choices[0].message.content.strip()

# Initialize Agents
search_agent = SearchAgent(model, index, df)
rca_agent = RCAAgent()
analysis_agent = AnalysisAgent()
decision_agent = DecisionAgent()

# Run the workflow
def alert_analysis_workflow(alert_json: dict):
    print("\n[Search Agent Running...]\n")
    similar_alerts = search_agent.search(alert_json, threshold=0.3)

    if similar_alerts:
        print("\n[Search Agent Output]\n")
        print(json.dumps(similar_alerts, indent=2))

        print("\n[RCA Agent Running...]\n")
        rca_result = rca_agent.analyze_root_cause(alert_json, similar_alerts)
        print("\n[RCA Agent Output]\n")
        print(rca_result)

        print("\n[Analysis Agent Running...]\n")
        analysis_result = analysis_agent.generate_analysis(rca_result)
        print("\n[Analysis Agent Output]\n")
        print(analysis_result)

        print("\n[Decision Agent Running...]\n")
        decision_result = decision_agent.decide_remediation(analysis_result)
        print("\n[Decision Agent Output]\n")
        print(decision_result)

        return {
            "search_results": similar_alerts,
            "root_cause_analysis": rca_result,
            "analysis": analysis_result,
            "decision": decision_result
        }
    else:
        print("\n[No similar alert found. Sending to orchestration layer.]\n")
        return {"message": "No similar alert found. Sending to orchestration layer."}

# Load an alert JSON and run the workflow
with open("Alerts/ALERT002.json", "r") as f:
    alert_json = json.load(f)

result = alert_analysis_workflow(alert_json)
