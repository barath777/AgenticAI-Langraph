from typing import List
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class SearchAgent:
    def __init__(self, faiss_index_path="storage/faiss_index.bin", processed_data_path="storage/processed_alerts.csv"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(faiss_index_path)
        self.df = pd.read_csv(processed_data_path)
    
    # Convert alert JSON to embedding vector
    def embed_alert(self, alert_json: dict) -> np.ndarray:
        alert_text = (
            f"{alert_json.get('timestamp', '')} "
            f"{alert_json.get('description', '')} "
            f"{alert_json.get('alert_type', '')} "
            f"{alert_json.get('application', '')} "
            f"{' '.join(alert_json.get('affected_services', []))} "
            f"{alert_json.get('severity', '')}"
        )
        embedding = self.model.encode([alert_text])
        return embedding / np.linalg.norm(embedding) 
    
    #Find similar historical alerts
    def search(self, alert_json: dict, threshold: float = 0.3) -> List[dict]:
        try:
            alert_embedding = self.embed_alert(alert_json)
            distances, indices = self.index.search(alert_embedding, k=3) 
            
            results = []
            for i in range(len(indices[0])):
                if distances[0][i] > threshold:
                    match = self.df.iloc[indices[0][i]]
                    results.append({
                        "alert_id": match["alert_id"],
                        "alert_type": match["alert_type"],
                        "severity": match["severity"],
                        "root_cause": match["root_cause"],
                        "resolution_steps": match["resolution_steps"],
                        "similarity_score": float(distances[0][i])
                    })
            # Only keep matches with the same alert type
            results = [r for r in results if r["alert_type"] == alert_json.get("alert_type")]
            return results if results else None
        
        except Exception as e:
            return f"Search failed: {str(e)}"

