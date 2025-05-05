import json
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class RecommendationAgent:
    def __init__(self):
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
        self.chain = llm | StrOutputParser()
    def generate_suggestions(self, alert_json, rca_summary, similar_alerts):
        try:
            if similar_alerts:
                similarities = [alert["similarity_score"] for alert in similar_alerts]
                avg_similarity = sum(similarities) / len(similarities)
                confidence_score = int(avg_similarity * 100)
            else:
                confidence_score = 40

            # If a very high similarity found, use its resolution directly
            top_match = max(similar_alerts, key=lambda x: x["similarity_score"]) if similar_alerts else None
            if top_match and top_match["similarity_score"] > 0.90:
                resolution = top_match["resolution_steps"]
                root_cause = top_match.get("root_cause", "Not specified")
                return (
                    f"**Immediate Recommendation Based on Historical Match**\n\n"
                    f"Root Cause: {root_cause}\n"
                    f"Recommended Fix: {resolution}\n\n"
                    f"Confidence: {int(top_match['similarity_score'] * 100)}%"
                )

            prompt = f"""
            You are an AI SRE assistant.

            Given:
            - ALERT: {json.dumps(alert_json, indent=2)}
            - RCA: {rca_summary}
            - SIMILAR ALERTS: {json.dumps(similar_alerts, indent=2)}

            Generate a short and precise resolution plan with:
            1. Root Cause
            2. Fix (as a list of steps if needed)
            3. Confidence score

            Be concise. No long explanations.

            Response:
            """

            recommendation = self.chain.invoke(prompt)
            
            # Determine final confidence score
            if top_match and top_match["similarity_score"] > 0.9:
                confidence_score = int(top_match["similarity_score"] * 100)
            else:
                confidence_score = (
                    int(avg_similarity * 100) if similar_alerts else 65
                )
            
            # Ensure recommendation ends with correct confidence
            recommendation += f"\n\nConfidence: {confidence_score}%"
            return recommendation

        except Exception as e:
            return f"Error generating remediation: {str(e)}"


