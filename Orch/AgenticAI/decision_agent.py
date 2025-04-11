import re
from typing import List, Dict

class DecisionAgent:
    def __init__(self):
        pass  


    def extract_confidence_scores(self, recommendation_text: str) -> List[int]:
        # First check for LLM's explicit score 
        llm_score = re.findall(r"Confidence Score:\s*(\d+\.\d+)", recommendation_text)
        if llm_score:
            return [int(float(llm_score[0]) * 100)] 
        
        # Fallback to historical similarity scores
        return [int(match) for match in re.findall(r"Confidence:\s*(\d+)%", recommendation_text)]

    def decide_action(self, recommendation_text: str) -> Dict:
        try:
            scores = self.extract_confidence_scores(recommendation_text)
            if not scores:
                return {"decision": "manual_review", "reason": "No confidence score found."}

            average_conf = sum(scores) / len(scores)

            if average_conf >= 65:
                decision = "auto_remediate"
            elif average_conf < 50:
                decision = "escalate"
            else:
                decision = "verify_with_human"

            return {
                "decision": decision,
                "average_confidence": average_conf,
                "recommendation_summary": recommendation_text
            }
        
        except Exception as e:
            return {"error": str(e)}
