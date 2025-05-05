import subprocess

class RemediationAgent:
    def __init__(self):
        self.description_to_bat_file = {
            "login failed due to network issue": r"C:\Users\ASMrAbhilash(Abhilas\Orch\AgenticAI\start-apps.bat",
            "server error during login: connect econnrefused ::1:27017, connect econnrefused 127.0.0.1:27017": r"C:\Windows\System32\Mongodb.bat"
        }

    def execute_remediation(self, decision_output: dict, alert_json: dict, similar_alerts: list):
        if decision_output.get("decision") != "auto_remediate":
            return "Remediation skipped: Not an auto_remediate decision."

        if not similar_alerts:
            return "Remediation skipped: No similar alerts found."

        for alert in similar_alerts:
            similarity_score = alert.get("similarity_score", 0)
            historical_app = alert.get("application", "").strip().lower()
            current_app = alert_json.get("application", "").strip().lower()
            alert_desc = alert_json.get("description", "").strip().lower()

            if similarity_score >= 0.75 and historical_app == current_app:
                bat_file = self.description_to_bat_file.get(alert_desc)

                if not bat_file:
                    return f"Remediation skipped: No bat file mapped for description: {alert_desc}"

                try:
                    if "Mongodb.bat" in bat_file:
                        # Admin mode with UAC prompt
                        subprocess.run(
                            f'powershell Start-Process -FilePath "{bat_file}" -Verb RunAs',
                            shell=True,
                            check=True
                        )
                    else:
                        # Normal execution without admin
                        subprocess.run(
                            f'"{bat_file}"',
                            shell=True,
                            check=True
                        )

                    return "Remediation executed successfully."
                except subprocess.CalledProcessError:
                    return "Remediation failed: Script returned error."
                except Exception:
                    return "Remediation failed: Exception occurred."

        return "Remediation skipped: No matching historical alert met the criteria."
