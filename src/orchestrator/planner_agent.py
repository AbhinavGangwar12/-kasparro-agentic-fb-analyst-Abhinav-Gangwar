from agents.data_agent import DataAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.insight_agent import InsightAgent
from agents.creative_agent import CreativeAgent


class PlannerAgent:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def run(self):
        print("\n--- KASPARRO AGENTIC SYSTEM ---\n")

        data_agent = DataAgent(self.csv_path)
        df = data_agent.execute_etl()

        evaluator = EvaluatorAgent(df)
        evaluator.build_and_train_model()
        anomalies = evaluator.detect_anomalies()

        insights = InsightAgent(df).generate_insights()
        creative_agent = CreativeAgent(df)
        themes = creative_agent.analyze_copy_performance()

        if not anomalies.empty:
            worst = anomalies.iloc[0]['normalized_campaign']
            creative = creative_agent.generate_recommendation(worst, themes)
        else:
            creative = "No anomalies detected."

        return {
            "insights": insights,
            "anomalies": anomalies.head(5),
            "creative_strategy": creative
        }
