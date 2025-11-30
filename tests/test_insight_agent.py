from agents.insight_agent import InsightAgent
from agents.data_agent import DataAgent

def test_insight_generation():
    df = DataAgent("data/sample_ads_data.csv").execute_etl()
    insights = InsightAgent(df).generate_insights()
    assert isinstance(insights, list)
