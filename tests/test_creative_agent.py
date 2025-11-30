from agents.creative_agent import CreativeAgent
from agents.data_agent import DataAgent

def test_creative_agent():
    df = DataAgent("data/sample_ads_data.csv").execute_etl()
    agent = CreativeAgent(df)
    themes = agent.analyze_copy_performance()
    assert isinstance(themes, list)
