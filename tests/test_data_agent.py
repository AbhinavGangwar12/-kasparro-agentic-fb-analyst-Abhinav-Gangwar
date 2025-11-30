import pandas as pd
from agents.data_agent import DataAgent

def test_data_agent_load():
    agent = DataAgent("data/sample_ads_data.csv")
    df = agent.execute_etl()
    assert isinstance(df, pd.DataFrame)
    assert "normalized_campaign" in df.columns
