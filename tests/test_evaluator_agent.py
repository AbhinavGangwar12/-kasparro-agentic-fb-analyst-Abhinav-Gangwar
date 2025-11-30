from agents.evaluator_agent import EvaluatorAgent
from agents.data_agent import DataAgent

def test_evaluator_training():
    df = DataAgent("data/sample_ads_data.csv").execute_etl()
    evaluator = EvaluatorAgent(df)
    evaluator.build_and_train_model()
    assert evaluator.model is not None
