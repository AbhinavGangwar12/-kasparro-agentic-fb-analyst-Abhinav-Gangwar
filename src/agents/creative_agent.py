import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class CreativeAgent:
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

    def get_embedding(self, text: str):
        if not text:
            return np.zeros(768)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

    def analyze_copy_performance(self):
        significant = self.df[self.df['impressions'] > 1000]
        high = significant[significant['calculated_roas'] > 3.0]

        if high.empty:
            return []

        return high['creative_message'].value_counts().head(3).index.tolist()

    def generate_recommendation(self, campaign, themes):
        return {
            "campaign": campaign,
            "themes_detected": themes,
            "recommendation": (
                "Integrate comfort, clarity, and sensory benefit framing to increase CTR and ROAS."
            )
        }
