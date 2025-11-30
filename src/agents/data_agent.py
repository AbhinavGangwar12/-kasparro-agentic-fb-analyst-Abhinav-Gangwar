import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler


class DataAgent:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.scaler = StandardScaler()

    def normalize_campaign_name(self, name: str) -> str:
        if pd.isna(name):
            return "Unknown Campaign"

        clean = re.sub(r'[^a-z0-9\s]', '', str(name).lower())

        if 'men' in clean and ('comfort' in clean or 'comf' in clean):
            return 'Men ComfortMax Launch'
        if 'women' in clean and ('seamless' in clean or 'seam' in clean):
            return 'Women Seamless Everyday'
        if 'women' in clean and 'summer' in clean:
            return 'Women Summer Invisible'
        if 'men' in clean and 'athleisure' in clean:
            return 'Men Athleisure Cooling'

        return clean.title()

    def execute_etl(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)

        self.df['normalized_campaign'] = self.df['campaign_name'].apply(self.normalize_campaign_name)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        numeric_cols = ['spend', 'impressions', 'clicks', 'purchases', 'revenue', 'roas']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore').fillna(0)

        self.df['calculated_roas'] = np.where(
            self.df['spend'] > 0, self.df['revenue'] / self.df['spend'], 0
        )
        self.df['calculated_ctr'] = np.where(
            self.df['impressions'] > 0, self.df['clicks'] / self.df['impressions'], 0
        )

        self.df['creative_type_code'] = self.df['creative_type'].astype('category').cat.codes
        self.df['platform_code'] = self.df['platform'].astype('category').cat.codes
        self.df['audience_code'] = self.df['audience_type'].astype('category').cat.codes

        return self.df
