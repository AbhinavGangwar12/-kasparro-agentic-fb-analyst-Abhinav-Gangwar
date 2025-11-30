class InsightAgent:
    def __init__(self, df):
        self.df = df

    def generate_insights(self):
        insights = []

        platform_perf = self.df.groupby('platform')['calculated_roas'].mean()
        best_platform = platform_perf.idxmax()

        insights.append(
            f"Best platform: {best_platform} with ROAS {platform_perf.max():.2f}."
        )

        biggest = self.df.groupby('normalized_campaign')['spend'].sum().idxmax()
        camp = self.df[self.df['normalized_campaign'] == biggest].sort_values('date')

        half = len(camp) // 2
        if half > 0:
            first = camp.iloc[:half]['calculated_roas'].mean()
            second = camp.iloc[half:]['calculated_roas'].mean()

            if second < first * 0.8:
                insights.append(
                    f"Fatigue detected in {biggest}: ROAS dropped {first:.2f} → {second:.2f}."
                )

        return insights
