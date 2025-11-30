from orchestrator.planner_agent import PlannerAgent

if __name__ == "__main__":
    system = PlannerAgent("data/sample_ads_data.csv")
    results = system.run()
    print(results)
