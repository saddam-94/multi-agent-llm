import os
from .utils import save_to_pdf
from .data_ingestor import Data_Ingestor
from .mutli_agents import (CompetitorFinderAgent, CompetitorInfoExtractor, 
                          CompetitorProfileAgent, ReportGeneratorAgent)


class Core_Agents():

    def __init__(self, model_name, device='cuda'):
        
        self.competitor_finder = CompetitorFinderAgent(model_name, device)

        self.data_scraper = Data_Ingestor()

        self.extractor = CompetitorInfoExtractor(model_name, device)
        self.profileAgent = CompetitorProfileAgent(model_name, device)
        self.reportAgent = ReportGeneratorAgent(model_name, device)

    def multi_agent_processor(self, input_query):

        competitors = self.competitor_finder.find_competitors(input_query)
        print(f"Competitors for '{input_query}':")
        print(competitors)
        del self.competitor_finder

        competitor_info_list = self.extractor.extract_info(competitors, self.data_scraper, max_len = 20000)
        for competitor_info in competitor_info_list:
            print("\n", "=="*50, "\n", competitor_info, "\n\n")

        del self.extractor

        profiles = self.profileAgent.generate_profile(competitors, competitor_info_list)

        for profile in profiles:
            print(f"\nCompetitor: {profile['competitor_name']}")
            print(profile['profile'])
            print("="*50, "\n\n")

        report = self.reportAgent.generate_report(competitors, profiles)
        print(report)
        with open("competitor_analysis_report.txt", "w") as f:
            f.write(report)

        del profiles

        save_to_pdf(report, input_query=input_query)
        del self.reportAgent
        return report