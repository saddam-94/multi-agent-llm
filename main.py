import os
from agents.core_agents import Core_Agents
from agents.utils import parse_args


class Main():

    def __init__(self, args):
        # self.model_name = "Qwen/Qwen2.5-0.5B-Instruct",
        self.core_agent = Core_Agents(args.model_name, device=args.device)

    def main(self, query: str):
        agentic_analysis = self.core_agent.multi_agent_processor(query)
        print(agentic_analysis)


if '__name__' == '__main__':
    args = parse_args()
    main = Main(args)
    print(main.main(args.query))
