from vllm import LLM, SamplingParams
import torch


class CompetitorFinderAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, max_model_len=2048, max_tokens=512, temperature=0.7, top_p=0.9, device="cuda"):
        torch.cuda.empty_cache()
        self.llm = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=max_model_len,
            device=device
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def find_competitors(self, input_query, N=3):
        prompt = (
            f"Identify {N} major competitors for the following product, service, or industry: \"{input_query}\". "
            "Provide only the names of the competitors in a numbered list."
        )

        messages = [
            {"role": "system", "content": "You are a market research expert."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        response = self.llm.generate([text], sampling_params=self.sampling_params)
        raw_output = response[0].outputs[0].text

        print("\n\nModel response:")
        print(raw_output)
        print("="*50, "\n\n")
        competitors = [line.split(". ", 1)[1].strip() for line in raw_output.splitlines() if line[0].isdigit() and ". " in line]
        
        return competitors


class CompetitorInfoExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, max_model_len=32768, max_tokens=16384, temperature=0.7, top_p=0.9, device="cuda"):
        torch.cuda.empty_cache()
        self.llm = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=max_model_len,
            device=device
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def extract_info(self, competitors, texts, max_len=20000):
        prompts = [
            f"Extract relevant information about {competitor}, from the provided text. "
            "Remove unnecessary data and handle any conflicts by choosing the most reliable and relevant details. "
            "Ensure high data accuracy.\n\n"
            f"Text:\n{text[:max_len]}\n\n"
            for competitor, text in zip(competitors, texts)
        ]

        messages_batch = [
            [{"role": "system", "content": "You are a helpful assistant capable of analyzing data and extracting relevant information. "
                                          "Extract information from the provided text, removing any unnecessary or irrelevant details. "
                                          "Handle data conflicts by selecting the most reliable and accurate details."},
            {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        texts_batch = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]

        responses = self.llm.generate(texts_batch, sampling_params=self.sampling_params)

        extracted_info = []
        for i, response in enumerate(responses):
            response_text = response.outputs[0].text
            extracted_info.append(response_text)

        return extracted_info


class CompetitorProfileAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, max_model_len=32768, max_tokens=16384, temperature=0.7, top_p=0.9, device="cuda"):
        torch.cuda.empty_cache()
        self.llm = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=max_model_len,
            device=device
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def generate_profile(self, competitors, competitor_info_list):
        profiles = []

        prompts = [
            f"Analyze the following information about {competitor_name} and create a structured profile. "
            "Include an overview, SWOT analysis (Strengths, Weaknesses, Opportunities, Threats), and actionable insights.\n\n"
            f"Competitor Information:\n{competitor_info}"
            for competitor_name, competitor_info in zip(competitors, competitor_info_list)
        ]

        messages_batch = [
            [{"role": "system", "content": "You are a helpful assistant capable of analyzing and structuring data."},
             {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        texts_batch = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]

        responses = self.llm.generate(texts_batch, sampling_params=self.sampling_params)

        for i, response in enumerate(responses):
            response_text = response.outputs[0].text
            profiles.append({
                "competitor_name": competitors[i],
                "profile": response_text
            })

        return profiles


class ReportGeneratorAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, max_model_len=32768, max_tokens=32768, temperature=0.7, top_p=0.9, device="cuda"):
        torch.cuda.empty_cache()

        self.llm = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=max_model_len,
            device=device
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def generate_report(self, competitors, profiles):

        prompt = (
            f"Based on the following competitor profiles, generate a detailed competitor analysis report.\n"
            f"Include an introduction, an overview of each competitor, feature comparisons, and strategic recommendations.\n"
            "The report should be structured as follows:\n"
            "1. Introduction\n"
            "2. Competitor Overview\n"
            "3. Feature Comparisons\n"
            "4. Strategic Recommendations\n\n"
            "Competitor Profiles:\n"
        )

        for competitor, profile in zip(competitors, profiles):
            prompt += f"\n{competitor}: {profile}"

        messages = [
            {"role": "system", "content": "You are an expert in competitor analysis and report generation."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = self.llm.generate([text], sampling_params=self.sampling_params)

        report = response[0].outputs[0].text

        return report
