# Install required libraries if not already installed
# !pip install langchain azure-identity requests pandas numpy

# Import necessary libraries
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import asyncio

# Import Azure and LangChain libraries
from azure.identity import DefaultAzureCredential
from azure.identity import AzureKeyCredential
import requests
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool



class LogFilterTool(BaseTool):
    name: str = "Time-Based Log Filter"
    description: str = "Filters log entries based on a specified time frame."
    
    log_file_path: str  # Declare as a field with a type annotation

    def _run(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[str]:
        start_datetime, end_datetime = self.get_time_frame(start_time, end_time)
        filtered_logs = self.filter_logs_by_time(start_datetime, end_datetime)
        return filtered_logs

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported for LogFilterTool.")

    def get_time_frame(self, start_time: Optional[str], end_time: Optional[str]) -> Tuple[datetime, datetime]:
        if end_time:
            end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        else:
            end_datetime = datetime.now()
        if start_time:
            start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        else:
            start_datetime = end_datetime - timedelta(days=1)
        return start_datetime, end_datetime

    def filter_logs_by_time(self, start_datetime: datetime, end_datetime: datetime) -> List[str]:
        filtered_logs = []
        timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

        with open(self.log_file_path, 'r') as file:
            for line in file:
                match = timestamp_pattern.match(line)
                if match:
                    log_time_str = match.group()
                    log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
                    if start_datetime <= log_time <= end_datetime:
                        filtered_logs.append(line)
                else:
                    if filtered_logs:
                        filtered_logs[-1] += line  # Append to the previous entry
        return filtered_logs





    # Define the path to your log file
log_file_path = 'system_log.txt'  # Replace with your actual log file path

# Initialize the tool with the log_file_path as a keyword argument
log_filter_tool = LogFilterTool(log_file_path=log_file_path)





class LogChunkerTool(BaseTool):
    name: str = "Log Chunker"
    description: str = "Chunks log entries into manageable sizes for processing."

    def _run(self, log_entries: List[str], chunk_size: int = 1000) -> List[List[str]]:
        chunks = self.chunk_logs(log_entries, chunk_size)
        return chunks

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported for LogChunkerTool.")

    def chunk_logs(self, log_entries: List[str], chunk_size: int) -> List[List[str]]:
        chunks = [log_entries[i:i + chunk_size] for i in range(0, len(log_entries), chunk_size)]
        return chunks




log_chunker_tool = LogChunkerTool()




class AzureOpenAILLM(LLM):
    def __init__(self, endpoint: str, api_key: str, deployment_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = '2023-05-15'  # Update to the correct API version if needed

    @property
    def _llm_type(self) -> str:
        return "azure-openai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 150) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/completions?api-version={self.api_version}"

        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": 1,
            "stop": stop,
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        completion = response.json()
        return completion['choices'][0]['text'].strip()







# Configure the Azure OpenAI LLM
# Replace the placeholders with your actual Azure OpenAI details
azure_endpoint = "https://<your-resource-name>.openai.azure.com"  # Your Azure OpenAI endpoint
azure_api_key = "<your-api-key>"  # Your Azure OpenAI API key
deployment_name = "<your-deployment-name>"  # Your model deployment name

llm = AzureOpenAILLM(
    endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=deployment_name
)




class LogSummarizerTool(BaseTool):
    name: str = "Log Summarizer"
    description: str = "Summarizes chunks of log entries."

    llm: AzureOpenAILLM  # Use the custom LLM class

    def _run(self, chunks: List[List[str]]) -> List[str]:
        summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"Summarizing chunk {idx + 1}/{len(chunks)}...")
            summary = self.summarize_chunk(chunk)
            summaries.append(summary)
        return summaries

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported for LogSummarizerTool.")

    def summarize_chunk(self, chunk: List[str]) -> str:
        chunk_text = ''.join(chunk)
        prompt = (
            "Summarize the following log entries, highlighting any errors, warnings, and unusual activities:\n\n"
            f"{chunk_text}\n\nSummary:"
        )
        try:
            response = self.llm._call(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error during summarization: {e}")
            return ""





log_summarizer_tool = LogSummarizerTool(llm=llm)






# List of tools the agent can use
tools = [
    log_filter_tool,
    log_chunker_tool,
    log_summarizer_tool,
    # You can add more tools as needed
]





# Initialize memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the agent
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)






def process_user_request(agent_chain, start_time: Optional[str] = None, end_time: Optional[str] = None, chunk_size: int = 1000):
    # Step 1: Filter logs
    filtered_logs = log_filter_tool._run(start_time=start_time, end_time=end_time)
    print(f"Number of log entries after filtering: {len(filtered_logs)}")

    # Step 2: Chunk logs
    log_chunks = log_chunker_tool._run(log_entries=filtered_logs, chunk_size=chunk_size)
    print(f"Total number of chunks: {len(log_chunks)}")

    # Step 3: Summarize chunks
    summaries = log_summarizer_tool._run(chunks=log_chunks)
    print(f"Generated {len(summaries)} summaries.")

    # Step 4: Present summaries or further process as needed
    return summaries





