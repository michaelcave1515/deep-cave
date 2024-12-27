import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.python import PythonTool
from phi.tools.base import Tool
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.openai import OpenAIEmbedder
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the list of tools for the agent
tools = [
    DuckDuckGo(),
    YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True,
    ),
    PythonTool(),
]

def load_data(log_table_path: str, fact_table_path: str) -> pd.DataFrame:
    """Loads data from the log and fact tables (CSV files), joins them,
    and performs basic preprocessing.

    Args:
        log_table_path: Path to the log table CSV file.
        fact_table_path: Path to the fact table CSV file.
    """
    try:
        # Load the CSV files into pandas DataFrames
        log_df = pd.read_csv(log_table_path)
        fact_df = pd.read_csv(fact_table_path)

        # Data Cleaning (example):
        # Convert relevant columns to lowercase for consistent joining
        log_df.columns = log_df.columns.str.lower()
        fact_df.columns = fact_df.columns.str.lower()
        # Convert date columns to datetime objects
        date_columns = [col for col in log_df.columns if col.endswith(("_dt", "_dttm"))]
        for col in date_columns:
            log_df[col] = pd.to_datetime(log_df[col], errors="coerce")

        date_columns = [col for col in fact_df.columns if col.endswith(("_dt", "_dttm"))]
        for col in date_columns:
            fact_df[col] = pd.to_datetime(fact_df[col], errors="coerce")

        # Identify Common Columns for Joining:
        common_columns = list(set(log_df.columns).intersection(fact_df.columns))

        # Perform the join operation using the common columns
        merged_df = pd.merge(log_df, fact_df, on=common_columns, how="inner")

        # Ensure there are no duplicate columns after the join
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        return merged_df

    except Exception as e:
        logger.error(f"Error loading or merging data: {str(e)}")
        return None

# Define Agents
class SchemaAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Schema Analyzer",
            model=Ollama(model="llama3:latest"),
            instructions="""You are a data analyst tasked with explaining the schema of a dataset.
            The dataset has the following columns and data types.
            Provide a concise description of the dataset, including:
            1. The overall purpose of the dataset.
            2. A brief explanation of each column and its meaning.
            3. Any potential relationships between columns.""",
            show_tool_calls=True,
            markdown=True,
        )

    async def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        schema_info = {}

        # Add column names and data types
        schema_info["columns"] = {col: str(data[col].dtype) for col in data.columns}

        # Basic descriptive statistics
        schema_info["stats"] = (
            data.describe(include="all", datetime_is_numeric=True).to_dict()
        )

        # Generate LLM analysis for schema description
        schema_prompt = f"""
        You are a data analyst tasked with explaining the schema of a dataset.
        The dataset has the following columns and data types:

        {json.dumps(schema_info['columns'], indent=2)}

        Provide a concise description of the dataset, including:
        1. The overall purpose of the dataset.
        2. A brief explanation of each column and its meaning.
        3. Any potential relationships between columns.

        Additionally, here are some basic statistics about the columns:
        {json.dumps(schema_info['stats'], indent=2)}

        Use these statistics to further explain the data in each column.
        Be clear and concise, using natural language that a non-technical user can understand.
        """

        llm_schema_description = await self.generate_analysis(schema_prompt)

        return {
            "schema": schema_info,
            "schema_description": llm_schema_description,
        }

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        messages = [{"role": "system", "content": prompt}]
        response = self.model.invoke(messages)
        return response

class DataTypeAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Data Type Analyzer",
            model=Ollama(model="llama3:latest"),
            instructions="""You are a data analyst tasked with explaining the data types and characteristics of a dataset.
            Provide a concise analysis for each column.""",
            show_tool_calls=True,
            markdown=True,
        )

    async def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        type_info = {}

        for col in data.columns:
            type_info[col] = {}

            # Data type
            type_info[col]["dtype"] = str(data[col].dtype)

            # Uniqueness
            type_info[col]["unique_values"] = data[col].nunique()
            type_info[col]["is_unique"] = data[col].is_unique

            # Missing values
            type_info[col]["missing_percentage"] = (
                (data[col].isnull().sum() / len(data)) * 100
            )

            # Value ranges (numerical)
            if pd.api.types.is_numeric_dtype(data[col]):
                type_info[col]["min"] = data[col].min()
                type_info[col]["max"] = data[col].max()
                type_info[col]["mean"] = data[col].mean()
                type_info[col]["median"] = data[col].median()
            # Value counts (categorical)
            elif pd.api.types.is_string_dtype(data[col]):
                type_info[col]["value_counts"] = (
                    data[col].value_counts().head(10).to_dict()
                )  # Top 10

        # Generate LLM analysis for data type description
        type_prompt = f"""
        You are a data analyst tasked with explaining the data types and characteristics of a dataset.

        Here is information about each column:
        {json.dumps(type_info, indent=2)}

        Provide a concise analysis for each column, including:
        1. An interpretation of its data type.
        2. Whether the column could be a unique identifier.
        3. The distribution of values (ranges for numerical, common values for categorical).
        4. The percentage of missing values and potential implications.

        Use natural language and be clear and concise.
        """

        llm_type_description = await self.generate_analysis(type_prompt)

        return {
            "data_types": type_info,
            "type_description": llm_type_description,
        }

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        messages = [{"role": "system", "content": prompt}]
        response = self.model.invoke(messages)
        return response

class FlightDelayAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Flight Delay Analyzer",
            model=Ollama(model="llama3:latest"),
            instructions="""You are an expert in analyzing flight data,
            specifically focusing on delays. Provide insights into the causes
            and patterns of flight delays based on the provided data.""",
            show_tool_calls=True,
            markdown=True,
        )

    async def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Perform analysis related to flight delays
        if "arvl_dlay_cat_cd" in data.columns:
            avg_arrival_delay = data["arvl_dlay_cat_cd"].mean()
        else:
            avg_arrival_delay = "Column 'arvl_dlay_cat_cd' not found in data"

        # Generate LLM analysis
        analysis_prompt = f"""
        Analyze the following flight delay statistics:
        - Average Arrival Delay: {avg_arrival_delay}
        - ... (other delay-related statistics) ...

        Provide insights into the causes and patterns of flight delays.
        """
        llm_analysis = await self.generate_analysis(analysis_prompt)

        return {
            "avg_arrival_delay": avg_arrival_delay,
            "llm_analysis": llm_analysis,
        }

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        messages = [{"role": "system", "content": prompt}]
        response = self.model.invoke(messages)
        return response

class QueryUnderstandingAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Query Understanding Agent",
            model=Ollama(model="llama3:latest"),
            instructions="""You are an expert at breaking down complex data
            analysis questions into smaller, manageable sub-queries.""",
            show_tool_calls=True,
            markdown=True,
        )

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        # Use the LLM to decompose the query
        decomposition_prompt = f"""
        You are an expert at breaking down complex data analysis questions into smaller, manageable sub-queries.

        The user has asked the following complex question: "{query}"

        Break this question down into a series of simpler questions that can be answered individually using a dataset.
        Output these sub-questions as a JSON list, like so:

        ```json
        [
            "Sub-question 1",
            "Sub-question 2",
            "Sub-question 3"
        ]
        ```
        """

        llm_response = await self.generate_analysis(decomposition_prompt)

        try:
            sub_queries = json.loads(llm_response)
        except json.JSONDecodeError:
            sub_queries = []
            print("Error: Could not parse sub-queries from LLM response.")

        return {
            "original_query": query,
            "sub_queries": sub_queries,
        }

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        messages = [{"role": "system", "content": prompt}]
        response = self.model.invoke(messages)
        return response

class NaturalLanguageQueryingAgent(Agent):
    def __init__(self, schema_agent: Agent, type_agent: Agent):
        super().__init__(
            name="Natural Language Querying Agent",
            model=Ollama(model="llama3:latest"),
            instructions="""You are a data analyst who can translate natural language
            queries into Python code that operates on a pandas DataFrame.""",
            show_tool_calls=True,
            markdown=True,
        )
        self.schema_agent = schema_agent
        self.type_agent = type_agent

    async def run(self, query: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Get schema and data type information from other agents
        schema_results = await self.schema_agent.run(data=data)
        type_results = await self.type_agent.run(data=data)

        schema_info = schema_results["schema"]
        schema_description = schema_results["schema_description"]
        type_info = type_results["data_types"]
        type_description = type_results["type_description"]

        # Construct the prompt for the LLM
        query_prompt = f"""
        You are a data analyst who can translate natural language queries into Python code that operates on a pandas DataFrame.

        Here is the schema of the DataFrame:
        {json.dumps(schema_info, indent=2)}

        Schema Description:
        {schema_description}

        Here is information about the data types and characteristics of each column:
        {json.dumps(type_info, indent=2)}

        Data Type Description:
        {type_description}

        The user wants to know: "{query}"

        Generate Python code using the pandas library to answer this query.
        Assume the DataFrame is named 'data'.

        Example:
        If the user asks 'What is the average of the 'avg_wait_tm_nbr' column?', you should generate:
        ```python
        average_wait_time = data['avg_wait_tm_nbr'].mean()
        result = f"The average wait time is: {{average_wait_time}}"
        ```

        If the user asks: 'Which rows have more than average avg_wait_tm_nbr?' you should generate:
        ```python
        average_wait_time = data['avg_wait_tm_nbr'].mean()
        filtered_data = data[data['avg_wait_tm_nbr'] > average_wait_time]
        result = filtered_data
        ```

        IMPORTANT:
        - Only generate code that is safe to execute. Do not use any potentially harmful functions.
        - Structure the result in a clear and presentable format.
        - If the query involves aggregation, output the result in natural language.
        - If the query involves filtering or selecting data, output the result as a DataFrame.

        Now, generate the Python code to answer the user's query:
        """

        # Use LLM to generate code
        messages = [{"role": "system", "content": query_prompt}, {"role": "user", "content": query}]
        response = self.model.invoke(messages)
        generated_code = response

        # Execute the generated code
        local_vars = {"data": data.copy(), "result": None}
        try:
            exec(generated_code, {}, local_vars)
            result = local_vars["result"]
            if isinstance(result, pd.DataFrame):
                result = result.to_string()
        except Exception as e:
            result = f"Error executing generated code: {e}"

        return {
            "query": query,
            "generated_code": generated_code,
            "result": result,
        }

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        messages = [{"role": "system", "content": prompt}]
        response = self.model.invoke(messages)
        return response

# Define a General Inquiry Agent
general_inquiry_agent = Agent(
    name="General Inquiry Agent",
    model=Ollama(model="llama3:latest"),
    tools=tools,
    instructions=[
        "You are a helpful assistant that can answer general questions about flight data.",
        "If the query is complex, use the Query Understanding Agent.",
        "If the query is about flight delays, use the Flight Delay Analysis Agent.",
    ],
    show_tool_calls=True,
    markdown=True,
)

async def main():
    # Load data (replace with your actual file paths)
    log_table_path = "log-table.csv"
    fact_table_path = "fact-table.csv"
    data = load_data(log_table_path, fact_table_path)

    # Initialize Agents
    schema_agent = SchemaAnalysisAgent()
    type_agent = DataTypeAnalysisAgent()
    flight_delay_agent = FlightDelayAnalysisAgent()
    query_understanding_agent = QueryUnderstandingAgent()
    nl_query_agent = NaturalLanguageQueryingAgent(schema_agent, type_agent)

    # Create a list of agents for the playground
    agents = [
        general_inquiry_agent,
        schema_agent,
        type_agent,
        flight_delay_agent,
        query_understanding_agent,
        nl_query_agent,
    ]

    # Create the playground app
    playground = Playground(agents=agents)
    app = playground.get_app()

    # Serve the app
    serve_playground_app("playground:app", reload=True)

if __name__ == "__main__":
    asyncio.run(main())