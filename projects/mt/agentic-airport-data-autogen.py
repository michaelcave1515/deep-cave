import asyncio
import json
from typing import Dict, List, Optional, Union

import pandas as pd
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.code_utils import execute_code
from autogen_agentchat.teams import GroupChat, Team
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaClient

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use Ollama models
MODEL_NAME = "llama3:latest"
ollama_client = OllamaClient(model=MODEL_NAME)

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

class SchemaAnalysisAgent(AssistantAgent):
    """Agent for analyzing the schema and data types of the input data."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            model_client=ollama_client,
            system_message="""You are a data analyst tasked with explaining the schema of a dataset.
            The dataset has the following columns and data types.
            Provide a concise description of the dataset, including:
            1. The overall purpose of the dataset.
            2. A brief explanation of each column and its meaning.
            3. Any potential relationships between columns.""",
            max_consecutive_auto_reply=4
        )

    async def analyze_schema(self, data: pd.DataFrame) -> str:
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

        response = await self.initiate_chat_async(
            recipient=self,
            message=schema_prompt
        )
        # Assuming the response contains the schema description in the last message
        return response.last_message.content if response.last_message else "Schema analysis failed."

class DataTypeAnalysisAgent(AssistantAgent):
    """Agent for analyzing data types and characteristics of the data."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            model_client=ollama_client,
            system_message="""You are a data analyst tasked with explaining the data types and characteristics of a dataset.
            Provide a concise analysis for each column.""",
            max_consecutive_auto_reply=4
        )

    async def analyze_data_types(self, data: pd.DataFrame) -> str:
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

        response = await self.initiate_chat_async(
            recipient=self,
            message=type_prompt
        )

        return response.last_message.content if response.last_message else "Data type analysis failed."

class FlightDelayAnalysisAgent(AssistantAgent):
    """Agent for analyzing flight delays."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            model_client=ollama_client,
            system_message="""You are an expert in analyzing flight data,
            specifically focusing on delays. Provide insights into the causes
            and patterns of flight delays based on the provided data.""",
            max_consecutive_auto_reply=4
        )

    async def analyze_delays(self, data: pd.DataFrame) -> str:
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
        response = await self.initiate_chat_async(
            recipient=self,
            message=analysis_prompt
        )
        return response.last_message.content if response.last_message else "Flight delay analysis failed."

class QueryUnderstandingAgent(AssistantAgent):
    """Agent that breaks down complex queries into smaller sub-queries."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            model_client=ollama_client,
            system_message="""You are an expert at breaking down complex data
            analysis questions into smaller, manageable sub-queries.""",
            max_consecutive_auto_reply=4
        )

    async def break_down_query(self, query: str) -> str:
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

        response = await self.initiate_chat_async(
            recipient=self,
            message=decomposition_prompt
        )

        try:
            sub_queries = json.loads(response.last_message.content)
            return json.dumps(sub_queries)
        except json.JSONDecodeError:
            print("Error: Could not parse sub-queries from LLM response.")
            return json.dumps([])

class NaturalLanguageQueryingAgent(AssistantAgent):
    """Agent for handling natural language queries about the data."""

    def __init__(self, name: str, schema_agent: SchemaAnalysisAgent, type_agent: DataTypeAnalysisAgent):
        super().__init__(
            name=name,
            model_client=ollama_client,
            system_message="""You are a data analyst who can translate natural language
            queries into Python code that operates on a pandas DataFrame.""",
            max_consecutive_auto_reply=4
        )
        self.schema_agent = schema_agent
        self.type_agent = type_agent

    async def process_query(self, query: str, data: pd.DataFrame) -> str:
        # Get schema and data type information from other agents
        schema_analysis_result = await self.schema_agent.analyze_schema(data)
        type_analysis_result = await self.type_agent.analyze_data_types(data)

        # Construct the prompt for the LLM
        query_prompt = f"""
        You are a data analyst who can translate natural language queries into Python code that operates on a pandas DataFrame.

        Here is the schema of the DataFrame:
        {schema_analysis_result}

        Here is information about the data types and characteristics of each column:
        {type_analysis_result}

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

        response = await self.initiate_chat_async(
            recipient=self,
            message=query_prompt
        )
        generated_code = response.last_message.content

        # Execute the generated code
        local_vars = {"data": data.copy(), "result": None}
        try:
            exec(generated_code, {}, local_vars)
            result = local_vars["result"]
            if isinstance(result, pd.DataFrame):
                result = result.to_string()
        except Exception as e:
            result = f"Error executing generated code: {e}"

        return result

# Define a UserProxyAgent
class UserAgent(UserProxyAgent):
    async def a_receive(self, message: Union[Dict, str], sender: ConversableAgent, request_reply: Optional[bool] = None, silent: Optional[bool] = False) -> Union[None, Dict, str]:
        return await super().a_receive(message, sender, request_reply, silent)
    
    async def a_send(self, message: Union[Dict, str], recipient: "ConversableAgent", request_reply: Optional[bool] = None, silent: Optional[bool] = False) -> None:
        if isinstance(message, dict) and message.get("content") == "TERMINATE":
            return
        
        user_input = input("User input (leave empty and press Enter to skip): ")
        
        if user_input:
            await super().a_send(message={"content": user_input, "role": "user"}, recipient=recipient, request_reply=True, silent=silent)
            return
        
        await super().a_send(message, recipient, request_reply, silent)

    def get_human_input(self, prompt: str) -> str:
        return ""

async def main():
    # Load data (replace with your actual file paths)
    log_table_path = "log-table.csv"
    fact_table_path = "fact-table.csv"
    data = load_data(log_table_path, fact_table_path)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Initialize Agents
    schema_agent = SchemaAnalysisAgent("Schema_Agent")
    type_agent = DataTypeAnalysisAgent("Data_Type_Agent")
    flight_delay_agent = FlightDelayAnalysisAgent("Flight_Delay_Agent")
    query_understanding_agent = QueryUnderstandingAgent("Query_Understanding_Agent")
    nl_query_agent = NaturalLanguageQueryingAgent("NL_Query_Agent", schema_agent, type_agent)
    user_proxy = UserAgent(
        "User_Proxy",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=1
    )

    # Define the team
    team = Team(
        agents=[
            user_proxy,
            schema_agent,
            type_agent,
            flight_delay_agent,
            query_understanding_agent,
            nl_query_agent,
        ],
    )

    # Initiate the chat
    await user_proxy.initiate_chat_async(
        team,
        message="What can you tell me about this data? Start by giving me the schema.",
        data=data,  # Pass the data to the agents
    )

    while True:
        try:
            next_message = input("Enter your query (or type 'exit' to quit): ")
            if next_message.lower() == "exit":
                break

            await user_proxy.send_async(
                recipient=team,
                message=next_message,
            )

            reply = await user_proxy.a_receive(sender=team)
            print(f"Response: {reply['content'] if reply else 'No response'}")
        except Exception as e:
            print(f"Error during interaction: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())