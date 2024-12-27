import os
from typing import List

import pandas as pd
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, format_tool_to_openai_function
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install necessary packages
# pip install crewai langchain-community

# You can use a local model with Ollama: https://docs.crewai.com/how-to/LLM-Connections/

# Set your API keys in the environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Define tools for your agents
@tool
def analyze_flight_delays(data: str) -> str:
    """Analyzes the provided flight data for delays."""
    # Convert the string back to DataFrame
    data = pd.read_json(data)
    # Perform analysis related to flight delays
    if "arvl_dlay_cat_cd" in data.columns:
        avg_arrival_delay = data["arvl_dlay_cat_cd"].mean()
        analysis = f"Average Arrival Delay: {avg_arrival_delay}"
    else:
        analysis = "Column 'arvl_dlay_cat_cd' not found in data"

    return analysis

@tool
def understand_query(query: str) -> str:
    """Breaks down a complex query into smaller sub-queries."""
    
    llm = Ollama(model="llama3:latest")
    
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
    response = llm.invoke(decomposition_prompt)

    try:
        sub_queries = json.loads(response)
        return json.dumps(sub_queries)  # Convert list to JSON string
    except json.JSONDecodeError:
        print("Error: Could not parse sub-queries from LLM response.")
        return json.dumps([])

@tool
def analyze_schema(data: str) -> str:
    """Analyzes the schema of the provided data."""
    data = pd.read_json(data)
    schema_info = {}

    # Add column names and data types
    schema_info["columns"] = {col: str(data[col].dtype) for col in data.columns}

    # Basic descriptive statistics
    schema_info["stats"] = data.describe(include="all", datetime_is_numeric=True).to_dict()

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
    llm = Ollama(model="llama3:latest")
    llm_schema_description = llm.invoke(schema_prompt)

    return llm_schema_description

@tool
def analyze_data_types(data: str) -> str:
    """Analyzes the data types and characteristics of the data."""
    data = pd.read_json(data)
    type_info = {}

    for col in data.columns:
        type_info[col] = {}

        # Data type
        type_info[col]["dtype"] = str(data[col].dtype)

        # Uniqueness
        type_info[col]["unique_values"] = data[col].nunique()
        type_info[col]["is_unique"] = data[col].is_unique

        # Missing values
        type_info[col]["missing_percentage"] = (data[col].isnull().sum() / len(data)) * 100

        # Value ranges (numerical)
        if pd.api.types.is_numeric_dtype(data[col]):
            type_info[col]["min"] = data[col].min()
            type_info[col]["max"] = data[col].max()
            type_info[col]["mean"] = data[col].mean()
            type_info[col]["median"] = data[col].median()
        # Value counts (categorical)
        elif pd.api.types.is_string_dtype(data[col]):
            type_info[col]["value_counts"] = data[col].value_counts().head(10).to_dict()  # Top 10
    
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

    llm = Ollama(model="llama3:latest")
    llm_type_description = llm.invoke(type_prompt)

    return llm_type_description

@tool
def process_natural_language_query(data: str, query: str) -> str:
    """Processes natural language queries by translating them into executable code."""
    data = pd.read_json(data)

    # Placeholder for schema and type descriptions
    schema_description = "Schema description not available."
    type_description = "Type description not available."

    query_prompt = f"""
    You are a data analyst who can translate natural language queries into Python code that operates on a pandas DataFrame.

    Here is the schema of the DataFrame:
    {{schema_description}}

    Here is information about the data types and characteristics of each column:
    {{type_description}}

    The user wants to know: "{query}"

    Generate Python code using the pandas library to answer this query.
    Assume the DataFrame is named 'data'.

    IMPORTANT:
    - Only generate code that is safe to execute. Do not use any potentially harmful functions.
    - Structure the result in a clear and presentable format.
    - If the query involves aggregation, output the result in natural language.
    - If the query involves filtering or selecting data, output the result as a DataFrame.

    Now, generate the Python code to answer the user's query:
    """

    llm = Ollama(model="llama3:latest")
    messages = [{"role": "system", "content": query_prompt}, {"role": "user", "content": query}]
    response = llm.invoke(messages)
    generated_code = response

    local_vars = {"data": data.copy(), "result": None}
    try:
        exec(generated_code, {}, local_vars)
        result = local_vars["result"]
        if isinstance(result, pd.DataFrame):
            result = result.to_string()
    except Exception as e:
        result = f"Error executing generated code: {e}"

    return result

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

# Define your agents with roles and goals
class FlightDataAgent(Agent):
    def __init__(self, role, goal, backstory, tools):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            tools=tools,
            llm=Ollama(model="llama3:latest")
        )

# Define your tasks with the specific data they operate on
class DataTask(Task):
    def __init__(self, description, agent, data, expected_output: str = None):
        super().__init__(
            description=description,
            agent=agent,
            expected_output=expected_output
        )
        self.data = data

    def execute(self, context=None):
        # Convert DataFrame to JSON string to pass to tools
        data_json = self.data.to_json(orient="records")
        return super().execute(context, data=data_json)

async def main():
    # Load data
    log_table_path = "log-table.csv"  # Replace with your log table path
    fact_table_path = "fact-table.csv"  # Replace with your fact table path
    data = load_data(log_table_path, fact_table_path)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    search_tool = DuckDuckGoSearchRun()

    # Define agents
    general_agent = FlightDataAgent(
        role="General Inquiry Agent",
        goal="Answer general questions about the data. Delegate to other agents for complex tasks.",
        backstory="An agent specialized in understanding user queries and routing them to the appropriate expert.",
        tools=[search_tool, understand_query, analyze_schema, analyze_data_types, process_natural_language_query],
    )

    delay_analysis_agent = FlightDataAgent(
        role="Flight Delay Analyst",
        goal="Analyze and provide insights into flight delays.",
        backstory="An agent with expertise in analyzing flight data to identify patterns and causes of delays.",
        tools=[analyze_flight_delays],
    )

    query_understanding_agent = FlightDataAgent(
        role="Query Understanding Agent",
        goal="Break down complex queries into manageable sub-queries.",
        backstory="An agent skilled in decomposing complex questions into simpler, actionable parts.",
        tools=[understand_query],
    )

    schema_analysis_agent = FlightDataAgent(
        role="Schema Analysis Agent",
        goal="Analyze and describe the schema of the dataset.",
        backstory="An agent that provides detailed explanations of the dataset's structure and meaning.",
        tools=[analyze_schema],
    )

    data_type_analysis_agent = FlightDataAgent(
        role="Data Type Analysis Agent",
        goal="Analyze and describe the data types and characteristics of the dataset.",
        backstory="An agent that provides detailed analysis of each column's data type, uniqueness, missing values, and distribution.",
        tools=[analyze_data_types],
    )

    nl_querying_agent = FlightDataAgent(
        role="Natural Language Querying Agent",
        goal="Translate natural language queries into executable code and answer questions based on the data.",
        backstory="An agent capable of understanding natural language and converting it into executable queries against the dataset.",
        tools=[process_natural_language_query],
    )

    # Define tasks
    general_query_task = DataTask(
        description="""Answer general questions about the flight data.
        If the query is complex, use the Query Understanding Agent.
        If the query is about flight delays, use the Flight Delay Analysis Agent.""",
        agent=general_agent,
        data=data,
    )

    delay_analysis_task = DataTask(
        description="""Analyze flight delays in the data. Provide insights into the causes and patterns of delays.""",
        agent=delay_analysis_agent,
        data=data,
        expected_output="Average delay and other relevant statistics."
    )

    query_understanding_task = DataTask(
        description="""Break down the complex user query into simpler sub-queries.""",
        agent=query_understanding_agent,
        data=data,
        expected_output="List of sub-queries in JSON format."
    )

    schema_analysis_task = DataTask(
        description="""Analyze the schema of the dataset. Provide a detailed description of each column and its meaning.""",
        agent=schema_analysis_agent,
        data=data,
        expected_output="Description of the dataset schema."
    )

    data_type_analysis_task = DataTask(
        description="""Analyze the data types and characteristics of each column in the dataset.""",
        agent=data_type_analysis_agent,
        data=data,
        expected_output="Analysis of data types and characteristics."
    )

    nl_querying_task = DataTask(
        description="""Translate the natural language query into executable code and provide the answer.""",
        agent=nl_querying_agent,
        data=data,
        expected_output="Answer to the natural language query."
    )

    # Instantiate the crew with a sequential process
    crew = Crew(
        agents=[
            general_agent,
            delay_analysis_agent,
            query_understanding_agent,
            schema_analysis_agent,
            data_type_analysis_agent,
            nl_querying_agent,
        ],
        tasks=[
            general_query_task,
            delay_analysis_task,
            query_understanding_task,
            schema_analysis_task,
            data_type_analysis_task,
            nl_querying_task,
        ],
        process=Process.sequential,
    )

    # Example usage
    # Get the results
    result = crew.kickoff()
    print("######################")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())