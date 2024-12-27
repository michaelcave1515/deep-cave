import asyncio
import json
import os
from typing import Dict, List, Optional, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.llms.ollama import Ollama

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tool to use in graph
@tool
def handoff_to_flight_delay_agent(data):
    """
    Hands off to the FlightDelayAnalysisAgent.
    """
    flight_delay_agent = FlightDelayAnalysisAgent()
    return flight_delay_agent.process_data(data)

@tool
def handoff_to_query_understanding_agent(data):
    """
    Hands off to the QueryUnderstandingAgent.
    """
    query_understanding_agent = QueryUnderstandingAgent()
    return query_understanding_agent.process_data(data)

@tool
def process_general_query(data, query):
    """
    Handles general queries using the LLM.
    """
    general_query_prompt = f"""
    You are a helpful assistant that can answer general questions about the flight data.

    Here is the schema of the data:
    # ... (You might want to include schema information here, similar to what's done in NaturalLanguageQueryingAgent) ...

    The user is asking: "{query}"

    Provide a concise answer based on your understanding of the data.
    """
    llm_response = generate_analysis(
        general_query_prompt
    )  # Assuming generate_analysis is accessible here
    return {"response": llm_response}

@tool
def analyze_schema(data):
    """
    Analyzes the schema of the provided data.
    """
    schema_agent = SchemaAnalysisAgent()
    return schema_agent.process_data(data)

@tool
def analyze_data_types(data):
    """
    Analyzes the data types and characteristics of the data.
    """
    type_agent = DataTypeAnalysisAgent()
    return type_agent.process_data(data)

@tool
def process_natural_language_query(data, query):
    """
    Processes natural language queries by translating them into executable code.
    """
    nl_query_agent = NaturalLanguageQueryingAgent(
        SchemaAnalysisAgent(), DataTypeAnalysisAgent()
    )
    return nl_query_agent.process_data(data, query)

# Define the list of tools for the agent
tools = [
    handoff_to_flight_delay_agent,
    handoff_to_query_understanding_agent,
    process_general_query,
    analyze_schema,
    analyze_data_types,
    process_natural_language_query,
]

class SchemaAnalysisAgent:
    """Agent for analyzing the schema and data types of the input data."""

    def __init__(self):
        self.llm = Ollama(model="llama3:latest")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
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
        response = self.llm.invoke(messages)
        return response

class DataTypeAnalysisAgent:
    """Agent for analyzing data types and characteristics of the data."""

    def __init__(self):
        self.llm = Ollama(model="llama3:latest")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
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
        response = self.llm.invoke(messages)
        return response

class FlightDelayAnalysisAgent:
    """Agent for analyzing flight delays."""

    def __init__(self):
        self.llm = Ollama(model="llama3:latest")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
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
        response = self.llm.invoke(messages)
        return response

class QueryUnderstandingAgent:
    """Agent that breaks down complex queries into smaller sub-queries."""

    def __init__(self):
        self.llm = Ollama(model="llama3:latest")

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
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
        response = self.llm.invoke(messages)
        return response

class NaturalLanguageQueryingAgent:
    """Agent for handling natural language queries about the data."""

    def __init__(
        self, schema_agent: SchemaAnalysisAgent, type_agent: DataTypeAnalysisAgent
    ):
        self.llm = Ollama(model="llama3:latest")
        self.schema_agent = schema_agent
        self.type_agent = type_agent

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        # Get schema and data type information from other agents
        schema_results = await self.schema_agent.process_data(data.copy())
        type_results = await self.type_agent.process_data(data.copy())

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
        response = self.llm.invoke(messages)
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
        response = self.llm.invoke(messages)
        return response

class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages (List[BaseMessage]): A list of BaseMessage objects representing the conversation history.
        next (str): The next step in the conversation.
        data (pd.DataFrame): The data being analyzed.
        query (Optional[str]): The user's query.
        sender (Optional[str]): The sender of the message.
    """

    messages: List[BaseMessage]
    next: str
    data: pd.DataFrame
    query: Optional[str] = None
    sender: Optional[str] = None

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

llm = Ollama(model="llama3:latest").bind_tools(tools)

# The agent node is the entry point for all user queries
def agent_node(state: AgentState, data: pd.DataFrame) -> Dict:
    """
    Processes the user query and determines the next step.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Dict: The updated state with the next step.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that can answer general questions about flight data.",
            ),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | llm
    result = chain.invoke({"input": state["query"]}, config={"run_name": "agent"})

    # Extract the content from the AIMessage
    content = result.content if not isinstance(result.content, str) else result.content

    # Check if there are any function calls in the result
    if result.additional_kwargs and "tool_calls" in result.additional_kwargs:
        calls = result.additional_kwargs["tool_calls"]
        logger.info(f"Agent identified tool calls: {calls}")

        # Assuming the first tool call determines the next action
        if calls:
            function_name = calls[0]["function"]["name"]

            # Determine the next step based on the function name
            if function_name == "handoff_to_flight_delay_agent":
                next_step = "flight_delay_analysis"
            elif function_name == "handoff_to_query_understanding_agent":
                next_step = "query_understanding"
            elif function_name == "process_general_query":
                next_step = "general_query"
            elif function_name == "analyze_schema":
                next_step = "schema_analysis"
            elif function_name == "analyze_data_types":
                next_step = "data_type_analysis"
            elif function_name == "process_natural_language_query":
                next_step = "natural_language_querying"
            else:
                next_step = END
        else:
            next_step = END

        # Convert AIMessage to list of BaseMessage for compatibility with MessagesState
        messages = []
        if content:
            messages.append(HumanMessage(content=content))
        for call in calls:
            messages.append(
                FunctionMessage(name=call["function"]["name"], content=json.dumps(call["function"]))
            )

        return {"messages": messages, "data": data, "next": next_step}
    else:
        # If no tool calls, treat this as the final response
        return {"messages": [HumanMessage(content=content)], "data": data, "next": END}

# Define agent functions
async def flight_delay_analysis_agent(state: AgentState) -> Dict:
    """
    Handles flight delay analysis.
    """
    data = state["data"]
    agent = FlightDelayAnalysisAgent()
    result = await agent.process_data(data)
    return {
        "messages": [HumanMessage(content=result["llm_analysis"])],
        "data": data,
        "next": END,
    }

async def query_understanding_agent(state: AgentState) -> Dict:
    """
    Handles query understanding.
    """
    data = state["data"]
    query = state["query"]
    agent = QueryUnderstandingAgent()
    result = await agent.process_data(data, query)
    return {
        "messages": [HumanMessage(content=json.dumps(result))],
        "data": data,
        "next": "natural_language_querying" if result["sub_queries"] else END,
    }

async def general_query_agent(state: AgentState) -> Dict:
    """
    Handles general queries.
    """
    data = state["data"]
    query = state["query"]
    result = process_general_query(data, query)
    return {
        "messages": [HumanMessage(content=result["response"])],
        "data": data,
        "next": END,
    }

async def schema_analysis_agent(state: AgentState) -> Dict:
    """
    Handles schema analysis.
    """
    data = state["data"]
    agent = SchemaAnalysisAgent()
    result = await agent.process_data(data)
    return {
        "messages": [HumanMessage(content=result["schema_description"])],
        "data": data,
        "next": END,
    }

async def data_type_analysis_agent(state: AgentState) -> Dict:
    """
    Handles data type analysis.
    """
    data = state["data"]
    agent = DataTypeAnalysisAgent()
    result = await agent.process_data(data)
    return {
        "messages": [HumanMessage(content=result["type_description"])],
        "data": data,
        "next": END,
    }

async def natural_language_querying_agent(state: AgentState) -> Dict:
    """
    Handles natural language querying.
    """
    data = state["data"]
    query = state["query"]
    agent = NaturalLanguageQueryingAgent(
        SchemaAnalysisAgent(), DataTypeAnalysisAgent()
    )
    result = await agent.process_data(data, query)
    return {
        "messages": [HumanMessage(content=result["result"])],
        "data": data,
        "next": END,
    }

# Create a mapping of function names to the actual functions
agent_functions = {
    "flight_delay_analysis": flight_delay_analysis_agent,
    "query_understanding": query_understanding_agent,
    "general_query": general_query_agent,
    "schema_analysis": schema_analysis_agent,
    "data_type_analysis": data_type_analysis_agent,
    "natural_language_querying": natural_language_querying_agent,
}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("agent", lambda state: agent_node(state, state["data"]))
workflow.add_node("flight_delay_analysis", flight_delay_analysis_agent)
workflow.add_node("query_understanding", query_understanding_agent)
workflow.add_node("general_query", general_query_agent)
workflow.add_node("schema_analysis", schema_analysis_agent)
workflow.add_node("data_type_analysis", data_type_analysis_agent)
workflow.add_node("natural_language_querying", natural_language_querying_agent)

# Set the entrypoint
workflow.set_entry_point("agent")

# Add edges
workflow.add_edge("flight_delay_analysis", END)
workflow.add_edge("query_understanding", "natural_language_querying")
workflow.add_edge("general_query", END)
workflow.add_edge("schema_analysis", END)
workflow.add_edge("data_type_analysis", END)
workflow.add_edge("natural_language_querying", END)

# Add conditional edge from agent node
workflow.add_conditional_edges(
    "agent",
    lambda x: x["next"],
    {
        "flight_delay_analysis": "flight_delay_analysis",
        "query_understanding": "query_understanding",
        "general_query": "general_query",
        "schema_analysis": "schema_analysis",
        "data_type_analysis": "data_type_analysis",
        "natural_language_querying": "natural_language_querying",
        END: END,
    },
)

# Compile the graph
app = workflow.compile()

async def main():
    # Load data (replace with your actual file paths)
    log_table_path = "log-table.csv"
    fact_table_path = "fact-table.csv"
    data = load_data(log_table_path, fact_table_path)

    # Example usage
    inputs = {
        "query": "what is the schema of this dataset",
        "data": data,
        "messages": [],
    }
    response = app.invoke(inputs)
    print("Final Response:", response)

    inputs = {
        "query": "what are the datatypes in this dataset",
        "data": data,
        "messages": [],
    }
    response = app.invoke(inputs)
    print("Final Response:", response)

    inputs = {
        "query": "analyze flight delays",
        "data": data,
        "messages": [],
    }
    response = app.invoke(inputs)
    print("Final Response:", response)

    inputs = {
        "query": "what is the average arrival delay",
        "data": data,
        "messages": [],
    }
    response = app.invoke(inputs)
    print("Final Response:", response)

if __name__ == "__main__":
    asyncio.run(main())