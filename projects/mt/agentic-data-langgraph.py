import asyncio
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
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
    try:
        llm_response = generate_analysis(general_query_prompt)
        return {"response": llm_response}
    except Exception as e:
        logger.error(f"Error in process_general_query: {e}")
        return {"response": f"Error processing query: {str(e)}"}

@tool
def analyze_schema(data):
    """
    Analyzes the schema of the provided data.
    """
    try:
        schema_agent = SchemaAnalysisAgent()
        return schema_agent.process_data(data)
    except Exception as e:
        logger.error(f"Error in analyze_schema: {e}")
        return {"error": f"Schema analysis failed: {str(e)}"}

@tool
def analyze_data_types(data):
    """
    Analyzes the data types and characteristics of the data.
    """
    try:
        type_agent = DataTypeAnalysisAgent()
        return type_agent.process_data(data)
    except Exception as e:
        logger.error(f"Error in analyze_data_types: {e}")
        return {"error": f"Data type analysis failed: {str(e)}"}

@tool
def process_natural_language_query(data, query):
    """
    Processes natural language queries by translating them into executable code.
    """
    try:
        nl_query_agent = NaturalLanguageQueryingAgent(
            SchemaAnalysisAgent(), DataTypeAnalysisAgent()
        )
        return nl_query_agent.process_data(data, query)
    except Exception as e:
        logger.error(f"Error in process_natural_language_query: {e}")
        return {"error": f"Query processing failed: {str(e)}"}

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
        self.llm = Ollama(model="llama3.2:1b")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
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
        except Exception as e:
            logger.error(f"Error in SchemaAnalysisAgent.process_data: {e}")
            return {"error": f"Schema analysis failed: {str(e)}"}

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        try:
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"Error in generate_analysis: {e}")
            return f"Error generating analysis: {str(e)}"

class DataTypeAnalysisAgent:
    """Agent for analyzing data types and characteristics of the data."""

    def __init__(self):
        self.llm = Ollama(model="llama3.2:1b")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            type_info = {}

            for col in data.columns:
                type_info[col] = {}
                type_info[col]["dtype"] = str(data[col].dtype)
                type_info[col]["unique_values"] = data[col].nunique()
                type_info[col]["is_unique"] = data[col].is_unique
                type_info[col]["missing_percentage"] = (
                    (data[col].isnull().sum() / len(data)) * 100
                )

                if pd.api.types.is_numeric_dtype(data[col]):
                    type_info[col]["min"] = data[col].min()
                    type_info[col]["max"] = data[col].max()
                    type_info[col]["mean"] = data[col].mean()
                    type_info[col]["median"] = data[col].median()
                elif pd.api.types.is_string_dtype(data[col]):
                    type_info[col]["value_counts"] = (
                        data[col].value_counts().head(10).to_dict()
                    )

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
        except Exception as e:
            logger.error(f"Error in DataTypeAnalysisAgent.process_data: {e}")
            return {"error": f"Data type analysis failed: {str(e)}"}

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        try:
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"Error in generate_analysis: {e}")
            return f"Error generating analysis: {str(e)}"

class FlightDelayAnalysisAgent:
    """Agent for analyzing flight delays."""

    def __init__(self):
        self.llm = Ollama(model="llama3.2:1b")

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if "arvl_dlay_cat_cd" in data.columns:
                avg_arrival_delay = data["arvl_dlay_cat_cd"].mean()
            else:
                avg_arrival_delay = "Column 'arvl_dlay_cat_cd' not found in data"

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
        except Exception as e:
            logger.error(f"Error in FlightDelayAnalysisAgent.process_data: {e}")
            return {"error": f"Flight delay analysis failed: {str(e)}"}

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        try:
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"Error in generate_analysis: {e}")
            return f"Error generating analysis: {str(e)}"

class QueryUnderstandingAgent:
    """Agent that breaks down complex queries into smaller sub-queries."""

    def __init__(self):
        self.llm = Ollama(model="llama3.2:1b")

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        try:
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
                logger.error("Could not parse sub-queries from LLM response.")

            return {
                "original_query": query,
                "sub_queries": sub_queries,
            }
        except Exception as e:
            logger.error(f"Error in QueryUnderstandingAgent.process_data: {e}")
            return {"error": f"Query understanding failed: {str(e)}"}

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        try:
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"Error in generate_analysis: {e}")
            return f"Error generating analysis: {str(e)}"

class NaturalLanguageQueryingAgent:
    """Agent for handling natural language queries about the data."""

    def __init__(self, schema_agent: SchemaAnalysisAgent, type_agent: DataTypeAnalysisAgent):
        self.llm = Ollama(model="llama3.2:1b")
        self.schema_agent = schema_agent
        self.type_agent = type_agent

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        try:
            schema_results = await self.schema_agent.process_data(data.copy())
            type_results = await self.type_agent.process_data(data.copy())

            schema_info = schema_results["schema"]
            schema_description = schema_results["schema_description"]
            type_info = type_results["data_types"]
            type_description = type_results["type_description"]

            query_prompt = f"""
            You are a data analyst who can translate natural language queries into Python code.

            Here is the schema of the DataFrame:
            {json.dumps(schema_info, indent=2)}
            
            Schema Description:
            {schema_description}

            Data Types:
            {json.dumps(type_info, indent=2)}

            Type Description:
            {type_description}

            The user wants to know: "{query}"

            Generate Python code using pandas to answer this query.
            Assume the DataFrame is named 'data'.
            """

            generated_code = await self.generate_analysis(query_prompt)

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
        except Exception as e:
            logger.error(f"Error in NaturalLanguageQueryingAgent.process_data: {e}")
            return {"error": f"Natural language querying failed: {str(e)}"}

    async def generate_analysis(self, prompt: str) -> str:
        """Helper function to generate analysis using the LLM."""
        try:
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"Error in generate_analysis: {e}")
            return f"Error generating analysis: {str(e)}"

class AgentState(TypedDict):
    """Represents the state of our agent."""
    messages: List[BaseMessage]
    next: str
    data: pd.DataFrame
    query: Optional[str] = None
    sender: Optional[str] = None

# [Previous imports and class definitions remain the same until load_data function]

def load_data(log_table_path: str, fact_table_path: str) -> pd.DataFrame:
    """Loads and preprocesses data from the log and fact tables."""
    try:
        log_df = pd.read_csv(log_table_path)
        fact_df = pd.read_csv(fact_table_path)

        # Data Cleaning
        log_df.columns = log_df.columns.str.lower()
        fact_df.columns = fact_df.columns.str.lower()
        
        # Convert date columns
        for df in [log_df, fact_df]:
            date_columns = [col for col in df.columns if col.endswith(("_dt", "_dttm"))]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Join tables
        common_columns = list(set(log_df.columns).intersection(fact_df.columns))
        merged_df = pd.merge(log_df, fact_df, on=common_columns, how="inner")
        
        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        return merged_df

    except Exception as e:
        logger.error(f"Error loading or merging data: {str(e)}")
        return None

def agent_node(state: AgentState, data: pd.DataFrame) -> Dict:
    """Processes the user query and determines the next step."""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can answer general questions about flight data."),
            ("human", "{input}"),
        ])
        
        llm = Ollama(model="llama3.2:1b").bind_tools(tools)
        chain = prompt | llm
        result = chain.invoke({"input": state["query"]}, config={"run_name": "agent"})

        content = result.content if not isinstance(result.content, str) else result.content

        if result.additional_kwargs and "tool_calls" in result.additional_kwargs:
            calls = result.additional_kwargs["tool_calls"]
            logger.info(f"Agent identified tool calls: {calls}")

            if calls:
                function_name = calls[0]["function"]["name"]
                next_step = {
                    "handoff_to_flight_delay_agent": "flight_delay_analysis",
                    "handoff_to_query_understanding_agent": "query_understanding",
                    "process_general_query": "general_query",
                    "analyze_schema": "schema_analysis",
                    "analyze_data_types": "data_type_analysis",
                    "process_natural_language_query": "natural_language_querying"
                }.get(function_name, END)
            else:
                next_step = END

            messages = []
            if content:
                messages.append(HumanMessage(content=content))
            for call in calls:
                messages.append(
                    FunctionMessage(name=call["function"]["name"], 
                                  content=json.dumps(call["function"]))
                )

            return {"messages": messages, "data": data, "next": next_step}
        else:
            return {"messages": [HumanMessage(content=content)], "data": data, "next": END}
            
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")], 
            "data": data, 
            "next": END
        }

# Agent functions
async def flight_delay_analysis_agent(state: AgentState) -> Dict:
    """Handles flight delay analysis."""
    try:
        data = state["data"]
        agent = FlightDelayAnalysisAgent()
        result = await agent.process_data(data)
        return {
            "messages": [HumanMessage(content=result["llm_analysis"])],
            "data": data,
            "next": END,
        }
    except Exception as e:
        logger.error(f"Error in flight_delay_analysis_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def query_understanding_agent(state: AgentState) -> Dict:
    """Handles query understanding."""
    try:
        data = state["data"]
        query = state["query"]
        agent = QueryUnderstandingAgent()
        result = await agent.process_data(data, query)
        return {
            "messages": [HumanMessage(content=json.dumps(result))],
            "data": data,
            "next": "natural_language_querying" if result.get("sub_queries") else END,
        }
    except Exception as e:
        logger.error(f"Error in query_understanding_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def general_query_agent(state: AgentState) -> Dict:
    """Handles general queries."""
    try:
        data = state["data"]
        query = state["query"]
        result = process_general_query(data, query)
        return {
            "messages": [HumanMessage(content=result["response"])],
            "data": data,
            "next": END,
        }
    except Exception as e:
        logger.error(f"Error in general_query_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def schema_analysis_agent(state: AgentState) -> Dict:
    """Handles schema analysis."""
    try:
        data = state["data"]
        agent = SchemaAnalysisAgent()
        result = await agent.process_data(data)
        return {
            "messages": [HumanMessage(content=result["schema_description"])],
            "data": data,
            "next": END,
        }
    except Exception as e:
        logger.error(f"Error in schema_analysis_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def data_type_analysis_agent(state: AgentState) -> Dict:
    """Handles data type analysis."""
    try:
        data = state["data"]
        agent = DataTypeAnalysisAgent()
        result = await agent.process_data(data)
        return {
            "messages": [HumanMessage(content=result["type_description"])],
            "data": data,
            "next": END,
        }
    except Exception as e:
        logger.error(f"Error in data_type_analysis_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def natural_language_querying_agent(state: AgentState) -> Dict:
    """Handles natural language querying."""
    try:
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
    except Exception as e:
        logger.error(f"Error in natural_language_querying_agent: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": data,
            "next": END,
        }

async def main():
    """Main function to run the agent system."""
    try:
        # Load data
        log_table_path = "log-table.csv"
        fact_table_path = "fact-table.csv"
        data = load_data(log_table_path, fact_table_path)
        
        if data is None:
            raise ValueError("Failed to load data")

        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", lambda state: agent_node(state, state["data"]))
        workflow.add_node("flight_delay_analysis", flight_delay_analysis_agent)
        workflow.add_node("query_understanding", query_understanding_agent)
        workflow.add_node("general_query", general_query_agent)
        workflow.add_node("schema_analysis", schema_analysis_agent)
        workflow.add_node("data_type_analysis", data_type_analysis_agent)
        workflow.add_node("natural_language_querying", natural_language_querying_agent)

        # Set entry point and edges
        workflow.set_entry_point("agent")
        
        # Add edges
        for node in ["flight_delay_analysis", "general_query", "schema_analysis", 
                    "data_type_analysis", "natural_language_querying"]:
            workflow.add_edge(node, END)
        
        workflow.add_edge("query_understanding", "natural_language_querying")

        # Add conditional edges
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

        # Compile and run
        app = workflow.compile()

        # Example queries
        test_queries = [
            "what is the schema of this dataset",
            "what are the datatypes in this dataset",
            "analyze flight delays",
            "what is the average arrival delay"
        ]

        for query in test_queries:
            try:
                inputs = {
                    "query": query,
                    "data": data,
                    "messages": [],
                }
                response = app.invoke(inputs)
                print(f"\nQuery: {query}")
                print("Response:", response)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
