import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
import ollama
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state type for our agent
class AgentState(TypedDict):
    """Represents the state of our agent."""
    messages: List[BaseMessage]
    next: str
    data: pd.DataFrame
    query: Optional[str] = None
    sender: Optional[str] = None  # Tracks which component sent the current message

class LLMInterface:
    """Interface for interacting with Ollama LLM"""
    
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        
    async def generate_analysis(self, prompt: str, context: Optional[Dict] = None) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                context=context
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error generating analysis: {str(e)}"

class SchemaAnalysisAgent:
    """Agent for analyzing the schema and data types of the input data."""

    def __init__(self):
        self.llm = LLMInterface()

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
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

            llm_schema_description = await self.llm.generate_analysis(schema_prompt)
            
            return {
                "schema": schema_info,
                "schema_description": llm_schema_description,
            }
        except Exception as e:
            logger.error(f"Error in SchemaAnalysisAgent.process_data: {e}")
            return {"error": f"Schema analysis failed: {str(e)}"}

class DataTypeAnalysisAgent:
    """Agent for analyzing data types and characteristics of the data."""

    def __init__(self):
        self.llm = LLMInterface()

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            type_info = {}

            for col in data.columns:
                type_info[col] = {}
                type_info[col]["dtype"] = str(data[col].dtype)
                type_info[col]["unique_values"] = data[col].nunique()
                type_info[col]["is_unique"] = data[col].is_unique
                type_info[col]["missing_percentage"] = (data[col].isnull().sum() / len(data)) * 100

                if pd.api.types.is_numeric_dtype(data[col]):
                    type_info[col]["min"] = float(data[col].min())
                    type_info[col]["max"] = float(data[col].max())
                    type_info[col]["mean"] = float(data[col].mean())
                    type_info[col]["median"] = float(data[col].median())
                elif pd.api.types.is_string_dtype(data[col]):
                    type_info[col]["value_counts"] = data[col].value_counts().head(10).to_dict()

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

            llm_type_description = await self.llm.generate_analysis(type_prompt)

            return {
                "data_types": type_info,
                "type_description": llm_type_description,
            }
        except Exception as e:
            logger.error(f"Error in DataTypeAnalysisAgent.process_data: {e}")
            return {"error": f"Data type analysis failed: {str(e)}"}

class FlightDelayAnalysisAgent:
    """Agent for analyzing flight delays."""

    def __init__(self):
        self.llm = LLMInterface()

    async def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            delay_stats = {}
            
            if "arvl_dlay_cat_cd" in data.columns:
                delay_stats["avg_arrival_delay"] = float(data["arvl_dlay_cat_cd"].mean())
                delay_stats["max_arrival_delay"] = float(data["arvl_dlay_cat_cd"].max())
                delay_stats["delay_distribution"] = data["arvl_dlay_cat_cd"].value_counts().to_dict()
            
            analysis_prompt = f"""
            Analyze the following flight delay statistics:
            {json.dumps(delay_stats, indent=2)}

            Provide insights into:
            1. The overall delay patterns
            2. The severity of delays
            3. Any notable patterns in the delay distribution
            4. Recommendations for improvement

            Use clear, non-technical language.
            """
            
            llm_analysis = await self.llm.generate_analysis(analysis_prompt)

            return {
                "delay_stats": delay_stats,
                "analysis": llm_analysis,
            }
        except Exception as e:
            logger.error(f"Error in FlightDelayAnalysisAgent.process_data: {e}")
            return {"error": f"Flight delay analysis failed: {str(e)}"}

class QueryUnderstandingAgent:
    """Agent for breaking down complex queries into smaller sub-queries."""

    def __init__(self):
        self.llm = LLMInterface()

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        try:
            decomposition_prompt = f"""
            You are an expert at breaking down complex data analysis questions.
            
            The user has asked: "{query}"

            Break this question down into simpler sub-questions that can be answered individually.
            Format your response as a JSON list of strings.
            """

            response = await self.llm.generate_analysis(decomposition_prompt)
            
            try:
                sub_queries = json.loads(response)
                if not isinstance(sub_queries, list):
                    sub_queries = [response]
            except json.JSONDecodeError:
                sub_queries = [response]

            return {
                "original_query": query,
                "sub_queries": sub_queries,
            }
        except Exception as e:
            logger.error(f"Error in QueryUnderstandingAgent.process_data: {e}")
            return {"error": f"Query understanding failed: {str(e)}"}

class NaturalLanguageQueryingAgent:
    """Agent for handling natural language queries about the data."""

    def __init__(self, schema_agent: SchemaAnalysisAgent = None, type_agent: DataTypeAnalysisAgent = None):
        self.llm = LLMInterface()
        self.schema_agent = schema_agent or SchemaAnalysisAgent()
        self.type_agent = type_agent or DataTypeAnalysisAgent()

    async def process_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        try:
            schema_results = await self.schema_agent.process_data(data.copy())
            type_results = await self.type_agent.process_data(data.copy())

            query_prompt = f"""
            You are a data analyst who can translate natural language queries into Python code.

            The user asks: "{query}"

            Schema information:
            {json.dumps(schema_results.get('schema', {}), indent=2)}

            Data type information:
            {json.dumps(type_results.get('data_types', {}), indent=2)}

            Generate Python pandas code to answer this query.
            The DataFrame is named 'data'.
            Return ONLY the code, no explanations.
            """

            generated_code = await self.llm.generate_analysis(query_prompt)

            try:
                # Create a safe local environment for code execution
                local_vars = {"data": data.copy(), "pd": pd, "result": None}
                exec(generated_code, {}, local_vars)
                result = local_vars.get("result")
                if isinstance(result, pd.DataFrame):
                    result = result.to_string()
            except Exception as e:
                result = f"Error executing generated code: {str(e)}"

            return {
                "query": query,
                "generated_code": generated_code,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Error in NaturalLanguageQueryingAgent.process_data: {e}")
            return {"error": f"Natural language querying failed: {str(e)}"}

# Define tools to use in graph
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
    try:
        llm = LLMInterface()
        general_query_prompt = f"""
        You are a helpful assistant that can answer general questions about the flight data.
        The user is asking: "{query}"
        Provide a concise answer based on your understanding of the data.
        """
        response = asyncio.run(llm.generate_analysis(general_query_prompt))
        return {"response": response}
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

def agent_node(state: AgentState) -> Dict:
    """Main agent node that processes queries and routes to appropriate specialized agents."""
    try:
        llm = LLMInterface()
        
        # Define available tools and their purposes
        tools_prompt = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        prompt = f"""
        You are a helpful assistant that can answer questions about flight data.
        Available tools:
        {tools_prompt}
        
        User query: {state['query']}
        
        Which tool would be most appropriate? Respond with just the tool name.
        """
        
        tool_choice = asyncio.run(llm.generate_analysis(prompt))
        tool_choice = tool_choice.strip().lower()
        
        next_step = {
            "handoff_to_flight_delay_agent": "flight_delay_analysis",
            "handoff_to_query_understanding_agent": "query_understanding",
            "process_general_query": "general_query",
            "analyze_schema": "schema_analysis",
            "analyze_data_types": "data_type_analysis",
            "process_natural_language_query": "natural_language_querying"
        }.get(tool_choice, END)

        return {
            "messages": [HumanMessage(content=f"Using tool: {tool_choice}")],
            "data": state["data"],
            "next": next_step,
            "query": state["query"],
            "sender": "agent"
        }
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        return {
            "messages": [HumanMessage(content=f"Error: {str(e)}")],
            "data": state["data"],
            "next": END,
            "sender": "agent"
        }

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

async def main():
    """Main function to run the system."""
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
        workflow.add_node("agent", lambda state: agent_node(state))
        workflow.add_node("flight_delay_analysis", lambda state: FlightDelayAnalysisAgent().process_data(state["data"]))
        workflow.add_node("query_understanding", lambda state: QueryUnderstandingAgent().process_data(state["data"], state["query"]))
        workflow.add_node("general_query", lambda state: process_general_query(state["data"], state["query"]))
        workflow.add_node("schema_analysis", lambda state: SchemaAnalysisAgent().process_data(state["data"]))
        workflow.add_node("data_type_analysis", lambda state: DataTypeAnalysisAgent().process_data(state["data"]))
        workflow.add_node("natural_language_querying", lambda state: NaturalLanguageQueryingAgent(
            SchemaAnalysisAgent(), DataTypeAnalysisAgent()
        ).process_data(state["data"], state["query"]))

        # Set entry point
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

        # Compile workflow
        app = workflow.compile()

        # Example queries
        test_queries = [
            "what is the schema of this dataset",
            "what are the datatypes in this dataset",
            "analyze flight delays",
            "what is the average arrival delay"
        ]

        # Process each query
        for query in test_queries:
            try:
                print(f"\nProcessing query: {query}")
                print("-" * 50)
                
                # Prepare input state
                inputs = {
                    "query": query,
                    "data": data,
                    "messages": [],
                }
                
                # Run through workflow
                response = await app.ainvoke(inputs)
                
                # Pretty print the response
                if isinstance(response, dict):
                    for key, value in response.items():
                        print(f"\n{key}:")
                        if isinstance(value, dict):
                            print(json.dumps(value, indent=2))
                        else:
                            print(value)
                else:
                    print(response)
                    
                print("-" * 50)
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                print(f"Error processing query: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error in main: {str(e)}")


# For Jupyter notebook compatibility
nest_asyncio.apply()

def run_analysis():
    """Function to run the analysis, handling both Jupyter and script environments."""
    try:
        # If running in Jupyter notebook
        if 'ipykernel' in sys.modules:
            asyncio.get_event_loop().run_until_complete(main())
        # If running as a script
        else:
            asyncio.run(main())
    except Exception as e:
        print(f"Error running analysis: {str(e)}")

if __name__ == "__main__":
    import sys
    run_analysis()