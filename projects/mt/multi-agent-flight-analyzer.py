from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import ollama
import asyncio
import json
from datetime import datetime

# Pydantic Models
class AnalysisMetrics(BaseModel):
    metric_name: str
    value: float

class AnalysisResponse(BaseModel):
    analysis: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    visualization_type: str = Field(default="bar")
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentQuery(BaseModel):
    query: str
    context: Optional[Dict] = None

class MultiAgentResponse(BaseModel):
    flight_analysis: AnalysisResponse
    passenger_analysis: AnalysisResponse
    security_analysis: AnalysisResponse
    query_timestamp: datetime = Field(default_factory=datetime.now)

class BaseAgent:
    def __init__(self, model_name: str = "llama2:1b"):
        self.model = model_name
        
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        raise NotImplementedError

class FlightAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_template = """
        You are a flight data analyst. Analyze this query and return a JSON response:
        {query}
        
        Focus on flight-specific metrics like delays, schedules, and routes.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "bar"
        }}
        """
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.prompt_template.format(query=query.query)
                }]
            )
            
            # Parse the response and validate with Pydantic
            response_data = json.loads(response['message']['content'])
            return AnalysisResponse(**response_data)
        except Exception as e:
            return AnalysisResponse(
                analysis=f"Error processing flight analysis: {str(e)}",
                metrics={},
                visualization_type="bar"
            )

class PassengerAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_template = """
        You are a passenger data analyst. Analyze this query and return a JSON response:
        {query}
        
        Focus on passenger metrics like wait times, citizenship status, and confirmations.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "line"
        }}
        """
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.prompt_template.format(query=query.query)
                }]
            )
            
            response_data = json.loads(response['message']['content'])
            return AnalysisResponse(**response_data)
        except Exception as e:
            return AnalysisResponse(
                analysis=f"Error processing passenger analysis: {str(e)}",
                metrics={},
                visualization_type="line"
            )

class SecurityAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_template = """
        You are a security data analyst. Analyze this query and return a JSON response:
        {query}
        
        Focus on security metrics like APIS data, referrals, and security checks.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "pie"
        }}
        """
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.prompt_template.format(query=query.query)
                }]
            )
            
            response_data = json.loads(response['message']['content'])
            return AnalysisResponse(**response_data)
        except Exception as e:
            return AnalysisResponse(
                analysis=f"Error processing security analysis: {str(e)}",
                metrics={},
                visualization_type="pie"
            )

class MultiAgentOrchestrator:
    def __init__(self):
        self.flight_agent = FlightAnalysisAgent()
        self.passenger_agent = PassengerAnalysisAgent()
        self.security_agent = SecurityAnalysisAgent()
    
    async def process_query(self, query_str: str) -> MultiAgentResponse:
        query = AgentQuery(query=query_str)
        
        tasks = [
            self.flight_agent.process(query),
            self.passenger_agent.process(query),
            self.security_agent.process(query)
        ]
        
        flight_analysis, passenger_analysis, security_analysis = await asyncio.gather(*tasks)
        
        return MultiAgentResponse(
            flight_analysis=flight_analysis,
            passenger_analysis=passenger_analysis,
            security_analysis=security_analysis
        )

async def setup_model():
    """Check if model exists and is ready"""
    try:
        # Just try a simple test query to verify model is ready
        await asyncio.to_thread(
            ollama.chat,
            model="llama2:1b",
            messages=[{'role': 'user', 'content': 'test'}]
        )
        print("Model llama2:1b is ready")
        return True
    except Exception as e:
        print(f"Error accessing model: {e}")
        print("If model is not pulled, run: ollama pull llama2:1b")
        return False

async def main():
    # Verify model is accessible
    if not await setup_model():
        return

    orchestrator = MultiAgentOrchestrator()
    
    # Example queries
    queries = [
        "What are the average wait times for US vs non-US passengers?",
        "What are the most common flight delay patterns?",
        "How many security referrals occurred and what were their types?"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        try:
            result = await orchestrator.process_query(query)
            # Print the response as formatted JSON
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    asyncio.run(main())