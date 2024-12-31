from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import ollama
import asyncio
import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Data Models
class FlightData(BaseModel):
    log_data: Optional[pd.DataFrame] = None
    fact_data: Optional[pd.DataFrame] = None
    
    class Config:
        arbitrary_types_allowed = True

# Pydantic Models
class AnalysisMetrics(BaseModel):
    metric_name: str
    value: float

class AnalysisResponse(BaseModel):
    analysis: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    visualization_type: str = Field(default="bar")
    timestamp: datetime = Field(default_factory=datetime.now)
    data_summary: Optional[Dict[str, Union[float, str]]] = None

class AgentQuery(BaseModel):
    query: str
    context: Optional[Dict] = None
    data: Optional[FlightData] = None

    class Config:
        arbitrary_types_allowed = True

class MultiAgentResponse(BaseModel):
    flight_analysis: AnalysisResponse
    passenger_analysis: AnalysisResponse
    security_analysis: AnalysisResponse
    query_timestamp: datetime = Field(default_factory=datetime.now)

# Data Loading
class DataLoader:
    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        try:
            print(f"Loading CSV file: {file_path}")
            
            # First attempt to read the CSV with dtype=object to avoid mixed type warnings
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                dtype=object,
                low_memory=False
            )
            
            # Convert numeric columns after loading
            numeric_columns = [
                'NON_US_AVG_WAIT_TM_NBR', 'US_AVG_WAIT_TM_NBR', 'AVG_WAIT_TM_NBR',
                'PSNGR_ON_BORD_QTY', 'PSNGR_ON_BORD_USC_QTY', 'PSNGR_ON_BORD_NON_USC_QTY',
                'APIS_NCIC_QTY', 'APIS_PRMRY_QTY', 'APIS_PSBL_NCIC_QTY',
                'APIS_PTIP_QTY', 'APIS_RFRL_QTY', 'APIS_CMI_QTY'
            ]
            
            # Convert numeric columns that exist in the DataFrame
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Define date columns
            date_columns = [
                'ARVL_EST_DT', 'DPRTR_SCHD_DT', 'ARVL_SCHD_DTTM', 'DPRTR_SCHD_DTTM',
                'DPRTR_EST_DTTM', 'DPRTR_ACTL_DTTM', 'ARVL_EST_DTTM', 'ARVL_ACTL_DTTM',
                'DPRTR_OFFBLOK_EST_DT', 'DPRTR_OFFBLOK_ACT_DT', 'DPRTR_AIRBRN_EST_DT',
                'DPRTR_AIRBRN_ACT_DT', 'ARVL_ONBLOK_EST_DT', 'ARVL_ONBLOK_ACT_DT',
                'ARVL_TCHDWN_EST_DT', 'ARVL_TCHDWN_ACT_DT', 'CRTE_DTTM', 'SRCE_UPD_DTTM',
                'OAG_LST_UPD_DTTM', 'ETA_UPD_DTTM', 'FLIT_STUS_DTTM', 'FRST_CNFIRM_DT',
                'LST_CNFIRM_DT', 'APIS_ARVL_SCHD_DTTM', 'APIS_DPRTR_SCHD_DTTM'
            ]
            
            # Convert date columns that exist in the DataFrame
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        print(f"Successfully converted {col} to datetime")
                    except Exception as e:
                        print(f"Error converting {col} to datetime: {e}")
            
            print(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
            return df
            
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def prepare_flight_data(log_path: str, fact_path: str) -> FlightData:
        log_df = DataLoader.load_csv(log_path)
        fact_df = DataLoader.load_csv(fact_path)
        
        return FlightData(
            log_data=log_df,
            fact_data=fact_df
        )

# Base Agent
class BaseAgent:
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model = model_name
        
    def analyze_data(self, data: FlightData) -> Dict[str, Union[float, str]]:
        raise NotImplementedError
        
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        raise NotImplementedError

class FlightAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_template = """
        You are a flight data analyst. Analyze this query and the provided metrics:
        Query: {query}
        Metrics: {metrics}
        
        Focus on flight-specific metrics like delays, schedules, and routes.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "bar"
        }}
        """
    
    def analyze_data(self, data: FlightData) -> Dict[str, Union[float, str]]:
        if data.log_data is None or data.log_data.empty:
            return {}
            
        metrics = {}
        try:
            # Analyze delays
            if 'DPRTR_DLAY_STUS_CD' in data.log_data.columns:
                delay_counts = data.log_data['DPRTR_DLAY_STUS_CD'].value_counts()
                metrics['delay_rate'] = float((len(delay_counts[delay_counts != '']) / len(data.log_data)) * 100)
                
            # Analyze schedule adherence
            if all(col in data.log_data.columns for col in ['DPRTR_SCHD_DTTM', 'DPRTR_ACTL_DTTM']):
                mask = data.log_data['DPRTR_ACTL_DTTM'].notna() & data.log_data['DPRTR_SCHD_DTTM'].notna()
                data.log_data['schedule_diff'] = np.where(
                    mask,
                    (data.log_data['DPRTR_ACTL_DTTM'] - data.log_data['DPRTR_SCHD_DTTM']).dt.total_seconds() / 60,
                    np.nan
                )
                metrics['avg_delay_minutes'] = float(data.log_data['schedule_diff'].mean())
                
            return metrics
        except Exception as e:
            print(f"Error analyzing flight data: {e}")
            return {}
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            metrics = self.analyze_data(query.data) if query.data else {}
            
            # Create a simplified prompt with just the metrics
            metrics_str = json.dumps(metrics, indent=2)
            prompt = f"""Analyze this flight data metrics and respond with analysis:
            Query: {query.query}
            Metrics: {metrics_str}
            
            Respond ONLY with valid JSON in this exact format:
            {{
                "analysis": "your analysis here",
                "metrics": {metrics_str},
                "visualization_type": "bar"
            }}"""

            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            try:
                # Try to parse the response content as JSON
                response_text = response['message']['content'].strip()
                # Find the first { and last } to extract just the JSON part
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    response_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response")
                    
                response_data['data_summary'] = metrics
                return AnalysisResponse(**response_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from LLM response: {e}")
                # Fallback response with the raw metrics
                return AnalysisResponse(
                    analysis=f"Analysis of metrics: {metrics_str}",
                    metrics=metrics,
                    visualization_type="bar",
                    data_summary=metrics
                )
                
        except Exception as e:
            print(f"Error in flight analysis: {str(e)}")
            return AnalysisResponse(
                analysis=f"Error processing flight analysis: {str(e)}",
                metrics=metrics if 'metrics' in locals() else {},
                visualization_type="bar",
                data_summary=metrics if 'metrics' in locals() else {}
            )

class PassengerAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt_template = """
        You are a passenger data analyst. Analyze this query and the provided metrics:
        Query: {query}
        Metrics: {metrics}
        
        Focus on passenger metrics like wait times, citizenship status, and confirmations.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "line"
        }}
        """
    
    def analyze_data(self, data: FlightData) -> Dict[str, Union[float, str]]:
        try:
            if data.fact_data is None or data.fact_data.empty:
                return {}
                
            metrics = {}
            
            # Analyze wait times
            wait_time_cols = ['NON_US_AVG_WAIT_TM_NBR', 'US_AVG_WAIT_TM_NBR', 'AVG_WAIT_TM_NBR']
            for col in wait_time_cols:
                if col in data.fact_data.columns:
                    val = data.fact_data[col].mean()
                    if pd.notna(val):
                        metrics[f'avg_{col.lower()}'] = float(val)
            
            # Analyze passenger counts
            passenger_cols = ['PSNGR_ON_BORD_QTY', 'PSNGR_ON_BORD_USC_QTY', 'PSNGR_ON_BORD_NON_USC_QTY']
            for col in passenger_cols:
                if col in data.fact_data.columns:
                    val = data.fact_data[col].sum()
                    if pd.notna(val):
                        metrics[f'total_{col.lower()}'] = float(val)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing passenger data: {e}")
            return {}
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            metrics = self.analyze_data(query.data) if query.data else {}
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.prompt_template.format(
                        query=query.query,
                        metrics=json.dumps(metrics)
                    )
                }]
            )
            
            response_data = json.loads(response['message']['content'])
            response_data['data_summary'] = metrics
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
        You are a security data analyst. Analyze this query and the provided metrics:
        Query: {query}
        Metrics: {metrics}
        
        Focus on security metrics like APIS data, referrals, and security checks.
        Return your analysis in this exact JSON format:
        {{
            "analysis": "your detailed analysis",
            "metrics": {{"metric1": value1, "metric2": value2}},
            "visualization_type": "pie"
        }}
        """
    
    def analyze_data(self, data: FlightData) -> Dict[str, Union[float, str]]:
        try:
            if data.fact_data is None or data.fact_data.empty:
                return {}
                
            metrics = {}
            
            # Analyze security metrics
            security_cols = [
                'APIS_NCIC_QTY', 'APIS_PRMRY_QTY', 'APIS_PSBL_NCIC_QTY',
                'APIS_PTIP_QTY', 'APIS_RFRL_QTY', 'APIS_CMI_QTY'
            ]
            
            for col in security_cols:
                if col in data.fact_data.columns:
                    total = data.fact_data[col].sum()
                    avg = data.fact_data[col].mean()
                    if pd.notna(total):
                        metrics[f'total_{col.lower()}'] = float(total)
                    if pd.notna(avg):
                        metrics[f'avg_{col.lower()}'] = float(avg)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing security data: {e}")
            return {}
    
    async def process(self, query: AgentQuery) -> AnalysisResponse:
        try:
            metrics = self.analyze_data(query.data) if query.data else {}
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.prompt_template.format(
                        query=query.query,
                        metrics=json.dumps(metrics)
                    )
                }]
            )
            
            response_data = json.loads(response['message']['content'])
            response_data['data_summary'] = metrics
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
        self.data = None
    
    def load_data(self, log_path: str, fact_path: str):
        """Load and prepare data for analysis"""
        self.data = DataLoader.prepare_flight_data(log_path, fact_path)
    
    async def process_query(self, query_str: str) -> MultiAgentResponse:
        query = AgentQuery(query=query_str, data=self.data)
        
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
        await asyncio.to_thread(
            ollama.chat,
            model="llama3.2:1b",
            messages=[{'role': 'user', 'content': 'test'}]
        )
        print("Model llama3.2:1b is ready")
        return True
    except Exception as e:
        print(f"Error accessing model: {e}")
        print("If model is not pulled, run: ollama pull llama3.2:1b")
        return False

async def main():
    print("Starting flight analysis system...")
    
    # Verify model is accessible
    if not await setup_model():
        return
    
    print("\nInitializing orchestrator and loading data...")
    # Initialize orchestrator and load data
    orchestrator = MultiAgentOrchestrator()
    try:
        print("Loading data from flight_log1.csv and flight_fact.csv...")
        orchestrator.load_data('flight_log1.csv', 'flight_fact.csv')
        print("Data loading complete.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
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
            print("\nResults:")
            print("Flight Analysis:", result.flight_analysis.analysis)
            print("Passenger Analysis:", result.passenger_analysis.analysis)
            print("Security Analysis:", result.security_analysis.analysis)
            print("\nMetrics:")
            if result.flight_analysis.data_summary:
                print("Flight Metrics:", json.dumps(result.flight_analysis.data_summary, indent=2))
            if result.passenger_analysis.data_summary:
                print("Passenger Metrics:", json.dumps(result.passenger_analysis.data_summary, indent=2))
            if result.security_analysis.data_summary:
                print("Security Metrics:", json.dumps(result.security_analysis.data_summary, indent=2))
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    print("Running Multi-Agent Flight Analyzer...")
    asyncio.run(main())