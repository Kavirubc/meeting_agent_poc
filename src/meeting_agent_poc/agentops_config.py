#!/usr/bin/env python3
"""
AgentOps Configuration for Meeting POC
Handles AgentOps initialization and session management for the AI Meeting Coach system.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AgentOps client instance
agentops_client = None

def initialize_agentops() -> Optional[object]:
    """
    Initialize AgentOps client with proper configuration for Meeting POC.
    
    Returns:
        AgentOps client instance if successful, None otherwise
    """
    global agentops_client
    
    if agentops_client is not None:
        logger.info("AgentOps already initialized")
        return agentops_client
    
    try:
        import agentops
        
        # Get configuration from environment
        api_key = os.getenv("AGENTOPS_API_KEY")
        project_name = os.getenv("AGENTOPS_PROJECT_NAME", "Meeting POC")
        
        if not api_key or api_key == "your_agentops_api_key_here":
            logger.warning("AgentOps API key not configured. Skipping AgentOps initialization.")
            logger.info("To enable AgentOps monitoring:")
            logger.info("1. Sign up at https://app.agentops.ai/")
            logger.info("2. Get your API key from https://app.agentops.ai/settings/projects")
            logger.info("3. Add AGENTOPS_API_KEY=your_actual_key to your .env file")
            return None
        
        # Initialize AgentOps with project configuration
        agentops_client = agentops.init(
            api_key=api_key,
            default_tags=[
                "meeting-poc", 
                "crewai", 
                "communication-analysis",
                "ai-coaching"
            ],
            auto_start_session=True
        )
        
        # Set project metadata
        if hasattr(agentops_client, 'set_tags'):
            agentops_client.set_tags({
                "project": project_name,
                "version": "0.1.0",
                "environment": "development",
                "crews": ["real-time", "post-meeting", "insights"]
            })
        
        logger.info(f"✅ AgentOps initialized successfully for project: {project_name}")
        logger.info("🔍 Monitor your agents at: https://app.agentops.ai/")
        
        return agentops_client
        
    except ImportError:
        logger.error("AgentOps not installed. Install with: pip install agentops")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize AgentOps: {e}")
        return None

def start_session(session_name: str = None) -> Optional[str]:
    """
    Start a new AgentOps session for tracking.
    
    Args:
        session_name: Optional name for the session
        
    Returns:
        Session ID if successful, None otherwise
    """
    global agentops_client
    
    if agentops_client is None:
        agentops_client = initialize_agentops()
        if agentops_client is None:
            return None
    
    try:
        import agentops
        
        session_name = session_name or "Meeting Analysis Session"
        session_id = agentops.start_session(tags=[session_name])
        
        logger.info(f"🚀 Started AgentOps session: {session_name}")
        if session_id:
            logger.info(f"📊 Session ID: {session_id}")
        
        return session_id
        
    except Exception as e:
        logger.error(f"Failed to start AgentOps session: {e}")
        return None

def end_session(rating: str = "Good", feedback: str = None) -> None:
    """
    End the current AgentOps session with rating and feedback.
    
    Args:
        rating: Session rating (Indeterminate, Good, Bad)
        feedback: Optional feedback about the session
    """
    try:
        import agentops
        
        end_state = rating if rating in ["Indeterminate", "Good", "Bad"] else "Good"
        
        if feedback:
            agentops.record({
                "event_type": "session_feedback",
                "feedback": feedback,
                "rating": rating
            })
        
        agentops.end_session(end_state)
        logger.info(f"🏁 Ended AgentOps session with rating: {end_state}")
        
    except Exception as e:
        logger.error(f"Failed to end AgentOps session: {e}")

def record_event(event_type: str, data: dict) -> None:
    """
    Record a custom event in AgentOps.
    
    Args:
        event_type: Type of event to record
        data: Event data dictionary
    """
    global agentops_client
    
    if agentops_client is None:
        return
    
    try:
        import agentops
        
        event_data = {
            "event_type": event_type,
            "timestamp": agentops.helpers.get_iso_timestamp(),
            **data
        }
        
        agentops.record(event_data)
        logger.debug(f"📝 Recorded AgentOps event: {event_type}")
        
    except Exception as e:
        logger.error(f"Failed to record AgentOps event: {e}")

def track_crew_execution(crew_type: str, inputs: dict = None) -> None:
    """
    Track the start of crew execution.
    
    Args:
        crew_type: Type of crew being executed (real-time, post-meeting, insights)
        inputs: Input parameters for the crew
    """
    event_data = {
        "crew_type": crew_type,
        "inputs": inputs or {},
        "action": "crew_start"
    }
    record_event("crew_execution", event_data)

def track_crew_completion(crew_type: str, success: bool, outputs: dict = None, error: str = None) -> None:
    """
    Track the completion of crew execution.
    
    Args:
        crew_type: Type of crew that was executed
        success: Whether the execution was successful
        outputs: Output data from the crew
        error: Error message if execution failed
    """
    event_data = {
        "crew_type": crew_type,
        "success": success,
        "outputs": outputs or {},
        "error": error,
        "action": "crew_complete"
    }
    record_event("crew_execution", event_data)

def track_tool_usage(tool_name: str, inputs: dict, outputs: dict = None, error: str = None) -> None:
    """
    Track custom tool usage.
    
    Args:
        tool_name: Name of the tool being used
        inputs: Input parameters for the tool
        outputs: Output data from the tool
        error: Error message if tool execution failed
    """
    event_data = {
        "tool_name": tool_name,
        "inputs": inputs,
        "outputs": outputs or {},
        "error": error,
        "success": error is None
    }
    record_event("tool_usage", event_data)

def get_session_url() -> Optional[str]:
    """
    Get the URL for the current AgentOps session.
    
    Returns:
        Session URL if available, None otherwise
    """
    try:
        import agentops
        
        if hasattr(agentops, 'get_session_url'):
            return agentops.get_session_url()
        else:
            return "https://app.agentops.ai/"
            
    except Exception as e:
        logger.error(f"Failed to get session URL: {e}")
        return None

# Auto-initialize AgentOps when module is imported
if __name__ != "__main__":
    initialize_agentops()
