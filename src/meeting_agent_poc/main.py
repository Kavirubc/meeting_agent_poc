#!/usr/bin/env python
import sys
import warnings
import argparse
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

from .crew import MeetingAgentPoc
from .agentops_config import (
    initialize_agentops,
    start_session,
    end_session,
    get_session_url
)

# Load environment variables from .env file
load_dotenv()

# Initialize AgentOps
initialize_agentops()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the default crew (legacy).
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    try:
        MeetingAgentPoc().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def run_real_time_analysis(chunk_file_path: str):
    """
    Run the real-time analysis crew on a 15-second audio/video chunk.
    
    Args:
        chunk_file_path (str): Path to the 15-second audio/video chunk file
    """
    if not Path(chunk_file_path).exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file_path}")
    
    # Start AgentOps session
    session_id = start_session("Real-time Analysis")
    
    try:
        meeting_coach = MeetingAgentPoc()
        result = meeting_coach.process_real_time_chunk(chunk_file_path)
        print(f"Real-time analysis complete. Result: {result}")
        
        # End session with success
        end_session("Good", "Real-time analysis completed successfully")
        
        # Show AgentOps session URL
        session_url = get_session_url()
        if session_url:
            print(f"\n🔍 View detailed analytics at: {session_url}")
        
        return result
    except Exception as e:
        # End session with error
        end_session("Bad", f"Real-time analysis failed: {str(e)}")
        raise Exception(f"An error occurred during real-time analysis: {e}")

def run_post_meeting_analysis(meeting_video_path: str, meeting_id: str):
    """
    Run the post-meeting analysis crew on a full meeting recording.
    
    Args:
        meeting_video_path (str): Path to the full meeting video file
        meeting_id (str): Unique identifier for the meeting
    """
    if not Path(meeting_video_path).exists():
        raise FileNotFoundError(f"Meeting video file not found: {meeting_video_path}")
    
    # Start AgentOps session
    session_id = start_session(f"Post-meeting Analysis - {meeting_id}")
    
    try:
        meeting_coach = MeetingAgentPoc()
        result = meeting_coach.process_full_meeting(meeting_video_path, meeting_id)
        print(f"Post-meeting analysis complete. Report saved. Result: {result}")
        
        # End session with success
        end_session("Good", f"Post-meeting analysis completed successfully for meeting {meeting_id}")
        
        # Show AgentOps session URL
        session_url = get_session_url()
        if session_url:
            print(f"\n🔍 View detailed analytics at: {session_url}")
        
        return result
    except Exception as e:
        # End session with error
        end_session("Bad", f"Post-meeting analysis failed: {str(e)}")
        raise Exception(f"An error occurred during post-meeting analysis: {e}")

def run_coaching_insights(user_id: str):
    """
    Run the meeting insights crew to generate personalized coaching recommendations.
    
    Args:
        user_id (str): Unique identifier for the user
    """
    # Start AgentOps session
    session_id = start_session(f"Coaching Insights - {user_id}")
    
    try:
        meeting_coach = MeetingAgentPoc()
        result = meeting_coach.generate_coaching_insights(user_id)
        print(f"Coaching insights generated. Result: {result}")
        
        # End session with success
        end_session("Good", f"Coaching insights generated successfully for user {user_id}")
        
        # Show AgentOps session URL
        session_url = get_session_url()
        if session_url:
            print(f"\n🔍 View detailed analytics at: {session_url}")
        
        return result
    except Exception as e:
        # End session with error
        end_session("Bad", f"Coaching insights generation failed: {str(e)}")
        raise Exception(f"An error occurred during coaching insights generation: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    # Start AgentOps session
    session_id = start_session("Crew Training")
    
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        MeetingAgentPoc().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
        
        # End session with success
        end_session("Good", f"Crew training completed with {sys.argv[1]} iterations")
        
        # Show AgentOps session URL
        session_url = get_session_url()
        if session_url:
            print(f"\n🔍 View training analytics at: {session_url}")
        
    except Exception as e:
        # End session with error
        end_session("Bad", f"Crew training failed: {str(e)}")
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MeetingAgentPoc().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    # Start AgentOps session
    session_id = start_session("Crew Testing")
    
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        MeetingAgentPoc().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
        
        # End session with success
        end_session("Good", f"Crew testing completed with {sys.argv[1]} iterations using {sys.argv[2]} LLM")
        
        # Show AgentOps session URL
        session_url = get_session_url()
        if session_url:
            print(f"\n🔍 View testing analytics at: {session_url}")
        
    except Exception as e:
        # End session with error
        end_session("Bad", f"Crew testing failed: {str(e)}")
        raise Exception(f"An error occurred while testing the crew: {e}")

def main():
    """
    Main entry point with command-line argument parsing for different crew operations.
    """
    parser = argparse.ArgumentParser(description="AI Meeting Coach - CrewAI System")
    parser.add_argument("command", choices=["run", "real-time", "post-meeting", "insights", "train", "replay", "test"],
                       help="Command to execute")
    
    # Arguments for different commands
    parser.add_argument("--chunk-file", type=str, help="Path to 15-second chunk file for real-time analysis")
    parser.add_argument("--meeting-video", type=str, help="Path to full meeting video file")
    parser.add_argument("--meeting-id", type=str, help="Unique meeting identifier")
    parser.add_argument("--user-id", type=str, help="User identifier for coaching insights")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--filename", type=str, help="Training output filename")
    parser.add_argument("--task-id", type=str, help="Task ID for replay")
    parser.add_argument("--eval-llm", type=str, help="Evaluation LLM for testing")
    
    args = parser.parse_args()
    
    try:
        if args.command == "run":
            run()
        elif args.command == "real-time":
            if not args.chunk_file:
                raise ValueError("--chunk-file is required for real-time analysis")
            run_real_time_analysis(args.chunk_file)
        elif args.command == "post-meeting":
            if not args.meeting_video or not args.meeting_id:
                raise ValueError("--meeting-video and --meeting-id are required for post-meeting analysis")
            run_post_meeting_analysis(args.meeting_video, args.meeting_id)
        elif args.command == "insights":
            if not args.user_id:
                raise ValueError("--user-id is required for coaching insights")
            run_coaching_insights(args.user_id)
        elif args.command == "train":
            if not args.iterations or not args.filename:
                raise ValueError("--iterations and --filename are required for training")
            # Override sys.argv for compatibility with existing train function
            sys.argv = [sys.argv[0], str(args.iterations), args.filename]
            train()
        elif args.command == "replay":
            if not args.task_id:
                raise ValueError("--task-id is required for replay")
            sys.argv = [sys.argv[0], args.task_id]
            replay()
        elif args.command == "test":
            if not args.iterations or not args.eval_llm:
                raise ValueError("--iterations and --eval-llm are required for testing")
            sys.argv = [sys.argv[0], str(args.iterations), args.eval_llm]
            test()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
