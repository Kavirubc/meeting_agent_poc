#!/usr/bin/env python3
"""
AI Meeting Coach - Example Usage Scripts

This file demonstrates how to use the three specialized crews:
1. Real-time Analysis Crew
2. Post-meeting Analysis Crew  
3. Meeting Insights Crew
"""

import os
import json
from pathlib import Path
from meeting_agent_poc.crew import MeetingAgentPoc

def example_real_time_analysis():
    """
    Example: Process a 15-second audio/video chunk for real-time feedback
    """
    print("=" * 60)
    print("EXAMPLE: Real-time Analysis")
    print("=" * 60)
    
    # Example chunk file path (you would replace this with actual file)
    chunk_file = "examples/meeting_chunk_15s.mp4"
    
    print(f"Processing 15-second chunk: {chunk_file}")
    print("This crew analyzes:")
    print("- Speech pace and filler words")
    print("- Eye contact and facial expressions")
    print("- Body language and posture")
    print("- Generates single actionable nudge")
    print()
    
    if Path(chunk_file).exists():
        try:
            meeting_coach = MeetingAgentPoc()
            result = meeting_coach.process_real_time_chunk(chunk_file)
            
            print("Real-time Analysis Result:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error during real-time analysis: {e}")
    else:
        print(f"Example file not found: {chunk_file}")
        print("To use this feature, provide a valid 15-second audio/video file.")
    
    print("\n" + "=" * 60 + "\n")

def example_post_meeting_analysis():
    """
    Example: Analyze a complete meeting recording
    """
    print("=" * 60)
    print("EXAMPLE: Post-meeting Analysis")
    print("=" * 60)
    
    # Example meeting file path
    meeting_video = "examples/full_meeting.mp4"
    meeting_id = "meeting_2024_06_07_001"
    
    print(f"Processing full meeting: {meeting_video}")
    print(f"Meeting ID: {meeting_id}")
    print("This crew provides:")
    print("- Complete transcription with speaker identification")
    print("- Communication pattern analysis")
    print("- Sentiment and engagement analysis")
    print("- Comprehensive body language analysis")
    print("- Detailed report with recommendations")
    print()
    
    if Path(meeting_video).exists():
        try:
            meeting_coach = MeetingAgentPoc()
            result = meeting_coach.process_full_meeting(meeting_video, meeting_id)
            
            print("Post-meeting Analysis Complete!")
            print(f"Report saved as: meeting_report_{meeting_id}.md")
            print(f"Transcription saved as: transcription_{meeting_id}.txt")
            
        except Exception as e:
            print(f"Error during post-meeting analysis: {e}")
    else:
        print(f"Example file not found: {meeting_video}")
        print("To use this feature, provide a valid meeting video file.")
    
    print("\n" + "=" * 60 + "\n")

def example_coaching_insights():
    """
    Example: Generate personalized coaching insights
    """
    print("=" * 60)
    print("EXAMPLE: Coaching Insights Generation")
    print("=" * 60)
    
    user_id = "john_doe_ai_engineer"
    
    print(f"Generating coaching insights for user: {user_id}")
    print("This crew analyzes:")
    print("- Historical communication patterns")
    print("- Long-term improvement trends")
    print("- Personalized coaching recommendations")
    print("- Progress tracking setup")
    print("- Goal setting and milestones")
    print()
    
    try:
        meeting_coach = MeetingAgentPoc()
        result = meeting_coach.generate_coaching_insights(user_id)
        
        print("Coaching Insights Generated!")
        print(f"Progress tracking file saved as: progress_tracking_{user_id}.json")
        
    except Exception as e:
        print(f"Error during coaching insights generation: {e}")
    
    print("\n" + "=" * 60 + "\n")

def create_sample_meeting_scenarios():
    """
    Create sample scenarios for testing the meeting coach system
    """
    print("=" * 60)
    print("SAMPLE MEETING SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Executive Standup Meeting",
            "description": "Daily standup with 5 participants, 15 minutes",
            "participants": ["CEO", "CTO", "VP Sales", "VP Marketing", "Product Manager"],
            "duration": "15 minutes",
            "focus_areas": ["Participation equity", "Meeting efficiency", "Decision making"]
        },
        {
            "name": "Client Presentation",
            "description": "Sales presentation to potential client, 45 minutes",
            "participants": ["Sales Rep", "Sales Manager", "Client Contacts"],
            "duration": "45 minutes", 
            "focus_areas": ["Presentation skills", "Engagement", "Persuasion"]
        },
        {
            "name": "Team Retrospective",
            "description": "Agile retrospective with development team, 60 minutes",
            "participants": ["Scrum Master", "Developers", "QA Engineers", "Product Owner"],
            "duration": "60 minutes",
            "focus_areas": ["Psychological safety", "Constructive feedback", "Team dynamics"]
        },
        {
            "name": "One-on-One Coaching",
            "description": "Manager coaching session with direct report, 30 minutes",
            "participants": ["Manager", "Employee"],
            "duration": "30 minutes",
            "focus_areas": ["Active listening", "Coaching skills", "Goal setting"]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Participants: {', '.join(scenario['participants'])}")
        print(f"   Duration: {scenario['duration']}")
        print(f"   Focus Areas: {', '.join(scenario['focus_areas'])}")
        print()
    
    print("To test with these scenarios:")
    print("1. Record or simulate meetings matching these descriptions")
    print("2. Use the appropriate crew based on your analysis needs")
    print("3. Review the generated insights and recommendations")
    print("\n" + "=" * 60 + "\n")

def display_usage_instructions():
    """
    Display comprehensive usage instructions
    """
    print("AI MEETING COACH - USAGE INSTRUCTIONS")
    print("=" * 60)
    print()
    
    print("COMMAND LINE USAGE:")
    print("------------------")
    print("# Real-time analysis (15-second chunks)")
    print("python -m meeting_agent_poc.main real-time --chunk-file path/to/chunk.mp4")
    print()
    print("# Post-meeting analysis (full recordings)")
    print("python -m meeting_agent_poc.main post-meeting --meeting-video path/to/meeting.mp4 --meeting-id meeting_001")
    print()
    print("# Coaching insights generation")
    print("python -m meeting_agent_poc.main insights --user-id john_doe")
    print()
    print("# Legacy crew (research and reporting)")
    print("python -m meeting_agent_poc.main run")
    print()
    
    print("PROGRAMMATIC USAGE:")
    print("------------------")
    print("from meeting_agent_poc.crew import MeetingAgentPoc")
    print()
    print("coach = MeetingAgentPoc()")
    print()
    print("# Real-time analysis")
    print("result = coach.process_real_time_chunk('chunk.mp4')")
    print()
    print("# Post-meeting analysis")
    print("result = coach.process_full_meeting('meeting.mp4', 'meeting_001')")
    print()
    print("# Coaching insights")
    print("result = coach.generate_coaching_insights('user_001')")
    print()
    
    print("SUPPORTED FILE FORMATS:")
    print("----------------------")
    print("Video: .mp4, .avi, .mov, .mkv")
    print("Audio: .wav, .mp3, .m4a, .flac")
    print()
    
    print("OUTPUT FILES:")
    print("-------------")
    print("- Real-time: JSON feedback in console")
    print("- Post-meeting: meeting_report_{meeting_id}.md")
    print("- Post-meeting: transcription_{meeting_id}.txt")
    print("- Insights: progress_tracking_{user_id}.json")
    print()

def main():
    """
    Run all examples and display usage instructions
    """
    print("AI MEETING COACH - EXAMPLE USAGE")
    print("CrewAI-based Multi-Agent System for Meeting Analysis")
    print("=" * 60)
    print()
    
    # Display usage instructions
    display_usage_instructions()
    
    # Show sample scenarios
    create_sample_meeting_scenarios()
    
    # Run examples (commented out to avoid errors with missing files)
    print("RUNNING EXAMPLES (uncomment in code to test with actual files):")
    print("--------------------------------------------------------------")
    print("# example_real_time_analysis()")
    print("# example_post_meeting_analysis()")
    print("# example_coaching_insights()")
    print()
    
    print("To run examples with actual files, uncomment the function calls above")
    print("and provide valid audio/video files in the 'examples/' directory.")

if __name__ == "__main__":
    main()
