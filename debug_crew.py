#!/usr/bin/env python3
"""
Isolated test for crew instantiation debugging
"""
import sys
import os
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

def test_crew_components():
    print("Testing individual crew components...")
    
    try:
        # Test tool imports
        print("1. Testing tool imports...")
        from meeting_agent_poc.tools import (
            AudioTranscriptionTool,
            SpeechAnalyticsTool,
            VideoFacialAnalysisTool,
            BodyLanguageAnalysisTool
        )
        print("   ✅ Tools imported successfully")
        
        # Test tool instantiation
        print("2. Testing tool instantiation...")
        audio_tool = AudioTranscriptionTool()
        speech_tool = SpeechAnalyticsTool()
        video_tool = VideoFacialAnalysisTool()
        body_tool = BodyLanguageAnalysisTool()
        print("   ✅ Tools instantiated successfully")
        
        # Test crew class import
        print("3. Testing crew class import...")
        from meeting_agent_poc.crew import MeetingAgentPoc
        print("   ✅ MeetingAgentPoc imported successfully")
        
        # Test crew instantiation with detailed error reporting
        print("4. Testing crew instantiation...")
        try:
            meeting_poc = MeetingAgentPoc()
            print("   ✅ MeetingAgentPoc instantiated successfully")
            
            # Test accessing crew
            print("5. Testing crew creation...")
            crew = meeting_poc.crew()
            print("   ✅ Crew created successfully")
            
        except Exception as e:
            print(f"   ❌ Crew instantiation failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print("   Full traceback:")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("CREW COMPONENT DEBUGGING")
    print("=" * 50)
    
    success = test_crew_components()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All crew components working properly!")
    else:
        print("❌ Crew component issues detected")
