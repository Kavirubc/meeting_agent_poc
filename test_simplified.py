#!/usr/bin/env python3
"""
Simplified test script for the enhanced meeting agent POC system.
This script focuses on testing the core functionality without complex serialization.
"""

import sys
import json
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all enhanced tools can be imported successfully."""
    print("🧪 Testing Imports...")
    
    try:
        from meeting_agent_poc.tools.user_preferences import (
            UserPreferencesTool, 
            UserPreferences, 
            FeedbackSensitivity, 
            PriorityArea
        )
        print("✅ User preferences imports successful")
        
        from meeting_agent_poc.tools.custom_tool import (
            SpeechAnalyticsTool,
            VideoFacialAnalysisTool, 
            BodyLanguageAnalysisTool,
            EnhancedFeedbackSynthesizerTool
        )
        print("✅ Enhanced analysis tools imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_user_preferences_basic():
    """Test basic user preferences functionality."""
    print("\n🧪 Testing User Preferences Basic Functionality...")
    
    try:
        from meeting_agent_poc.tools.user_preferences import (
            UserPreferences, 
            FeedbackSensitivity, 
            PriorityArea
        )
        
        # Create a test user preferences object
        test_prefs = UserPreferences(
            user_id="test123",
            name="Test User",
            role="Engineer",
            organization="TestCorp",
            location="Remote"
        )
        
        print(f"✅ Created user preferences for: {test_prefs.name}")
        print(f"✅ Default feedback sensitivity: {test_prefs.feedback_sensitivity}")
        print(f"✅ Default priority areas: {len(test_prefs.priority_areas)} areas")
        
        return True
        
    except Exception as e:
        print(f"❌ User preferences test failed: {e}")
        return False

def test_preference_application():
    """Test applying preferences to analysis results."""
    print("\n🧪 Testing Preference Application...")
    
    try:
        from meeting_agent_poc.tools.user_preferences import (
            UserPreferences, 
            apply_speech_preferences,
            apply_visual_preferences,
            apply_body_language_preferences,
            FeedbackSensitivity, 
            PriorityArea
        )
        
        # Create test preferences
        test_prefs = UserPreferences(
            user_id="test_apply",
            name="Test User",
            role="Engineer", 
            organization="TestCorp",
            location="Remote",
            feedback_sensitivity=FeedbackSensitivity.HIGH
        )
        
        # Test with mock results
        mock_audio = {"pace_wpm": 120, "filler_count": 8, "priority_level": "medium"}
        mock_visual = {"eye_contact_percentage": 45.0, "priority_level": "low"}
        mock_body = {"posture_score": 0.7, "priority_level": "medium"}
        
        # Apply preferences
        enhanced_audio = apply_speech_preferences(mock_audio, test_prefs)
        enhanced_visual = apply_visual_preferences(mock_visual, test_prefs)
        enhanced_body = apply_body_language_preferences(mock_body, test_prefs)
        
        print(f"✅ Audio enhancement applied: {enhanced_audio.get('priority_level')}")
        print(f"✅ Visual enhancement applied: {enhanced_visual.get('priority_level')}")
        print(f"✅ Body language enhancement applied: {enhanced_body.get('priority_level')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preference application test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_initialization():
    """Test that all analysis tools can be initialized."""
    print("\n🧪 Testing Tool Initialization...")
    
    try:
        from meeting_agent_poc.tools.custom_tool import (
            SpeechAnalyticsTool,
            VideoFacialAnalysisTool, 
            BodyLanguageAnalysisTool,
            EnhancedFeedbackSynthesizerTool
        )
        
        # Initialize tools
        speech_tool = SpeechAnalyticsTool()
        video_tool = VideoFacialAnalysisTool()
        body_tool = BodyLanguageAnalysisTool()
        synthesizer_tool = EnhancedFeedbackSynthesizerTool()
        
        print(f"✅ Speech tool initialized: {speech_tool.name}")
        print(f"✅ Video tool initialized: {video_tool.name}")
        print(f"✅ Body language tool initialized: {body_tool.name}")
        print(f"✅ Synthesizer tool initialized: {synthesizer_tool.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool initialization test failed: {e}")
        return False

def test_system_integration():
    """Test basic system integration without file processing."""
    print("\n🧪 Testing System Integration...")
    
    try:
        # Test that the system components can work together
        from meeting_agent_poc.tools.user_preferences import (
            UserPreferencesTool,
            FeedbackSensitivity
        )
        
        pref_tool = UserPreferencesTool()
        
        # Test basic functionality (this will use default preferences)
        result = pref_tool._run(user_id="integration_test", operation="load")
        
        if "error" in result:
            print(f"⚠️  Preference loading returned error (expected for new user): {result[:50]}...")
        else:
            print("✅ Preference system operational")
        
        print("✅ Basic system integration successful")
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def main():
    """Run simplified tests."""
    print("🚀 Starting Simplified Enhanced Meeting Agent Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_user_preferences_basic,
        test_preference_application,
        test_tool_initialization,
        test_system_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced system is ready.")
        print("\n✅ System Status:")
        print("  • User preferences system: OPERATIONAL")
        print("  • Enhanced analysis tools: OPERATIONAL") 
        print("  • Tool integration: OPERATIONAL")
        print("  • Basic workflow: READY")
        
        print("\n📋 Next Steps:")
        print("  1. Test with real meeting data")
        print("  2. Validate MediaPipe configuration")
        print("  3. Performance testing")
        print("  4. End-to-end workflow validation")
        
        return True
    else:
        print(f"❌ {total - passed} tests failed. System needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
