#!/usr/bin/env python3
"""
Test script for Gemini video analysis integration.

This script demonstrates the complete video analysis pipeline with Gemini 2.0 Flash fallback.
Before running, ensure GOOGLE_API_KEY is set in your environment or .env file.

Usage:
    python test_gemini_integration.py
"""

import os
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gemini_api_setup():
    """Test if Google API key is properly configured."""
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("❌ Google API key not configured.")
        print("Please set GOOGLE_API_KEY in your .env file or environment variables.")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        return False
    
    print(f"✅ Google API key configured (length: {len(api_key)})")
    return True

def test_gemini_video_tool():
    """Test the Gemini video analysis tool directly."""
    try:
        from meeting_agent_poc.tools.gemini_video_tool import GeminiVideoAnalysisTool
        
        video_path = "knowledge/meeting/Screen Recording 2025-06-07 at 00.29.18.mov"
        
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return False
        
        print("🧪 Testing Gemini video analysis tool...")
        tool = GeminiVideoAnalysisTool()
        
        # Test comprehensive analysis
        result = tool._run(
            video_file_path=video_path,
            analysis_type="comprehensive",
            user_id="test_user"
        )
        
        result_data = json.loads(result)
        
        if result_data.get("status") == "success":
            print("✅ Gemini video analysis successful!")
            print(f"   Segments analyzed: {result_data.get('segments_analyzed', 'unknown')}")
            print(f"   Total duration: {result_data.get('total_duration', 'unknown')}s")
            print(f"   Analysis method: {result_data.get('analysis_method', 'unknown')}")
            return True
        else:
            print(f"❌ Gemini analysis failed: {result_data.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini tool test failed: {e}")
        return False

def test_facial_analysis_with_fallback():
    """Test VideoFacialAnalysisTool with Gemini fallback."""
    try:
        from meeting_agent_poc.tools.custom_tool import VideoFacialAnalysisTool
        
        video_path = "knowledge/meeting/Screen Recording 2025-06-07 at 00.29.18.mov"
        
        print("🧪 Testing VideoFacialAnalysisTool with Gemini fallback...")
        tool = VideoFacialAnalysisTool()
        
        result = tool._run(video_path, "test_user")
        result_data = json.loads(result)
        
        analysis_method = result_data.get("analysis_method", "unknown")
        
        if result_data.get("status") != "failed":
            print(f"✅ Facial analysis successful!")
            print(f"   Analysis method: {analysis_method}")
            print(f"   Overall engagement: {result_data.get('overall_engagement', 'unknown')}")
            print(f"   Attention score: {result_data.get('attention_score', 'unknown')}")
            
            if "gemini" in analysis_method:
                print("   🤖 Used Gemini AI fallback")
            elif "mediapipe" in analysis_method:
                print("   📸 Used MediaPipe computer vision")
            
            return True
        else:
            print(f"❌ Facial analysis failed: {result_data.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Facial analysis test failed: {e}")
        return False

def test_body_language_with_fallback():
    """Test BodyLanguageAnalysisTool with Gemini fallback."""
    try:
        from meeting_agent_poc.tools.custom_tool import BodyLanguageAnalysisTool
        
        video_path = "knowledge/meeting/Screen Recording 2025-06-07 at 00.29.18.mov"
        
        print("🧪 Testing BodyLanguageAnalysisTool with Gemini fallback...")
        tool = BodyLanguageAnalysisTool()
        
        result = tool._run(video_path, "test_user")
        result_data = json.loads(result)
        
        analysis_method = result_data.get("analysis_method", "unknown")
        
        if result_data.get("status") != "failed":
            print(f"✅ Body language analysis successful!")
            print(f"   Analysis method: {analysis_method}")
            print(f"   Posture assessment: {result_data.get('posture_assessment', 'unknown')}")
            print(f"   Gesture frequency: {result_data.get('gesture_frequency', 'unknown')}")
            
            if "gemini" in analysis_method:
                print("   🤖 Used Gemini AI fallback")
            elif "mediapipe" in analysis_method or analysis_method == "unknown":
                print("   📸 Used MediaPipe computer vision")
            
            return True
        else:
            print(f"❌ Body language analysis failed: {result_data.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Body language analysis test failed: {e}")
        return False

def main():
    """Run comprehensive integration tests."""
    print("🚀 Testing Gemini 2.0 Flash Video Analysis Integration")
    print("=" * 60)
    
    # Test 1: API Key Setup
    print("\n1. Testing API Configuration...")
    api_configured = test_gemini_api_setup()
    
    if not api_configured:
        print("\n⚠️  Skipping Gemini-specific tests due to missing API key.")
        print("   The integration code is ready - just add your API key to test.")
        return
    
    # Test 2: Direct Gemini Tool
    print("\n2. Testing Direct Gemini Video Tool...")
    gemini_success = test_gemini_video_tool()
    
    # Test 3: Facial Analysis with Fallback
    print("\n3. Testing Facial Analysis with Gemini Fallback...")
    facial_success = test_facial_analysis_with_fallback()
    
    # Test 4: Body Language Analysis with Fallback
    print("\n4. Testing Body Language Analysis with Gemini Fallback...")
    body_success = test_body_language_with_fallback()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Integration Test Summary:")
    print(f"   API Configuration: {'✅' if api_configured else '❌'}")
    print(f"   Gemini Video Tool: {'✅' if gemini_success else '❌'}")
    print(f"   Facial Analysis: {'✅' if facial_success else '❌'}")
    print(f"   Body Language Analysis: {'✅' if body_success else '❌'}")
    
    if all([api_configured, gemini_success, facial_success, body_success]):
        print("\n🎉 All tests passed! Gemini integration is working perfectly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
