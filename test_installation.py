#!/usr/bin/env python3
"""
AI Meeting Coach - Installation and Functionality Tests

Run this script to verify that the system is properly installed and configured.
"""

import sys
import importlib
import json
from pathlib import Path

# Add src directory to path for imports
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test if all required dependencies can be imported"""
    print("Testing dependency imports...")
    
    required_modules = [
        ('crewai', 'CrewAI framework'),
        ('cv2', 'OpenCV for computer vision'),
        ('mediapipe', 'MediaPipe for face/pose detection'),
        ('librosa', 'Librosa for audio analysis'),
        ('numpy', 'NumPy for numerical computing'),
        ('transformers', 'Transformers for NLP'),
        ('textblob', 'TextBlob for sentiment analysis'),
        ('pandas', 'Pandas for data manipulation'),
        ('matplotlib', 'Matplotlib for plotting'),
        ('speech_recognition', 'SpeechRecognition for audio processing')
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} - {description}")
            passed += 1
        except ImportError as e:
            print(f"❌ {module_name} - {description} (Error: {e})")
            failed += 1
    
    print(f"\nImport Test Results: {passed} passed, {failed} failed")
    return failed == 0

def test_custom_tools():
    """Test if custom tools can be imported and instantiated"""
    print("\nTesting custom tools...")
    
    try:
        from meeting_agent_poc.tools import (
            AudioTranscriptionTool,
            SpeechAnalyticsTool,
            VideoFacialAnalysisTool,
            BodyLanguageAnalysisTool
        )
        
        tools = [
            (AudioTranscriptionTool, "Audio Transcription Tool"),
            (SpeechAnalyticsTool, "Speech Analytics Tool"),
            (VideoFacialAnalysisTool, "Video Facial Analysis Tool"),
            (BodyLanguageAnalysisTool, "Body Language Analysis Tool")
        ]
        
        for tool_class, description in tools:
            try:
                tool = tool_class()
                print(f"✅ {description} - Initialized successfully")
            except Exception as e:
                print(f"❌ {description} - Initialization failed: {e}")
                return False
        
        print("✅ All custom tools initialized successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import custom tools: {e}")
        return False

def test_crew_configuration():
    """Test if crew configuration files are properly formatted"""
    print("\nTesting crew configuration files...")
    
    config_files = [
        'src/meeting_agent_poc/config/agents.yaml',
        'src/meeting_agent_poc/config/tasks.yaml',
        'src/meeting_agent_poc/config/crews.yaml'
    ]
    
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            print(f"✅ {config_file} - Exists")
            # Basic file size check
            if file_path.stat().st_size > 100:
                print(f"✅ {config_file} - Has content")
            else:
                print(f"⚠️  {config_file} - File is very small, might be empty")
        else:
            print(f"❌ {config_file} - Missing")
            return False
    
    return True

def test_environment_setup():
    """Test if environment variables are properly configured"""
    print("\nTesting environment setup...")
    
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check if OPENAI_API_KEY is present (without revealing the key)
        with open(env_file, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content:
                print("✅ OPENAI_API_KEY found in .env")
            else:
                print("⚠️  OPENAI_API_KEY not found in .env - required for OpenAI integration")
        
        return True
    else:
        print("⚠️  .env file not found - create one with OPENAI_API_KEY for full functionality")
        return False

def test_crew_instantiation():
    """Test if the main crew class can be instantiated"""
    print("\nTesting crew instantiation...")
    
    try:
        from meeting_agent_poc.crew import MeetingAgentPoc
        
        print("✅ MeetingAgentPoc class imported successfully")
        
        # Test instantiation (this might fail if dependencies are missing)
        try:
            meeting_coach = MeetingAgentPoc()
            print("✅ MeetingAgentPoc instantiated successfully")
            
            # Test method existence
            methods = ['process_real_time_chunk', 'process_full_meeting', 'generate_coaching_insights']
            for method in methods:
                if hasattr(meeting_coach, method):
                    print(f"✅ Method {method} exists")
                else:
                    print(f"❌ Method {method} missing")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  MeetingAgentPoc instantiation failed: {e}")
            print("   This might be due to missing dependencies or configuration issues")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import MeetingAgentPoc: {e}")
        return False

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("AI MEETING COACH - INSTALLATION TEST REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
    print()
    
    print("Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The AI Meeting Coach is ready to use.")
        print("\nNext steps:")
        print("1. Run: python examples.py")
        print("2. Try: python -m meeting_agent_poc.main run")
        print("3. Test with real meeting files when available")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Set up .env file with OPENAI_API_KEY")
        print("3. Check that all configuration files are present")
    
    print("\n" + "="*60)

def main():
    """Run all tests and generate report"""
    print("AI MEETING COACH - INSTALLATION VERIFICATION")
    print("="*60)
    print("This script tests if the system is properly installed and configured.")
    print("="*60)
    
    # Run all tests
    test_results = {
        "Dependency Imports": test_imports(),
        "Custom Tools": test_custom_tools(),
        "Configuration Files": test_crew_configuration(),
        "Environment Setup": test_environment_setup(),
        "Crew Instantiation": test_crew_instantiation()
    }
    
    # Generate report
    generate_test_report(test_results)

if __name__ == "__main__":
    main()
