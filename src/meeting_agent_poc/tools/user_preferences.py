"""
User Preferences Management System for Meeting Agent POC

This module handles user preferences for personalized feedback and analysis.
It provides functionality to load, save, and apply user-specific settings
for meeting analysis and real-time feedback.
"""

from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Import AgentOps configuration
from ..agentops_config import track_tool_usage


class FeedbackSensitivity(Enum):
    """Sensitivity levels for feedback"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


class PriorityArea(Enum):
    """Priority areas for coaching focus"""
    SPEECH_PACE = "speech_pace"
    FILLER_WORDS = "filler_words"
    EYE_CONTACT = "eye_contact"
    POSTURE = "posture"
    VOCAL_ENERGY = "vocal_energy"
    GESTURES = "gestures"
    INTERRUPTIONS = "interruptions"
    CLARITY = "clarity"
    SENTIMENT = "sentiment"
    BODY_LANGUAGE = "body_language"
    VISUAL_ENGAGEMENT = "visual_engagement"


@dataclass
class UserPreferences:
    """User preferences data structure"""
    user_id: str
    name: str
    role: str
    organization: str
    location: str
    
    # Feedback preferences
    feedback_sensitivity: FeedbackSensitivity = FeedbackSensitivity.MEDIUM
    priority_areas: List[PriorityArea] = None
    real_time_feedback_enabled: bool = True
    nudge_frequency: int = 30  # seconds between nudges
    
    # Analysis preferences
    detailed_analysis: bool = True
    include_body_language: bool = True
    include_sentiment_analysis: bool = True
    include_interruption_analysis: bool = True
    
    # Thresholds (customizable based on user goals)
    pace_thresholds: Dict[str, int] = None
    filler_thresholds: Dict[str, int] = None
    eye_contact_thresholds: Dict[str, float] = None
    energy_preferences: List[str] = None
    
    # Goals and coaching preferences
    coaching_goals: List[str] = None
    improvement_timeline: str = "3_months"
    focus_on_strengths: bool = False
    
    # Privacy and sharing preferences
    allow_recording_analysis: bool = True
    share_insights_with_team: bool = False
    anonymize_reports: bool = False
    
    # Meeting type preferences
    meeting_types: List[str] = None  # e.g., ["presentation", "team_sync", "client_call"]
    default_meeting_duration: int = 60  # minutes
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.priority_areas is None:
            self.priority_areas = [
                PriorityArea.SPEECH_PACE,
                PriorityArea.FILLER_WORDS,
                PriorityArea.EYE_CONTACT
            ]
        
        if self.pace_thresholds is None:
            self.pace_thresholds = {
                "min_wpm": 130,
                "max_wpm": 170,
                "critical_min": 100,
                "critical_max": 200
            }
        
        if self.filler_thresholds is None:
            self.filler_thresholds = {
                "low": 2,
                "medium": 4,
                "high": 6,
                "critical": 8
            }
        
        if self.eye_contact_thresholds is None:
            self.eye_contact_thresholds = {
                "minimum": 0.3,
                "good": 0.5,
                "excellent": 0.7
            }
        
        if self.energy_preferences is None:
            self.energy_preferences = ["moderate", "high"]
        
        if self.coaching_goals is None:
            self.coaching_goals = [
                "Improve speaking pace",
                "Reduce filler words",
                "Enhance eye contact"
            ]
        
        if self.meeting_types is None:
            self.meeting_types = ["team_sync", "presentation", "one_on_one"]


class UserPreferencesToolInput(BaseModel):
    """Input schema for UserPreferencesTool."""
    user_id: str = Field(..., description="User ID to load preferences for")
    operation: str = Field(..., description="Operation: 'load', 'save', 'update', or 'reset'")
    preferences_data: Optional[Dict[str, Any]] = Field(None, description="Preferences data for save/update operations")


class UserPreferencesTool(BaseTool):
    name: str = "UserPreferencesTool"
    description: str = "Manages user preferences for personalized meeting analysis and feedback"
    args_schema: Type[BaseModel] = UserPreferencesToolInput

    def _run(self, user_id: str, operation: str, preferences_data: Optional[Dict[str, Any]] = None) -> str:
        """Run the user preferences tool"""
        start_time = time.time()
        inputs = {"user_id": user_id, "operation": operation}
        
        try:
            if operation == "load":
                result = self._load_preferences(user_id)
            elif operation == "save":
                result = self._save_preferences(user_id, preferences_data)
            elif operation == "update":
                result = self._update_preferences(user_id, preferences_data)
            elif operation == "reset":
                result = self._reset_preferences(user_id)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Track successful tool usage
            track_tool_usage(
                tool_name="UserPreferencesTool",
                inputs=inputs,
                outputs={"operation": operation, "status": "success"},
                error=None
            )
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"error": str(e), "status": "failed"}
            
            # Track failed tool usage
            track_tool_usage(
                tool_name="UserPreferencesTool",
                inputs=inputs,
                outputs={"status": "failed"},
                error=str(e)
            )
            
            return json.dumps(error_result)

    def _get_preferences_path(self, user_id: str) -> Path:
        """Get the file path for user preferences"""
        preferences_dir = Path(__file__).parent.parent.parent.parent / "knowledge" / "user_preferences"
        preferences_dir.mkdir(parents=True, exist_ok=True)
        return preferences_dir / f"{user_id}_preferences.json"

    def _load_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from file"""
        preferences_path = self._get_preferences_path(user_id)
        
        if preferences_path.exists():
            try:
                with open(preferences_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to UserPreferences object for validation
                preferences = UserPreferences(**data)
                return asdict(preferences)
            except Exception as e:
                # If preferences file is corrupted, create default
                return self._create_default_preferences(user_id)
        else:
            # Create default preferences for new user
            return self._create_default_preferences(user_id)

    def _save_preferences(self, user_id: str, preferences_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save user preferences to file"""
        if not preferences_data:
            raise ValueError("No preferences data provided")
        
        # Ensure user_id is set
        preferences_data["user_id"] = user_id
        
        # Validate preferences data
        preferences = UserPreferences(**preferences_data)
        
        # Save to file
        preferences_path = self._get_preferences_path(user_id)
        with open(preferences_path, 'w') as f:
            json.dump(asdict(preferences), f, indent=2, default=str)
        
        return {
            "status": "saved",
            "user_id": user_id,
            "preferences": asdict(preferences)
        }

    def _update_preferences(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific user preferences"""
        if not updates:
            raise ValueError("No update data provided")
        
        # Load existing preferences
        current_preferences = self._load_preferences(user_id)
        
        # Update with new values
        current_preferences.update(updates)
        
        # Save updated preferences
        return self._save_preferences(user_id, current_preferences)

    def _reset_preferences(self, user_id: str) -> Dict[str, Any]:
        """Reset user preferences to defaults"""
        default_preferences = self._create_default_preferences(user_id)
        return self._save_preferences(user_id, default_preferences)

    def _create_default_preferences(self, user_id: str) -> Dict[str, Any]:
        """Create default preferences for a user"""
        # Try to load basic user info from the existing user_preference.txt
        user_info = self._load_basic_user_info()
        
        preferences = UserPreferences(
            user_id=user_id,
            name=user_info.get("name", "User"),
            role=user_info.get("role", "Professional"),
            organization=user_info.get("organization", ""),
            location=user_info.get("location", "")
        )
        
        return asdict(preferences)

    def _load_basic_user_info(self) -> Dict[str, str]:
        """Load basic user info from existing user_preference.txt"""
        try:
            user_file_path = Path(__file__).parent.parent.parent.parent / "knowledge" / "user_preference.txt"
            
            if user_file_path.exists():
                user_info = {}
                with open(user_file_path, 'r') as f:
                    content = f.read()
                    
                    # Extract basic info using simple parsing
                    if "User name is" in content:
                        name_line = [line for line in content.split('\n') if "User name is" in line][0]
                        user_info["name"] = name_line.split("User name is ")[1].rstrip('.')
                    
                    if "User is an" in content or "User is a" in content:
                        role_lines = [line for line in content.split('\n') if "User is an" in line or "User is a" in line]
                        if role_lines:
                            role_line = role_lines[0]
                            user_info["role"] = role_line.split("User is a")[1].strip().rstrip('.') if "User is a" in role_line else role_line.split("User is an ")[1].rstrip('.')
                    
                    if "based in" in content:
                        location_lines = [line for line in content.split('\n') if "based in" in line]
                        if location_lines:
                            location_line = location_lines[0]
                            user_info["location"] = location_line.split("based in ")[1].rstrip('.')
                
                return user_info
        except Exception:
            pass
        
        return {}


# Utility functions for applying preferences to analysis

def apply_speech_preferences(results: Dict[str, Any], preferences: UserPreferences) -> Dict[str, Any]:
    """Apply user preferences to speech analysis results"""
    if not preferences:
        return results
    
    # Adjust pace feedback based on user thresholds
    pace_wpm = results.get("pace_wpm", 0)
    pace_thresholds = preferences.pace_thresholds
    
    # Check for critical pace thresholds (optional)
    if "critical_min" in pace_thresholds and "critical_max" in pace_thresholds:
        if pace_wpm < pace_thresholds["critical_min"] or pace_wpm > pace_thresholds["critical_max"]:
            results["priority_level"] = "critical"
    
    # Check for standard pace thresholds
    if pace_wpm < pace_thresholds.get("min_wpm", 100) or pace_wpm > pace_thresholds.get("max_wpm", 200):
        if results.get("priority_level") != "critical":
            results["priority_level"] = "high"
    
    # Adjust filler word feedback
    filler_count = results.get("filler_count", 0)
    filler_thresholds = preferences.filler_thresholds
    
    if filler_count >= filler_thresholds.get("critical", 10):
        current_priority = results.get("priority_level", "low")
        if current_priority != "critical":
            results["priority_level"] = "critical"
    elif filler_count >= filler_thresholds.get("high", 6):
        current_priority = results.get("priority_level", "low")
        priority_order = ["low", "medium", "high", "critical"]
        if priority_order.index(current_priority) < priority_order.index("high"):
            results["priority_level"] = "high"
    
    # Customize feedback based on priority areas
    if PriorityArea.SPEECH_PACE not in preferences.priority_areas:
        # Reduce pace-related feedback priority
        if "pace" in results.get("immediate_audio_feedback", "").lower():
            results["priority_level"] = "low"
    
    return results


def apply_visual_preferences(results: Dict[str, Any], preferences: UserPreferences) -> Dict[str, Any]:
    """Apply user preferences to visual analysis results"""
    if not preferences:
        return results
    
    # Adjust eye contact feedback based on user thresholds
    eye_contact_pct = results.get("eye_contact_percentage", 0) / 100
    eye_contact_thresholds = preferences.eye_contact_thresholds
    
    if eye_contact_pct < eye_contact_thresholds["minimum"]:
        results["priority_level"] = "high"
    elif eye_contact_pct >= eye_contact_thresholds["excellent"]:
        results["priority_level"] = "low"
    
    # Customize feedback based on priority areas
    if PriorityArea.EYE_CONTACT not in preferences.priority_areas:
        # Reduce eye contact feedback priority
        if "eye contact" in results.get("immediate_visual_feedback", "").lower():
            results["priority_level"] = "low"
    
    if not preferences.include_body_language:
        # Skip detailed body language analysis
        results["simplified_analysis"] = True
    
    return results


def apply_body_language_preferences(results: Dict[str, Any], preferences: UserPreferences) -> Dict[str, Any]:
    """Apply user preferences to body language analysis results"""
    
    if not preferences:
        return results
    
    # Apply sensitivity adjustments for body language
    if preferences.feedback_sensitivity == FeedbackSensitivity.LOW:
        # Only flag serious body language issues
        if results.get("posture_assessment") not in ["poor", "very poor"]:
            results["priority_level"] = "low"
        if results.get("gesture_frequency", 1.0) < 0.1 or results.get("gesture_frequency", 1.0) > 5.0:
            results["gesture_feedback_threshold"] = 0.7
    
    elif preferences.feedback_sensitivity == FeedbackSensitivity.HIGH:
        # Be more sensitive to body language issues
        if results.get("posture_assessment") == "fair":
            results["posture_needs_attention"] = True
        if results.get("movement_consistency", 0.5) < 0.6:
            results["consistency_flag"] = True
    
    # Apply priority area preferences
    if PriorityArea.POSTURE in preferences.priority_areas:
        posture_score = results.get("posture_score", 0.5)
        if posture_score < 0.6:
            results["posture_priority_boost"] = True
    
    if PriorityArea.GESTURES in preferences.priority_areas:
        gesture_freq = results.get("gesture_frequency", 1.0)
        if gesture_freq < 0.5 or gesture_freq > 3.0:
            results["gesture_priority_boost"] = True
    
    # Apply thresholds based on coaching goals
    if "professional_presence" in preferences.coaching_goals:
        # Higher standards for professional settings
        results["professional_standards"] = True
        if results.get("overall_body_language_score", 0.5) < 0.7:
            results["professionalism_flag"] = True
    
    if "confidence_building" in preferences.coaching_goals:
        # Focus on posture and gesture confidence
        if results.get("posture_stability", 0.5) < 0.5:
            results["confidence_posture_flag"] = True
    
    # Disable body language analysis if not included in preferences
    if not preferences.include_body_language:
        # Skip detailed body language analysis
        results["simplified_analysis"] = True
    
    return results


def apply_visual_preferences(results: Dict[str, Any], preferences: UserPreferences) -> Dict[str, Any]:
    """Apply user preferences to visual analysis results"""
    
    if not preferences:
        return results
    
    # Apply sensitivity adjustments for visual analysis
    if preferences.feedback_sensitivity == FeedbackSensitivity.LOW:
        # Only flag serious visual issues
        eye_contact_pct = results.get("eye_contact_percentage", 50)
        if eye_contact_pct > 30:  # Lower threshold for low sensitivity
            results["eye_contact_priority"] = "low"
    
    elif preferences.feedback_sensitivity == FeedbackSensitivity.HIGH:
        # Be more sensitive to visual issues
        eye_contact_pct = results.get("eye_contact_percentage", 50)
        if eye_contact_pct < 60:  # Higher threshold for high sensitivity
            results["eye_contact_needs_attention"] = True
        
        engagement_score = results.get("visual_engagement_score", 0.5)
        if engagement_score < 0.7:
            results["engagement_flag"] = True
    
    # Apply priority area preferences
    if PriorityArea.EYE_CONTACT in preferences.priority_areas:
        eye_contact_pct = results.get("eye_contact_percentage", 50) / 100
        if eye_contact_pct < preferences.eye_contact_thresholds["minimum"]:
            results["eye_contact_priority_boost"] = True
    
    # Apply thresholds based on eye contact preferences
    eye_contact_pct = results.get("eye_contact_percentage", 50) / 100
    if eye_contact_pct < preferences.eye_contact_thresholds.get("minimum", 0.3):
        results["below_minimum_eye_contact"] = True
    elif eye_contact_pct > preferences.eye_contact_thresholds.get("excellent", 0.7):
        results["excellent_eye_contact"] = True
    
    # Apply coaching goals
    if "engagement_improvement" in preferences.coaching_goals:
        engagement_score = results.get("visual_engagement_score", 0.5)
        if engagement_score < 0.6:
            results["engagement_improvement_flag"] = True
    
    # Apply detailed analysis preferences
    if not preferences.detailed_analysis:
        # Skip detailed visual analysis for simplified feedback
        results["simplified_analysis"] = True
    
    return results


def prioritize_feedback_with_preferences(
    audio_results: Dict[str, Any], 
    visual_results: Dict[str, Any], 
    preferences: UserPreferences
) -> Dict[str, Any]:
    """Prioritize feedback based on user preferences and analysis results"""
    
    feedback_options = []
    
    # Audio feedback options
    if audio_results.get("priority_level") in ["high", "critical"]:
        if PriorityArea.SPEECH_PACE in preferences.priority_areas and audio_results.get("pace_wpm", 150) > preferences.pace_thresholds["max_wpm"]:
            feedback_options.append({
                "type": "pace_down",
                "message": "Slow down your speaking pace",
                "priority": 3 if audio_results.get("priority_level") == "critical" else 2,
                "confidence": 0.9
            })
        
        if PriorityArea.FILLER_WORDS in preferences.priority_areas and audio_results.get("filler_count", 0) >= preferences.filler_thresholds["medium"]:
            feedback_options.append({
                "type": "reduce_fillers",
                "message": "Reduce filler words",
                "priority": 2,
                "confidence": 0.8
            })
        
        if PriorityArea.VOCAL_ENERGY in preferences.priority_areas and audio_results.get("vocal_energy") == "low":
            feedback_options.append({
                "type": "increase_energy",
                "message": "Increase vocal energy",
                "priority": 1,
                "confidence": 0.7
            })
    
    # Visual feedback options
    if visual_results.get("priority_level") in ["high", "critical"]:
        eye_contact_pct = visual_results.get("eye_contact_percentage", 50) / 100
        if PriorityArea.EYE_CONTACT in preferences.priority_areas and eye_contact_pct < preferences.eye_contact_thresholds["minimum"]:
            feedback_options.append({
                "type": "eye_contact",
                "message": "Maintain better eye contact",
                "priority": 2,
                "confidence": 0.8
            })
    
    # Sort by priority and confidence
    feedback_options.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
    
    if feedback_options:
        top_feedback = feedback_options[0]
        return {
            "primary_nudge": top_feedback["type"],
            "nudge_message": top_feedback["message"],
            "reasoning": f"Prioritized based on user preferences and {top_feedback['priority']} priority level",
            "confidence_score": top_feedback["confidence"],
            "next_check_priority": "audio" if top_feedback["type"] in ["pace_down", "reduce_fillers", "increase_energy"] else "video"
        }
    else:
        return {
            "primary_nudge": "none",
            "nudge_message": "Good communication overall",
            "reasoning": "No critical issues detected based on user preferences",
            "confidence_score": 0.7,
            "next_check_priority": "both"
        }
