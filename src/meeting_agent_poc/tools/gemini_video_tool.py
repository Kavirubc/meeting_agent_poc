"""
Gemini 2.0 Flash Video Analysis Tool

This module implements video analysis using Google's Gemini 2.0 Flash model
to provide authentic behavioral insights from meeting videos instead of
relying on computer vision fallbacks that may generate synthetic data.
"""

import os
import json
import time
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import cv2
import numpy as np

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from pydantic import BaseModel, Field
from crewai_tools import BaseTool

# Import user preferences if available
try:
    from .user_preferences import UserPreferencesTool, USER_PREFERENCES_AVAILABLE
    if USER_PREFERENCES_AVAILABLE:
        from .user_preferences import apply_visual_preferences
except ImportError:
    USER_PREFERENCES_AVAILABLE = False

# Import tracking
try:
    from ..agentops_config import track_tool_usage
except ImportError:
    def track_tool_usage(*args, **kwargs):
        pass


class GeminiVideoAnalysisToolInput(BaseModel):
    """Input schema for GeminiVideoAnalysisTool."""
    video_file_path: str = Field(..., description="Path to video file for analysis")
    user_id: Optional[str] = Field(None, description="User ID for personalized feedback")
    analysis_type: str = Field(
        default="comprehensive", 
        description="Type of analysis: 'facial', 'body_language', 'comprehensive'"
    )


class GeminiVideoAnalysisTool(BaseTool):
    name: str = "GeminiVideoAnalysisTool"
    description: str = "AI-powered video analysis using Gemini 2.0 Flash for authentic behavioral insights from meeting videos"
    args_schema: type[BaseModel] = GeminiVideoAnalysisToolInput
    
    # Declare the gemini_model as a model field
    gemini_model: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self):
        super().__init__()
        self._setup_gemini()

    def _setup_gemini(self):
        """Initialize Gemini AI with API key from environment"""
        if not GEMINI_AVAILABLE:
            print("Warning: google-generativeai not available. Install with: pip install google-generativeai")
            self.gemini_model = None
            return False
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key == 'YOUR_GOOGLE_API_KEY_HERE':
            print("Warning: GOOGLE_API_KEY not configured in environment")
            self.gemini_model = None
            return False
        
        try:
            genai.configure(api_key=api_key)
            # Use the available Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')  # Use stable version
            self.gemini_model = model
            return True
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self.gemini_model = None
            return False

    def _run(self, video_file_path: str, user_id: Optional[str] = None, analysis_type: str = "comprehensive") -> str:
        """Main execution method for video analysis"""
        start_time = time.time()
        inputs = {"video_file_path": video_file_path, "user_id": user_id, "analysis_type": analysis_type}
        
        try:
            if not GEMINI_AVAILABLE:
                return self._fallback_analysis(video_file_path, "gemini_unavailable", user_id)
            
            if not hasattr(self, 'gemini_model') or self.gemini_model is None:
                return self._fallback_analysis(video_file_path, "gemini_not_configured", user_id)
            
            # Get video properties
            video_info = self._get_video_info(video_file_path)
            if not video_info:
                return self._fallback_analysis(video_file_path, "video_unreadable", user_id)
            
            # Create video segments for analysis
            segments = self._create_video_segments(video_file_path, video_info)
            if not segments:
                return self._fallback_analysis(video_file_path, "segmentation_failed", user_id)
            
            # Analyze segments with Gemini
            analysis_results = self._analyze_segments_with_gemini(segments, analysis_type)
            
            # Synthesize results
            final_results = self._synthesize_analysis_results(
                analysis_results, video_info, analysis_type, user_id
            )
            
            # Clean up temporary files
            self._cleanup_segments(segments)
            
            # Track successful usage
            track_tool_usage(
                tool_name="GeminiVideoAnalysisTool",
                inputs=inputs,
                outputs={
                    "analysis_type": analysis_type,
                    "segments_analyzed": len(segments),
                    "video_duration": video_info.get("duration", 0),
                    "status": "success"
                },
                error=None
            )
            
            return json.dumps(final_results)
            
        except Exception as e:
            error_result = self._fallback_analysis(video_file_path, f"analysis_error: {str(e)}", user_id)
            
            # Track failed usage
            track_tool_usage(
                tool_name="GeminiVideoAnalysisTool",
                inputs=inputs,
                outputs={"status": "failed"},
                error=str(e)
            )
            
            return error_result

    def _get_video_info(self, video_file_path: str) -> Optional[Dict]:
        """Extract basic video information"""
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "file_size": os.path.getsize(video_file_path)
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None

    def _create_video_segments(self, video_file_path: str, video_info: Dict) -> List[Dict]:
        """Create video segments for analysis (10-15 second chunks)"""
        segments = []
        duration = video_info["duration"]
        segment_duration = 15  # seconds
        
        if duration < 10:
            # For very short videos, analyze the whole thing
            segments.append({
                "start_time": 0,
                "duration": duration,
                "file_path": video_file_path,
                "temp_file": False
            })
            return segments
        
        # Create segments
        num_segments = min(8, int(duration / segment_duration) + 1)  # Max 8 segments for cost control
        actual_segment_duration = duration / num_segments
        
        try:
            for i in range(num_segments):
                start_time = i * actual_segment_duration
                end_time = min((i + 1) * actual_segment_duration, duration)
                
                # Create temporary segment file
                temp_file = self._extract_video_segment(
                    video_file_path, start_time, end_time - start_time
                )
                
                if temp_file:
                    segments.append({
                        "start_time": start_time,
                        "duration": end_time - start_time,
                        "file_path": temp_file,
                        "temp_file": True,
                        "segment_index": i
                    })
        
        except Exception as e:
            print(f"Error creating segments: {e}")
            # Fallback to whole video
            segments = [{
                "start_time": 0,
                "duration": duration,
                "file_path": video_file_path,
                "temp_file": False
            }]
        
        return segments

    def _extract_video_segment(self, video_file_path: str, start_time: float, duration: float) -> Optional[str]:
        """Extract a video segment using ffmpeg"""
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"meeting_segment_{int(start_time)}_{int(time.time())}.mp4")
            
            # Use ffmpeg to extract segment
            import subprocess
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite
                "-i", video_file_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                temp_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(temp_file):
                return temp_file
            else:
                print(f"FFmpeg failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None

    def _analyze_segments_with_gemini(self, segments: List[Dict], analysis_type: str) -> List[Dict]:
        """Analyze video segments using Gemini 2.0 Flash"""
        results = []
        
        for segment in segments:
            try:
                # Upload video file to Gemini
                video_file = genai.upload_file(
                    path=segment["file_path"],
                    mime_type="video/mp4"
                )
                
                # Wait for processing
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    print(f"Video processing failed for segment {segment.get('segment_index', 0)}")
                    continue
                
                # Create analysis prompt based on type
                prompt = self._create_analysis_prompt(analysis_type)
                
                # Generate analysis
                response = self.gemini_model.generate_content(
                    [video_file, prompt],
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                # Parse response
                analysis_data = self._parse_gemini_response(response.text, segment)
                results.append(analysis_data)
                
                # Clean up uploaded file
                genai.delete_file(video_file.name)
                
            except Exception as e:
                print(f"Error analyzing segment {segment.get('segment_index', 0)}: {e}")
                results.append({
                    "segment_index": segment.get("segment_index", 0),
                    "start_time": segment["start_time"],
                    "error": str(e),
                    "analysis_status": "failed"
                })
        
        return results

    def _create_analysis_prompt(self, analysis_type: str) -> str:
        """Create appropriate analysis prompt based on type"""
        base_prompt = """
        You are analyzing a video segment from a business meeting. Please provide a detailed behavioral analysis focusing on communication effectiveness.

        Analyze the following aspects and provide your assessment in JSON format:
        """
        
        if analysis_type == "facial":
            return base_prompt + """
            Focus on facial expressions and eye contact:
            {
                "eye_contact": {
                    "level": "low|moderate|high",
                    "consistency": "inconsistent|somewhat_consistent|very_consistent",
                    "estimated_percentage": 0-100,
                    "notes": "detailed observations"
                },
                "facial_expressions": {
                    "dominant_emotion": "neutral|happy|focused|concerned|confused|engaged",
                    "emotion_changes": ["list of observed emotions"],
                    "expressiveness": "low|moderate|high",
                    "authenticity": "natural|somewhat_stiff|very_animated"
                },
                "engagement_indicators": {
                    "visual_attention": "distracted|somewhat_focused|highly_focused",
                    "responsiveness": "minimal|moderate|high",
                    "overall_engagement": 1-10
                }
            }
            """
        
        elif analysis_type == "body_language":
            return base_prompt + """
            Focus on posture, gestures, and body language:
            {
                "posture": {
                    "alignment": "poor|fair|good|excellent",
                    "stability": "fidgety|somewhat_stable|very_stable",
                    "energy_level": "low|moderate|high",
                    "professionalism": 1-10
                },
                "gestures": {
                    "frequency": "rare|occasional|frequent|excessive",
                    "variety": "limited|moderate|diverse",
                    "effectiveness": "distracting|neutral|enhancing",
                    "naturalness": 1-10
                },
                "movement_patterns": {
                    "head_movement": "minimal|moderate|excessive",
                    "hand_visibility": "hidden|partially_visible|clearly_visible",
                    "overall_animation": "static|moderately_animated|highly_animated"
                }
            }
            """
        
        else:  # comprehensive
            return base_prompt + """
            Provide a comprehensive analysis covering all behavioral aspects:
            {
                "overall_presence": {
                    "confidence": 1-10,
                    "professionalism": 1-10,
                    "approachability": 1-10,
                    "energy": 1-10
                },
                "communication_effectiveness": {
                    "visual_engagement": 1-10,
                    "non_verbal_clarity": 1-10,
                    "overall_impact": 1-10
                },
                "specific_observations": {
                    "strengths": ["list of positive behaviors"],
                    "areas_for_improvement": ["list of areas to work on"],
                    "notable_moments": ["specific time-stamped observations"]
                },
                "behavioral_summary": "2-3 sentence summary of overall behavioral patterns"
            }
            """

    def _parse_gemini_response(self, response_text: str, segment: Dict) -> Dict:
        """Parse Gemini's JSON response and add metadata"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis_data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                analysis_data = {
                    "raw_response": response_text,
                    "parsing_error": "No valid JSON found in response"
                }
            
            # Add metadata
            analysis_data.update({
                "segment_index": segment.get("segment_index", 0),
                "start_time": segment["start_time"],
                "duration": segment["duration"],
                "analysis_method": "gemini_2_0_flash",
                "analysis_status": "success",
                "timestamp": time.time()
            })
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            return {
                "segment_index": segment.get("segment_index", 0),
                "start_time": segment["start_time"],
                "raw_response": response_text,
                "parsing_error": f"JSON decode error: {str(e)}",
                "analysis_status": "parse_failed"
            }

    def _synthesize_analysis_results(self, analysis_results: List[Dict], video_info: Dict, 
                                   analysis_type: str, user_id: Optional[str]) -> Dict:
        """Synthesize individual segment analyses into comprehensive results"""
        
        successful_analyses = [r for r in analysis_results if r.get("analysis_status") == "success"]
        
        if not successful_analyses:
            return self._create_failed_synthesis(analysis_results, video_info, analysis_type)
        
        # Extract and aggregate insights based on analysis type
        if analysis_type == "facial":
            return self._synthesize_facial_analysis(successful_analyses, video_info, user_id)
        elif analysis_type == "body_language":
            return self._synthesize_body_language_analysis(successful_analyses, video_info, user_id)
        else:
            return self._synthesize_comprehensive_analysis(successful_analyses, video_info, user_id)

    def _synthesize_facial_analysis(self, analyses: List[Dict], video_info: Dict, user_id: Optional[str]) -> Dict:
        """Synthesize facial analysis results"""
        
        # Extract eye contact data
        eye_contact_estimates = []
        dominant_emotions = []
        engagement_scores = []
        
        for analysis in analyses:
            if "eye_contact" in analysis:
                ec_data = analysis["eye_contact"]
                if "estimated_percentage" in ec_data:
                    eye_contact_estimates.append(ec_data["estimated_percentage"])
            
            if "facial_expressions" in analysis:
                fe_data = analysis["facial_expressions"]
                if "dominant_emotion" in fe_data:
                    dominant_emotions.append(fe_data["dominant_emotion"])
            
            if "engagement_indicators" in analysis:
                ei_data = analysis["engagement_indicators"]
                if "overall_engagement" in ei_data:
                    engagement_scores.append(ei_data["overall_engagement"])
        
        # Calculate aggregated metrics
        avg_eye_contact = np.mean(eye_contact_estimates) if eye_contact_estimates else None
        most_common_emotion = max(set(dominant_emotions), key=dominant_emotions.count) if dominant_emotions else "neutral"
        avg_engagement = np.mean(engagement_scores) if engagement_scores else None
        
        # Create emotion distribution
        emotion_dist = {}
        if dominant_emotions:
            for emotion in set(dominant_emotions):
                emotion_dist[emotion] = dominant_emotions.count(emotion) / len(dominant_emotions)
        
        results = {
            "status": "success",  # Add standard status field
            "analysis_status": "SUCCESS",
            "analysis_method": "gemini_2_0_flash_facial",
            "data_authenticity": "AI_GENERATED_AUTHENTIC",
            
            "eye_contact_percentage": round(avg_eye_contact, 2) if avg_eye_contact is not None else None,
            "dominant_emotion": most_common_emotion,
            "emotion_distribution": emotion_dist,
            "visual_engagement_score": round(avg_engagement / 10, 2) if avg_engagement is not None else None,
            
            "video_duration": video_info["duration"],
            "segments_analyzed": len(analyses),
            "confidence_level": min(0.95, len(analyses) / 8),  # Higher confidence with more segments
            
            "detailed_insights": self._extract_detailed_insights(analyses, "facial"),
            "immediate_visual_feedback": self._generate_ai_feedback(analyses, "facial"),
            "priority_level": self._determine_ai_priority(analyses, "facial"),
            "actionable_visual_suggestions": self._generate_ai_suggestions(analyses, "facial")
        }
        
        # Apply user preferences if available
        if USER_PREFERENCES_AVAILABLE and user_id:
            try:
                prefs_tool = UserPreferencesTool()
                prefs_result = prefs_tool._run(user_id, "load")
                prefs_data = json.loads(prefs_result)
                if prefs_data.get("status") != "failed":
                    results = apply_visual_preferences(results, prefs_data)
            except Exception:
                pass
        
        return results

    def _synthesize_body_language_analysis(self, analyses: List[Dict], video_info: Dict, user_id: Optional[str]) -> Dict:
        """Synthesize body language analysis results"""
        
        posture_scores = []
        gesture_effectiveness = []
        professionalism_scores = []
        
        for analysis in analyses:
            if "posture" in analysis:
                p_data = analysis["posture"]
                if "professionalism" in p_data:
                    professionalism_scores.append(p_data["professionalism"])
            
            if "gestures" in analysis:
                g_data = analysis["gestures"]
                if "naturalness" in g_data:
                    gesture_effectiveness.append(g_data["naturalness"])
        
        avg_professionalism = np.mean(professionalism_scores) if professionalism_scores else None
        avg_gesture_effectiveness = np.mean(gesture_effectiveness) if gesture_effectiveness else None
        
        results = {
            "status": "success",  # Add standard status field
            "analysis_status": "SUCCESS",
            "analysis_method": "gemini_2_0_flash_body_language",
            "data_authenticity": "AI_GENERATED_AUTHENTIC",
            
            "posture_assessment": self._classify_posture_from_score(avg_professionalism),
            "posture_score": round(avg_professionalism / 10, 3) if avg_professionalism is not None else None,
            "gesture_effectiveness_score": round(avg_gesture_effectiveness / 10, 3) if avg_gesture_effectiveness is not None else None,
            "overall_body_language_score": self._calculate_overall_body_score(avg_professionalism, avg_gesture_effectiveness),
            
            "video_duration": video_info["duration"],
            "segments_analyzed": len(analyses),
            "confidence_level": min(0.95, len(analyses) / 8),
            
            "detailed_insights": self._extract_detailed_insights(analyses, "body_language"),
            "immediate_body_language_feedback": self._generate_ai_feedback(analyses, "body_language"),
            "priority_level": self._determine_ai_priority(analyses, "body_language"),
            "actionable_body_language_suggestions": self._generate_ai_suggestions(analyses, "body_language")
        }
        
        return results

    def _synthesize_comprehensive_analysis(self, analyses: List[Dict], video_info: Dict, user_id: Optional[str]) -> Dict:
        """Synthesize comprehensive analysis results"""
        
        confidence_scores = []
        professionalism_scores = []
        engagement_scores = []
        
        strengths = []
        improvements = []
        
        for analysis in analyses:
            if "overall_presence" in analysis:
                op_data = analysis["overall_presence"]
                if "confidence" in op_data:
                    confidence_scores.append(op_data["confidence"])
                if "professionalism" in op_data:
                    professionalism_scores.append(op_data["professionalism"])
            
            if "communication_effectiveness" in analysis:
                ce_data = analysis["communication_effectiveness"]
                if "visual_engagement" in ce_data:
                    engagement_scores.append(ce_data["visual_engagement"])
            
            if "specific_observations" in analysis:
                so_data = analysis["specific_observations"]
                if "strengths" in so_data:
                    strengths.extend(so_data["strengths"])
                if "areas_for_improvement" in so_data:
                    improvements.extend(so_data["areas_for_improvement"])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else None
        avg_professionalism = np.mean(professionalism_scores) if professionalism_scores else None
        avg_engagement = np.mean(engagement_scores) if engagement_scores else None
        
        results = {
            "status": "success",  # Add standard status field
            "analysis_status": "SUCCESS", 
            "analysis_method": "gemini_2_0_flash_comprehensive",
            "data_authenticity": "AI_GENERATED_AUTHENTIC",
            
            "overall_confidence_score": round(avg_confidence / 10, 2) if avg_confidence is not None else None,
            "overall_professionalism_score": round(avg_professionalism / 10, 2) if avg_professionalism is not None else None,
            "overall_engagement_score": round(avg_engagement / 10, 2) if avg_engagement is not None else None,
            
            "key_strengths": list(set(strengths)),
            "improvement_areas": list(set(improvements)),
            
            "video_duration": video_info["duration"],
            "segments_analyzed": len(analyses),
            "confidence_level": min(0.95, len(analyses) / 8),
            
            "detailed_insights": self._extract_detailed_insights(analyses, "comprehensive"),
            "immediate_feedback": self._generate_ai_feedback(analyses, "comprehensive"),
            "priority_level": self._determine_ai_priority(analyses, "comprehensive"),
            "actionable_suggestions": self._generate_ai_suggestions(analyses, "comprehensive")
        }
        
        return results

    def _extract_detailed_insights(self, analyses: List[Dict], analysis_type: str) -> List[str]:
        """Extract detailed insights from all analyses"""
        insights = []
        
        for i, analysis in enumerate(analyses):
            segment_insights = []
            start_time = analysis.get("start_time", 0)
            
            if analysis_type == "facial" and "facial_expressions" in analysis:
                fe_data = analysis["facial_expressions"]
                if "notes" in fe_data:
                    segment_insights.append(f"[{start_time:.1f}s] {fe_data['notes']}")
            
            elif analysis_type == "body_language" and "posture" in analysis:
                # Extract relevant body language insights
                for key, value in analysis.items():
                    if isinstance(value, dict) and "notes" in value:
                        segment_insights.append(f"[{start_time:.1f}s] {value['notes']}")
            
            elif analysis_type == "comprehensive" and "behavioral_summary" in analysis:
                segment_insights.append(f"[{start_time:.1f}s] {analysis['behavioral_summary']}")
            
            insights.extend(segment_insights)
        
        return insights[:10]  # Return top 10 insights

    def _generate_ai_feedback(self, analyses: List[Dict], analysis_type: str) -> str:
        """Generate actionable feedback based on AI analysis"""
        
        if not analyses:
            return "Unable to generate feedback - no successful analysis segments"
        
        # Aggregate common themes
        all_feedback = []
        
        for analysis in analyses:
            if analysis_type == "facial":
                if "eye_contact" in analysis:
                    ec_data = analysis["eye_contact"]
                    level = ec_data.get("level", "moderate")
                    if level == "low":
                        all_feedback.append("Increase eye contact with camera")
                    elif level == "high":
                        all_feedback.append("Excellent eye contact maintained")
            
            elif analysis_type == "body_language":
                if "posture" in analysis:
                    p_data = analysis["posture"]
                    alignment = p_data.get("alignment", "fair")
                    if alignment in ["poor", "fair"]:
                        all_feedback.append("Improve posture and body alignment")
            
            elif analysis_type == "comprehensive":
                if "overall_presence" in analysis:
                    op_data = analysis["overall_presence"]
                    confidence = op_data.get("confidence", 5)
                    if confidence < 6:
                        all_feedback.append("Work on projecting more confidence")
        
        # Return most common feedback
        if all_feedback:
            most_common = max(set(all_feedback), key=all_feedback.count)
            return most_common
        
        return "Continue maintaining good communication presence"

    def _determine_ai_priority(self, analyses: List[Dict], analysis_type: str) -> str:
        """Determine priority level based on AI analysis"""
        
        high_priority_indicators = 0
        total_indicators = 0
        
        for analysis in analyses:
            if analysis_type == "facial":
                if "eye_contact" in analysis:
                    level = analysis["eye_contact"].get("level", "moderate")
                    if level == "low":
                        high_priority_indicators += 1
                    total_indicators += 1
                
                if "engagement_indicators" in analysis:
                    engagement = analysis["engagement_indicators"].get("overall_engagement", 5)
                    if engagement < 5:
                        high_priority_indicators += 1
                    total_indicators += 1
            
            elif analysis_type == "body_language":
                if "posture" in analysis:
                    prof_score = analysis["posture"].get("professionalism", 5)
                    if prof_score < 5:
                        high_priority_indicators += 1
                    total_indicators += 1
            
            elif analysis_type == "comprehensive":
                if "overall_presence" in analysis:
                    confidence = analysis["overall_presence"].get("confidence", 5)
                    if confidence < 5:
                        high_priority_indicators += 1
                    total_indicators += 1
        
        if total_indicators == 0:
            return "medium"
        
        priority_ratio = high_priority_indicators / total_indicators
        
        if priority_ratio > 0.6:
            return "high"
        elif priority_ratio > 0.3:
            return "medium"
        else:
            return "low"

    def _generate_ai_suggestions(self, analyses: List[Dict], analysis_type: str) -> List[str]:
        """Generate actionable suggestions based on AI analysis"""
        
        suggestions = []
        
        # Aggregate suggestions from all analyses
        for analysis in analyses:
            if "specific_observations" in analysis:
                improvements = analysis["specific_observations"].get("areas_for_improvement", [])
                suggestions.extend(improvements)
        
        # Return unique suggestions, limited to top 5
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:5]

    def _classify_posture_from_score(self, score: Optional[float]) -> str:
        """Classify posture based on numeric score"""
        if score is None:
            return "unknown"
        
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good"
        elif score >= 4:
            return "fair"
        else:
            return "poor"

    def _calculate_overall_body_score(self, professionalism: Optional[float], gesture_effectiveness: Optional[float]) -> Optional[float]:
        """Calculate overall body language score"""
        scores = [s for s in [professionalism, gesture_effectiveness] if s is not None]
        
        if not scores:
            return None
        
        return round(np.mean(scores) / 10, 3)

    def _create_failed_synthesis(self, analysis_results: List[Dict], video_info: Dict, analysis_type: str) -> Dict:
        """Create synthesis result when all analyses failed"""
        
        return {
            "analysis_status": "FAILED",
            "error_type": "all_segments_failed",
            "analysis_method": "gemini_2_0_flash_failed",
            "data_authenticity": "NO_ANALYSIS_PERFORMED",
            
            "video_duration": video_info["duration"],
            "segments_attempted": len(analysis_results),
            "segments_successful": 0,
            "confidence_level": 0.0,
            
            "immediate_feedback": "Unable to analyze video - all segments failed processing",
            "priority_level": "analysis_failed",
            "actionable_suggestions": [
                "Check video file format and quality",
                "Ensure good lighting and clear visibility",
                "Verify Gemini API configuration"
            ],
            
            "error_details": [r.get("error", "Unknown error") for r in analysis_results],
            "limitations": [
                "No behavioral analysis was performed",
                "Results are not available due to processing failures",
                "Manual review of video may be necessary"
            ]
        }

    def _cleanup_segments(self, segments: List[Dict]):
        """Clean up temporary video segment files"""
        for segment in segments:
            if segment.get("temp_file", False):
                try:
                    file_path = segment["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {segment.get('file_path')}: {e}")

    def _fallback_analysis(self, video_file_path: str, error_reason: str, user_id: Optional[str]) -> str:
        """Fallback analysis when Gemini is not available"""
        
        try:
            video_info = self._get_video_info(video_file_path)
            duration = video_info["duration"] if video_info else 0
        except:
            duration = 0
        
        results = {
            "analysis_status": "FAILED",
            "error_type": error_reason,
            "analysis_method": "gemini_fallback",
            "data_authenticity": "NO_ANALYSIS_PERFORMED",
            
            "video_duration": duration,
            "confidence_level": 0.0,
            
            "immediate_feedback": f"Video analysis unavailable: {error_reason}",
            "priority_level": "analysis_unavailable",
            "actionable_suggestions": [
                "Configure GOOGLE_API_KEY environment variable",
                "Install google-generativeai package", 
                "Ensure video file is accessible and valid"
            ],
            
            "limitations": [
                "No AI-powered video analysis performed",
                "Gemini 2.0 Flash API not available or configured",
                "Consider using alternative analysis methods"
            ],
            "required_setup": [
                "Set GOOGLE_API_KEY in environment variables",
                "Install: pip install google-generativeai",
                "Ensure ffmpeg is available for video processing"
            ]
        }
        
        return json.dumps(results)
