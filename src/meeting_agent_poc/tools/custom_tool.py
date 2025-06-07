from crewai.tools import BaseTool
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
import cv2
import numpy as np
import json
import os
import librosa
import speech_recognition as sr
from textblob import TextBlob
import re
import mediapipe as mp
from collections import Counter
import time
import subprocess


class AudioTranscriptionToolInput(BaseModel):
    """Input schema for AudioTranscriptionTool."""
    audio_file_path: str = Field(..., description="Path to the audio file to transcribe")

class AudioTranscriptionTool(BaseTool):
    name: str = "AudioTranscriptionTool"
    description: str = "Transcribes audio files to text using speech recognition"
    args_schema: Type[BaseModel] = AudioTranscriptionToolInput

    def _run(self, audio_file_path: str) -> str:
        try:
            recognizer = sr.Recognizer()
            
            # Convert to WAV if needed
            if not audio_file_path.endswith('.wav'):
                converted_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], '.wav')
                # Using ffmpeg to convert audio
                subprocess.run(['ffmpeg', '-i', audio_file_path, converted_path], 
                             capture_output=True, check=True)
                audio_file_path = converted_path
            
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
            
            return json.dumps({
                "transcription": text,
                "status": "success",
                "file_path": audio_file_path
            })
        except Exception as e:
            return json.dumps({
                "transcription": "",
                "status": "error",
                "error": str(e)
            })


class SpeechAnalyticsToolInput(BaseModel):
    """Input schema for SpeechAnalyticsTool."""
    transcript: str = Field(..., description="Transcribed text from audio")
    audio_file_path: str = Field(..., description="Path to audio file")

class SpeechAnalyticsTool(BaseTool):
    name: str = "SpeechAnalyticsTool"
    description: str = "Analyzes speech patterns including pace, fillers, and vocal characteristics"
    args_schema: Type[BaseModel] = SpeechAnalyticsToolInput
    
    def _run(self, transcript: str, audio_file_path: str) -> str:
        try:
            # Load audio for duration calculation
            y, sr = librosa.load(audio_file_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate speech metrics
            results = {
                "pace_wpm": self._calculate_pace(transcript, duration),
                "filler_count": self._count_fillers(transcript),
                "volume_consistency": self._analyze_volume(y),
                "vocal_energy": self._analyze_energy(y),
                "clarity_score": self._calculate_clarity(transcript),
                "sentiment_score": self._analyze_sentiment(transcript),
                "duration_seconds": duration
            }
            
            # Generate feedback
            results["immediate_audio_feedback"] = self._generate_audio_feedback(results)
            results["priority_level"] = self._determine_priority(results)
            
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": str(e), "status": "failed"})
    
    def _calculate_pace(self, transcript: str, duration: float) -> int:
        words = len(transcript.split())
        if duration > 0:
            return int((words / duration) * 60)
        return 0
    
    def _count_fillers(self, transcript: str) -> int:
        fillers = ["um", "uh", "like", "so", "you know", "actually", "basically"]
        text_lower = transcript.lower()
        count = 0
        for filler in fillers:
            count += len(re.findall(r'\b' + filler + r'\b', text_lower))
        return count
    
    def _analyze_volume(self, audio_data: np.ndarray) -> str:
        volume_std = np.std(audio_data)
        if volume_std < 0.01:
            return "stable"
        elif volume_std < 0.05:
            return "fluctuating"
        else:
            return "highly_variable"
    
    def _analyze_energy(self, audio_data: np.ndarray) -> str:
        energy = np.mean(np.abs(audio_data))
        if energy < 0.01:
            return "low"
        elif energy < 0.05:
            return "moderate"
        elif energy < 0.1:
            return "high"
        else:
            return "excessive"
    
    def _calculate_clarity(self, transcript: str) -> float:
        if not transcript.strip():
            return 0.0
        
        # Simple clarity score based on sentence structure and word complexity
        sentences = transcript.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Penalize very short or very long sentences
        if avg_sentence_length < 5 or avg_sentence_length > 25:
            return 0.6
        else:
            return 0.8
    
    def _analyze_sentiment(self, transcript: str) -> float:
        blob = TextBlob(transcript)
        return blob.sentiment.polarity
    
    def _generate_audio_feedback(self, results: Dict) -> str:
        feedback = []
        
        if results["pace_wpm"] > 180:
            feedback.append("Slow down your speaking pace")
        elif results["pace_wpm"] < 120:
            feedback.append("Increase your speaking pace")
        
        if results["filler_count"] > 3:
            feedback.append("Reduce filler words")
        
        if results["vocal_energy"] == "low":
            feedback.append("Increase vocal energy and enthusiasm")
        
        return feedback[0] if feedback else "Good speech quality"
    
    def _determine_priority(self, results: Dict) -> str:
        if results["filler_count"] > 5 or results["pace_wpm"] > 200:
            return "critical"
        elif results["filler_count"] > 3 or results["pace_wpm"] < 100:
            return "high"
        elif results["vocal_energy"] == "low":
            return "medium"
        else:
            return "low"


class VideoFacialAnalysisToolInput(BaseModel):
    """Input schema for VideoFacialAnalysisTool."""
    video_file_path: str = Field(..., description="Path to video file for analysis")

class VideoFacialAnalysisTool(BaseTool):
    name: str = "VideoFacialAnalysisTool"
    description: str = "Analyzes facial expressions and eye contact in video"
    args_schema: Type[BaseModel] = VideoFacialAnalysisToolInput
    
    def _run(self, video_file_path: str) -> str:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            
            cap = cv2.VideoCapture(video_file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            eye_contact_frames = 0
            emotion_scores = []
            
            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
                
                frame_count = 0
                while cap.read()[0] and frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for performance (analyze every 10th frame)
                    if frame_count % 10 == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        
                        if results.multi_face_landmarks:
                            face_landmarks = results.multi_face_landmarks[0]
                            
                            # Analyze eye contact (simplified)
                            if self._detect_eye_contact(face_landmarks):
                                eye_contact_frames += 1
                            
                            # Analyze emotion (simplified)
                            emotion = self._detect_basic_emotion(face_landmarks)
                            emotion_scores.append(emotion)
                    
                    frame_count += 1
            
            cap.release()
            
            # Calculate metrics
            eye_contact_percentage = (eye_contact_frames / (frame_count / 10)) * 100 if frame_count > 0 else 0
            dominant_emotion = max(set(emotion_scores), key=emotion_scores.count) if emotion_scores else "neutral"
            
            results = {
                "eye_contact_percentage": round(eye_contact_percentage, 2),
                "dominant_emotion": dominant_emotion,
                "visual_engagement_score": self._calculate_engagement_score(eye_contact_percentage, emotion_scores),
                "total_frames_analyzed": frame_count // 10,
                "video_duration": total_frames / fps if fps > 0 else 0
            }
            
            results["immediate_visual_feedback"] = self._generate_visual_feedback(results)
            results["priority_level"] = self._determine_visual_priority(results)
            
            return json.dumps(results)
            
        except Exception as e:
            return json.dumps({"error": str(e), "status": "failed"})
    
    def _detect_eye_contact(self, face_landmarks) -> bool:
        # Simplified eye contact detection based on eye landmarks
        # This is a basic implementation - in production, you'd use more sophisticated methods
        left_eye = face_landmarks.landmark[33]  # Left eye landmark
        right_eye = face_landmarks.landmark[362]  # Right eye landmark
        
        # Simple heuristic: if eyes are looking forward (basic check)
        return abs(left_eye.x - right_eye.x) > 0.05
    
    def _detect_basic_emotion(self, face_landmarks) -> str:
        # Simplified emotion detection based on mouth landmarks
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        mouth_top = face_landmarks.landmark[13]
        mouth_bottom = face_landmarks.landmark[14]
        
        # Basic smile detection
        mouth_width = abs(mouth_right.x - mouth_left.x)
        mouth_height = abs(mouth_top.y - mouth_bottom.y)
        
        if mouth_width > 0.05 and mouth_height < 0.02:
            return "happy"
        elif mouth_height > 0.03:
            return "surprised"
        else:
            return "neutral"
    
    def _calculate_engagement_score(self, eye_contact_pct: float, emotions: List[str]) -> float:
        base_score = eye_contact_pct / 100
        
        # Boost score for positive emotions
        positive_emotions = ["happy", "engaged"]
        positive_count = sum(1 for e in emotions if e in positive_emotions)
        emotion_boost = (positive_count / len(emotions)) * 0.2 if emotions else 0
        
        return min(base_score + emotion_boost, 1.0)
    
    def _generate_visual_feedback(self, results: Dict) -> str:
        if results["eye_contact_percentage"] < 30:
            return "Maintain better eye contact with the camera"
        elif results["dominant_emotion"] in ["sad", "angry", "frustrated"]:
            return "Try to maintain a more positive facial expression"
        elif results["visual_engagement_score"] < 0.5:
            return "Increase visual engagement and facial expressiveness"
        else:
            return "Good visual presence and engagement"
    
    def _determine_visual_priority(self, results: Dict) -> str:
        if results["eye_contact_percentage"] < 20:
            return "critical"
        elif results["eye_contact_percentage"] < 40:
            return "high"
        elif results["visual_engagement_score"] < 0.5:
            return "medium"
        else:
            return "low"


class BodyLanguageAnalysisToolInput(BaseModel):
    """Input schema for BodyLanguageAnalysisTool."""
    video_file_path: str = Field(..., description="Path to video file")

class BodyLanguageAnalysisTool(BaseTool):
    name: str = "BodyLanguageAnalysisTool"
    description: str = "Analyzes body language, posture, and gestures"
    args_schema: Type[BaseModel] = BodyLanguageAnalysisToolInput
    
    def _run(self, video_file_path: str) -> str:
        try:
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            cap = cv2.VideoCapture(video_file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            posture_scores = []
            gesture_count = 0
            hand_visible_frames = 0
            
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5) as pose, \
                 mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
                
                frame_count = 0
                while cap.read()[0] and frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Analyze every 15th frame for performance
                    if frame_count % 15 == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Pose analysis
                        pose_results = pose.process(rgb_frame)
                        if pose_results.pose_landmarks:
                            posture_score = self._analyze_posture(pose_results.pose_landmarks)
                            posture_scores.append(posture_score)
                        
                        # Hand gesture analysis
                        hand_results = hands.process(rgb_frame)
                        if hand_results.multi_hand_landmarks:
                            hand_visible_frames += 1
                            gesture_count += len(hand_results.multi_hand_landmarks)
                    
                    frame_count += 1
            
            cap.release()
            
            # Calculate metrics
            avg_posture_score = np.mean(posture_scores) if posture_scores else 0.5
            gesture_frequency = gesture_count / (frame_count / 15) if frame_count > 0 else 0
            hand_visibility = (hand_visible_frames / (frame_count / 15)) * 100 if frame_count > 0 else 0
            
            results = {
                "posture_assessment": self._classify_posture(avg_posture_score),
                "gesture_frequency": round(gesture_frequency, 2),
                "hand_visibility_percentage": round(hand_visibility, 2),
                "overall_body_language_score": self._calculate_body_language_score(avg_posture_score, gesture_frequency),
                "frames_analyzed": frame_count // 15
            }
            
            results["immediate_body_language_feedback"] = self._generate_body_language_feedback(results)
            results["priority_level"] = self._determine_body_language_priority(results)
            
            return json.dumps(results)
            
        except Exception as e:
            return json.dumps({"error": str(e), "status": "failed"})
    
    def _analyze_posture(self, pose_landmarks) -> float:
        # Analyze shoulder alignment and overall posture
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        nose = pose_landmarks.landmark[0]
        
        # Calculate shoulder alignment
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # Calculate head position relative to shoulders
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        head_alignment = abs(nose.x - shoulder_center_x)
        
        # Score based on alignment (lower values = better posture)
        posture_score = 1.0 - min(shoulder_diff * 10 + head_alignment * 5, 1.0)
        return max(posture_score, 0.0)
    
    def _classify_posture(self, score: float) -> str:
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "good"
        elif score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_body_language_score(self, posture_score: float, gesture_freq: float) -> float:
        # Optimal gesture frequency is around 0.5-2.0 per frame analyzed
        gesture_score = 1.0 if 0.5 <= gesture_freq <= 2.0 else max(0.0, 1.0 - abs(gesture_freq - 1.25) * 0.4)
        
        return (posture_score * 0.7 + gesture_score * 0.3)
    
    def _generate_body_language_feedback(self, results: Dict) -> str:
        if results["posture_assessment"] == "poor":
            return "Improve your posture - sit up straight"
        elif results["gesture_frequency"] < 0.2:
            return "Use more hand gestures to enhance communication"
        elif results["gesture_frequency"] > 3.0:
            return "Reduce excessive hand movements"
        else:
            return "Good body language and posture"
    
    def _determine_body_language_priority(self, results: Dict) -> str:
        if results["posture_assessment"] == "poor":
            return "high"
        elif results["gesture_frequency"] > 4.0 or results["gesture_frequency"] < 0.1:
            return "medium"
        else:
            return "low"


# Legacy tool for compatibility
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
