# AI Meeting Coach - CrewAI Multi-Agent System

Welcome to the AI Meeting Coach, a comprehensive multi-agent system powered by [CrewAI](https://crewai.com) designed to revolutionize meeting analysis and communication coaching. This system provides real-time feedback, comprehensive post-meeting analysis, and personalized coaching insights using three specialized AI crews.

## 🎯 System Overview

The AI Meeting Coach consists of three specialized crews, each optimized for different aspects of meeting analysis:

### 1. **Real-time Analysis Crew** ⚡

- **Purpose**: Processes 15-second audio/video chunks for immediate feedback
- **Response Time**: < 10 seconds
- **Output**: Single actionable nudge for immediate implementation
- **Use Case**: Live meeting assistance and real-time coaching

### 2. **Post-meeting Analysis Crew** 📊

- **Purpose**: Comprehensive analysis of complete meeting recordings
- **Processing Time**: 3-5 minutes for 60-minute meeting
- **Output**: Detailed reports with insights and recommendations
- **Use Case**: Meeting effectiveness assessment and team development

### 3. **Meeting Insights Crew** 🎯

- **Purpose**: Long-term coaching and improvement recommendations
- **Analysis Scope**: Historical data across multiple meetings
- **Output**: Personalized coaching plans and progress tracking
- **Use Case**: Professional development and skill improvement

## 🚀 Features

### Real-time Analysis

- **Speech Analysis**: Pace, filler words, vocal energy, clarity
- **Visual Analysis**: Eye contact, facial expressions, posture
- **Body Language**: Gestures, engagement indicators
- **Smart Prioritization**: Single most critical feedback nudge

### Post-meeting Analysis

- **Transcription**: Complete with speaker identification and timestamps
- **Communication Patterns**: Speaking time, turn-taking, interruptions
- **Sentiment Analysis**: Emotional dynamics and engagement levels
- **Body Language Assessment**: Comprehensive non-verbal communication analysis
- **Executive Reports**: Actionable insights and recommendations

### Coaching Insights

- **Historical Analysis**: Long-term communication pattern recognition
- **Personalized Coaching**: Customized improvement plans
- **Progress Tracking**: Measurable goals and milestone tracking
- **Skill Development**: Evidence-based coaching recommendations

## 📦 Installation

### Prerequisites

- Python 3.10-3.13
- FFmpeg (for audio/video processing)
- OpenAI API key

### Dependencies Installation

The system requires multiple AI/ML libraries for audio, video, and text analysis:

```bash
# Install UV package manager (recommended)
pip install uv

# Navigate to project directory
cd meeting_agent_poc

# Install all dependencies
uv install
```

### Manual Installation

```bash
pip install crewai[tools]>=0.126.0
pip install opencv-python mediapipe numpy scipy
pip install librosa transformers torch pillow
pip install pandas matplotlib seaborn scikit-learn
pip install textblob nltk pydub speechrecognition
pip install openai-whisper
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL=gpt-4o-mini
```

## 🎮 Usage

### Command Line Interface

#### Real-time Analysis

Process 15-second chunks for immediate feedback:

```bash
python -m meeting_agent_poc.main real-time --chunk-file path/to/chunk.mp4
```

#### Post-meeting Analysis

Analyze complete meeting recordings:

```bash
python -m meeting_agent_poc.main post-meeting --meeting-video path/to/meeting.mp4 --meeting-id meeting_001
```

#### Coaching Insights

Generate personalized coaching recommendations:

```bash
python -m meeting_agent_poc.main insights --user-id john_doe
```

#### Legacy Mode

Run the original research crew:

```bash
python -m meeting_agent_poc.main run
```

### Programmatic Usage

```python
from meeting_agent_poc.crew import MeetingAgentPoc

# Initialize the meeting coach system
coach = MeetingAgentPoc()

# Real-time analysis for live feedback
result = coach.process_real_time_chunk('chunk_15s.mp4')

# Post-meeting comprehensive analysis
result = coach.process_full_meeting('full_meeting.mp4', 'meeting_001')

# Generate coaching insights
result = coach.generate_coaching_insights('user_001')
```

### Example Usage Script

Run the provided examples to see the system in action:

```bash
python examples.py
```

## 🏗️ System Architecture

### Agents Configuration

#### Real-time Analysis Crew

- **real_time_audio_analyst**: Speech patterns and vocal delivery
- **real_time_video_analyst**: Visual communication and body language
- **real_time_feedback_synthesizer**: Priority-based feedback coordination

#### Post-meeting Analysis Crew

- **meeting_transcription_specialist**: Accurate transcription with speaker ID
- **communication_pattern_analyst**: Speaking patterns and dynamics
- **sentiment_engagement_analyst**: Emotional intelligence and engagement
- **body_language_expert**: Non-verbal communication analysis
- **meeting_report_generator**: Comprehensive report synthesis

#### Meeting Insights Crew

- **historical_data_analyst**: Long-term pattern recognition
- **coaching_recommendation_specialist**: Personalized coaching plans
- **progress_tracking_specialist**: Goal setting and progress measurement

### Custom Tools

The system includes specialized tools for analysis:

- **AudioTranscriptionTool**: Speech-to-text with speaker identification
- **SpeechAnalyticsTool**: Pace, fillers, vocal characteristics
- **VideoFacialAnalysisTool**: Facial expressions and eye contact
- **BodyLanguageAnalysisTool**: Posture, gestures, engagement

## 📊 Output Examples

### Real-time Feedback

```json
{
  "primary_nudge": "pace_down",
  "nudge_message": "Slow down your speaking pace",
  "reasoning": "Speaking at 200 WPM, ideal range is 150-180",
  "confidence_score": 0.85,
  "next_check_priority": "audio"
}
```

### Post-meeting Report Structure

```markdown
# Meeting Analysis Report - Meeting_001

## Executive Summary

- Overall Effectiveness Score: 7.2/10
- Key Strengths: Active participation, clear decision-making
- Priority Areas: Speaking time distribution, interruption patterns

## Individual Performance

### John Doe (Meeting Leader)

- Speaking Time: 45% (recommended: 25-35%)
- Engagement Score: 8.5/10
- Key Recommendation: Increase space for team input

## Team Dynamics

- Participation Equity: Moderate concern
- Sentiment Progression: Positive trend
- Decision Quality: High

## Action Items

1. Implement round-robin speaking structure
2. Schedule follow-up on Project X decisions
3. Address technical audio issues for remote participants
```

### Coaching Insights

```json
{
  "coaching_overview": {
    "primary_focus_areas": ["Pacing", "Eye Contact", "Gesture Usage"],
    "coaching_philosophy": "Incremental improvement with measurable goals",
    "estimated_timeline": "8-12 weeks"
  },
  "improvement_modules": [
    {
      "skill_area": "Speaking Pace",
      "current_level": "Fast (180+ WPM)",
      "target_level": "Optimal (150-170 WPM)",
      "specific_exercises": ["Metronome practice", "Pause insertion"],
      "measurement_criteria": ["Weekly WPM tracking", "Audience feedback"]
    }
  ]
}
```

## 🎯 Use Cases

### Real-time Scenarios

- **Live Presentations**: Immediate feedback on delivery
- **Client Meetings**: Real-time coaching for sales effectiveness
- **Virtual Meetings**: Enhanced video presence coaching
- **Public Speaking**: Live presentation improvement

### Post-meeting Scenarios

- **Team Retrospectives**: Communication effectiveness assessment
- **Executive Reviews**: Leadership communication analysis
- **Training Sessions**: Facilitator effectiveness measurement
- **Customer Interactions**: Service quality evaluation

### Coaching Scenarios

- **Executive Coaching**: C-level communication development
- **Sales Training**: Presentation skills improvement
- **Team Development**: Communication culture building
- **Personal Growth**: Individual skill enhancement

## 🛠️ Configuration

### Crew Settings

Customize crew behavior in `src/meeting_agent_poc/config/crews.yaml`:

```yaml
real_time_analysis_crew:
  max_execution_time: 10 # seconds
  priority: "speed"
  retry_attempts: 1

post_meeting_analysis_crew:
  max_execution_time: 300 # seconds
  priority: "thoroughness"
  analysis_depth: "comprehensive"

meeting_insights_crew:
  historical_data_range: "6_months"
  goal_setting: true
```

### Agent Customization

Modify agents in `src/meeting_agent_poc/config/agents.yaml` to adjust:

- Role definitions and expertise areas
- Backstory for specialized knowledge
- Tool assignments and capabilities
- Interaction patterns and delegation

### Task Configuration

Update tasks in `src/meeting_agent_poc/config/tasks.yaml` to modify:

- Analysis requirements and scope
- Output formats and structures
- Context dependencies
- Retry and error handling

## 📈 Performance Optimization

### Real-time Processing

- Frame sampling for video analysis (every 10th frame)
- Optimized audio chunk processing
- Minimal context to reduce latency
- Priority-based feedback selection

### Comprehensive Analysis

- Parallel task execution where possible
- Efficient video processing with MediaPipe
- Batch audio analysis for better accuracy
- Memory optimization for large files

### Resource Management

- Configurable analysis depth
- GPU acceleration support (when available)
- Memory-efficient video streaming
- Intelligent caching for repeated analysis

## 🧪 Testing

### Unit Tests

```bash
# Run individual crew tests
python -m pytest tests/test_real_time_crew.py
python -m pytest tests/test_post_meeting_crew.py
python -m pytest tests/test_insights_crew.py
```

### Integration Tests

```bash
# Test full system workflow
python -m meeting_agent_poc.main test --iterations 5 --eval-llm gpt-4o-mini
```

### Sample Data

Use the provided sample scenarios in `examples.py` to test different meeting types and situations.

## 🚀 Deployment

### Local Development

```bash
# Install in development mode
uv install --dev

# Run with hot reload
python -m meeting_agent_poc.main run
```

### Production Deployment

```bash
# Build the package
uv build

# Install in production environment
pip install dist/meeting_agent_poc-*.whl
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv install --no-dev
CMD ["python", "-m", "meeting_agent_poc.main", "run"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for API changes
- Ensure all crews pass integration tests

## 📋 Roadmap

### Phase 1: Core Features ✅

- [x] Real-time analysis crew
- [x] Post-meeting analysis crew
- [x] Meeting insights crew
- [x] Custom tools implementation
- [x] CLI interface

### Phase 2: Enhanced Features 🚧

- [ ] Web dashboard interface
- [ ] Integration with video conferencing platforms
- [ ] Advanced emotion detection
- [ ] Multi-language support
- [ ] Team analytics dashboard

### Phase 3: Enterprise Features 🔮

- [ ] SSO integration
- [ ] Enterprise reporting
- [ ] API for third-party integrations
- [ ] Advanced privacy controls
- [ ] Custom model training

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CrewAI](https://crewai.com) for the multi-agent framework
- [OpenAI](https://openai.com) for language model capabilities
- [MediaPipe](https://mediapipe.dev) for computer vision tools
- [Librosa](https://librosa.org) for audio analysis capabilities

## 📞 Support

For support, questions, or feedback:

- 📧 Email: support@meeting-coach.ai
- 💬 [Join our Discord](https://discord.com/invite/meeting-coach)
- 📖 [Visit our documentation](https://docs.meeting-coach.ai)
- 🐛 [Report issues on GitHub](https://github.com/meeting-coach/issues)

---

**Transform your meetings with AI-powered communication coaching. Start improving your team's communication effectiveness today!** 🚀
