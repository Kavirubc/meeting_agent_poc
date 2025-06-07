from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from dotenv import load_dotenv
from .tools import AudioTranscriptionTool, SpeechAnalyticsTool, VideoFacialAnalysisTool, BodyLanguageAnalysisTool

# Load environment variables from .env file
load_dotenv()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MeetingAgentPoc():
    """MeetingAgentPoc crew with multiple specialized crews for AI Meeting Coach"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # ============ REAL-TIME ANALYSIS CREW AGENTS ============
    @agent
    def real_time_audio_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['real_time_audio_analyst'],
            tools=[AudioTranscriptionTool(), SpeechAnalyticsTool()],
            verbose=True
        )

    @agent
    def real_time_video_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['real_time_video_analyst'],
            tools=[VideoFacialAnalysisTool(), BodyLanguageAnalysisTool()],
            verbose=True
        )

    @agent
    def real_time_feedback_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['real_time_feedback_synthesizer'],
            verbose=True
        )

    # ============ POST-MEETING ANALYSIS CREW AGENTS ============
    @agent
    def meeting_transcription_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['meeting_transcription_specialist'],
            tools=[AudioTranscriptionTool()],
            verbose=True
        )

    @agent
    def communication_pattern_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['communication_pattern_analyst'],
            verbose=True
        )

    @agent
    def sentiment_engagement_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_engagement_analyst'],
            verbose=True
        )

    @agent
    def body_language_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['body_language_expert'],
            tools=[VideoFacialAnalysisTool(), BodyLanguageAnalysisTool()],
            verbose=True
        )

    @agent
    def meeting_report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['meeting_report_generator'],
            verbose=True
        )

    # ============ MEETING INSIGHTS CREW AGENTS ============
    @agent
    def historical_data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['historical_data_analyst'],
            verbose=True
        )

    @agent
    def coaching_recommendation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['coaching_recommendation_specialist'],
            verbose=True
        )

    @agent
    def progress_tracking_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['progress_tracking_specialist'],
            verbose=True
        )

    # ============ LEGACY AGENTS ============
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    # ============ REAL-TIME ANALYSIS TASKS ============
    @task
    def real_time_audio_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['real_time_audio_analysis_task'],
        )

    @task
    def real_time_video_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['real_time_video_analysis_task'],
        )

    @task
    def synthesize_real_time_feedback_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_real_time_feedback_task'],
        )

    # ============ POST-MEETING ANALYSIS TASKS ============
    @task
    def comprehensive_transcription_task(self) -> Task:
        return Task(
            config=self.tasks_config['comprehensive_transcription_task'],
        )

    @task
    def communication_patterns_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['communication_patterns_analysis_task'],
        )

    @task
    def sentiment_engagement_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_engagement_analysis_task'],
        )

    @task
    def body_language_comprehensive_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['body_language_comprehensive_analysis_task'],
        )

    @task
    def generate_comprehensive_meeting_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_comprehensive_meeting_report_task'],
        )

    # ============ MEETING INSIGHTS TASKS ============
    @task
    def historical_pattern_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['historical_pattern_analysis_task'],
        )

    @task
    def personalized_coaching_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['personalized_coaching_plan_task'],
        )

    @task
    def progress_tracking_setup_task(self) -> Task:
        return Task(
            config=self.tasks_config['progress_tracking_setup_task'],
        )

    # ============ LEGACY TASKS ============
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    # ============ CREW DEFINITIONS ============
    @crew
    def crew(self) -> Crew:
        """Creates the default MeetingAgentPoc crew (legacy)"""
        return Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[self.research_task(), self.reporting_task()],
            process=Process.sequential,
            verbose=True,
        )

    def real_time_analysis_crew(self) -> Crew:
        """Creates the Real-time Analysis Crew"""
        return Crew(
            agents=[
                self.real_time_audio_analyst(),
                self.real_time_video_analyst(),
                self.real_time_feedback_synthesizer()
            ],
            tasks=[
                self.real_time_audio_analysis_task(),
                self.real_time_video_analysis_task(),
                self.synthesize_real_time_feedback_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=False  # Disabled for speed in real-time processing
        )

    def post_meeting_analysis_crew(self) -> Crew:
        """Creates the Post-meeting Analysis Crew"""
        return Crew(
            agents=[
                self.meeting_transcription_specialist(),
                self.communication_pattern_analyst(),
                self.sentiment_engagement_analyst(),
                self.body_language_expert(),
                self.meeting_report_generator()
            ],
            tasks=[
                self.comprehensive_transcription_task(),
                self.communication_patterns_analysis_task(),
                self.sentiment_engagement_analysis_task(),
                self.body_language_comprehensive_analysis_task(),
                self.generate_comprehensive_meeting_report_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True
        )

    def meeting_insights_crew(self) -> Crew:
        """Creates the Meeting Insights Crew"""
        return Crew(
            agents=[
                self.historical_data_analyst(),
                self.coaching_recommendation_specialist(),
                self.progress_tracking_specialist()
            ],
            tasks=[
                self.historical_pattern_analysis_task(),
                self.personalized_coaching_plan_task(),
                self.progress_tracking_setup_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True
        )

    # ============ EXECUTION METHODS ============
    def process_real_time_chunk(self, chunk_file_path: str):
        """Process 15-second chunk for real-time feedback"""
        inputs = {"chunk_file_path": chunk_file_path}
        return self.real_time_analysis_crew().kickoff(inputs=inputs)

    def process_full_meeting(self, meeting_video_path: str, meeting_id: str):
        """Process complete meeting for comprehensive analysis"""
        inputs = {
            "meeting_video_path": meeting_video_path,
            "meeting_id": meeting_id
        }
        return self.post_meeting_analysis_crew().kickoff(inputs=inputs)

    def generate_coaching_insights(self, user_id: str):
        """Generate long-term coaching insights"""
        inputs = {"user_id": user_id}
        return self.meeting_insights_crew().kickoff(inputs=inputs)
