# Basic crew execution (what we just tested)
python run_meeting_agent.py run

# Real-time analysis (15-second chunks)
python run_meeting_agent.py real-time --chunk-file /path/to/audio_chunk.wav

# Post-meeting analysis (full recordings)
python run_meeting_agent.py post-meeting --meeting-video "/Users/kaviruhapuarachchi/Downloads/meeting_agent_poc/knowledge/meeting/Screen Recording 2025-06-07 at 00.29.18.mov" --meeting-id meeting123

# Generate coaching insights
python run_meeting_agent.py insights --user-id user123

# Training and testing modes
python run_meeting_agent.py train --iterations 5 --filename training_output.json
python run_meeting_agent.py test --iterations 3 --eval-llm gpt-4
python run_meeting_agent.py replay --task-id task_id_here