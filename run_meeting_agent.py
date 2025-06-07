#!/usr/bin/env python3
"""
Convenience script to run the Meeting Agent POC system.
This script sets up the proper Python path and runs the main module.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    src_dir = script_dir / "src"
    
    # Set up the Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)
    
    # Run the main module with all passed arguments
    cmd = [sys.executable, "-m", "meeting_agent_poc.main"] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, env=env, cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running meeting agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
