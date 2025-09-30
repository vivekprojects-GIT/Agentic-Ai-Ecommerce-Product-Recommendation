"""Simple script to run API server."""

import subprocess
import sys
import signal

def main():
    """Run the API server."""
    print("Commerce Agent - Starting API Server")
    print("=" * 50)
    
    try:
        print("Starting FastAPI server...")
        print("\nServices:")
        print("   UI & API: http://localhost:8080")
        print("   API Docs: http://localhost:8080/docs")
        print("\nPress Ctrl+C to stop")
        
        # Start API server
        process = subprocess.Popen([sys.executable, "api.py"])
        
        # Wait for process
        process.wait()
    
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        print("Server stopped")

if __name__ == "__main__":
    main()