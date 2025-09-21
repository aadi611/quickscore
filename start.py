#!/usr/bin/env python3
"""
QuickScore Application Launcher
Simple script to start both backend and frontend servers together.
"""
import os
import sys
import subprocess
import platform
import time
import signal
from pathlib import Path


def check_prerequisites():
    """Check if all prerequisites are met before starting."""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not Path("app").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the QuickScore root directory")
        return False
    
    # Check virtual environment
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python.exe")
        pip_path = Path("venv/Scripts/pip.exe")
    else:
        python_path = Path("venv/bin/python")
        pip_path = Path("venv/bin/pip")
    
    if not python_path.exists():
        print("❌ Virtual environment not found!")
        print("   Please run: python setup.py")
        return False
    print("✅ Virtual environment found")
    
    # Check critical Python dependencies
    critical_packages = [
        "fastapi", "uvicorn", "pydantic", "pydantic_settings"
    ]
    
    missing_packages = []
    for package in critical_packages:
        try:
            result = subprocess.run([str(python_path), "-c", f"import {package}"], 
                                  check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            missing_packages.append(package.replace("_", "-"))
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        print(f"   Please run: {pip_path} install {' '.join(missing_packages)}")
        print("   Or run the full setup: python setup.py")
        return False
    print("✅ Python dependencies available")
    
    # Check Node.js and npm
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ Node.js and npm found")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Node.js/npm not found!")
        print("   Please install Node.js from: https://nodejs.org/")
        print("   Make sure to add it to your system PATH")
        return False
    
    # Check frontend dependencies
    frontend_node_modules = Path("frontend/node_modules")
    if not frontend_node_modules.exists():
        print("❌ Frontend dependencies not installed!")
        print("   Please run: cd frontend && npm install")
        return False
    print("✅ Frontend dependencies installed")
    
    # Check if frontend is built
    frontend_dist = Path("frontend/dist")
    if not frontend_dist.exists():
        print("⚠️  Frontend not built, attempting to build it now...")
        try:
            result = subprocess.run(["npm", "run", "build"], 
                                  cwd=Path("frontend"), check=True, capture_output=True, text=True)
            print("✅ Frontend built successfully")
        except subprocess.CalledProcessError as e:
            print("❌ Failed to build frontend:")
            print(f"   {e.stderr}")
            print("   Please run: cd frontend && npm run build")
            return False
    else:
        print("✅ Frontend built and ready")
    
    return True


def start_backend():
    """Start the FastAPI backend server."""
    print("🔧 Starting FastAPI backend...")
    
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python.exe")
    else:
        python_path = Path("venv/bin/python")
    
    try:
        backend_process = subprocess.Popen([
            str(python_path), "-m", "uvicorn", "app.main:app", 
            "--reload", "--port", "8000", "--host", "127.0.0.1"
        ], cwd=Path.cwd())
        
        print("✅ Backend starting on http://127.0.0.1:8000")
        return backend_process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None


def start_frontend():
    """Start the frontend preview server."""
    print("🎨 Starting frontend preview server...")
    
    try:
        frontend_process = subprocess.Popen([
            "npm", "run", "preview", "--", "--host", "127.0.0.1", "--port", "4173"
        ], cwd=Path("frontend"))
        
        print("✅ Frontend starting on http://127.0.0.1:4173")
        return frontend_process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        print("   Make sure Node.js and npm are installed and in your PATH")
        return None


def wait_for_servers():
    """Wait for servers to start up."""
    print("⏳ Waiting for servers to start...")
    time.sleep(3)
    
    # Test backend
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is responding")
        else:
            print("⚠️  Backend started but health check failed")
    except:
        print("⚠️  Backend may still be starting...")
    
    # Test frontend (just check if port is responding)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 4173))
        sock.close()
        if result == 0:
            print("✅ Frontend is responding")
        else:
            print("⚠️  Frontend may still be starting...")
    except:
        print("⚠️  Frontend status unknown")


def main():
    """Main application launcher."""
    print("=" * 60)
    print("🚀 QuickScore Application Launcher")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("\n🔄 Starting servers...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Wait a moment
    time.sleep(2)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Wait for servers to be ready
    wait_for_servers()
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 QuickScore Application Started Successfully!")
    print("=" * 60)
    print("🌐 URLs:")
    print("   📱 Frontend UI:   http://127.0.0.1:4173")
    print("   🔧 Backend API:   http://127.0.0.1:8000")
    print("   📚 API Docs:      http://127.0.0.1:8000/docs")
    print("   ❤️  Health Check: http://127.0.0.1:8000/health")
    print("   🎭 Demo Data:     http://127.0.0.1:8000/api/v1/demo/startups")
    print("   🚀 Demo Page:     file:///C:/Hackathon/QuickScore/demo.html")
    print("\n💡 Tips:")
    print("   • Open http://127.0.0.1:4173 in your browser for the main UI")
    print("   • Open demo.html for a standalone demo with sample data")
    print("   • Both servers will reload automatically on code changes")
    print("   • Press Ctrl+C to stop both servers")
    print("=" * 60)
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down servers...")
        try:
            frontend_process.terminate()
            backend_process.terminate()
            
            # Wait for graceful shutdown
            time.sleep(2)
            
            # Force kill if needed
            try:
                frontend_process.kill()
                backend_process.kill()
            except:
                pass
                
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        print("✅ Servers stopped. Goodbye!")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the script running and monitor processes
        while True:
            # Check if processes are still running
            backend_running = backend_process.poll() is None
            frontend_running = frontend_process.poll() is None
            
            if not backend_running and not frontend_running:
                print("❌ Both servers have stopped")
                break
            elif not backend_running:
                print("❌ Backend server has stopped")
                frontend_process.terminate()
                break
            elif not frontend_running:
                print("❌ Frontend server has stopped")
                backend_process.terminate()
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    print("\n👋 Application stopped")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()