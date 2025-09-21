#!/usr/bin/env python3
"""
QuickScore setup and quick start script.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_requirements():
    """Check if required software is installed."""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 9:
        print(f"❌ Python 3.9+ required, found {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False
    
    return True


def setup_environment():
    """Set up Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment"):
        return False
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    # Determine the correct pip path based on platform
    if platform.system() == "Windows":
        pip_path = Path("venv/Scripts/pip.exe")
        python_path = Path("venv/Scripts/python.exe")
    else:
        pip_path = Path("venv/bin/pip")
        python_path = Path("venv/bin/python")
    
    if not pip_path.exists():
        print("❌ Virtual environment pip not found")
        return False
    
    # Upgrade pip
    if not run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command([str(pip_path), "install", "-r", "requirements.txt"], "Installing dependencies"):
        return False
    
    # Install Playwright browsers
    if not run_command([str(python_path), "-m", "playwright", "install", "chromium"], "Installing Playwright browsers"):
        print("⚠️  Playwright browser installation failed - web scraping may not work")
    
    return True


def setup_environment_file():
    """Create .env file from example if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("📄 .env file already exists")
        return True
    
    if not env_example_path.exists():
        print("❌ .env.example file not found")
        return False
    
    # Copy example to .env
    try:
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from example")
        print("⚠️  Please edit .env file with your actual API keys and database URLs")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def check_docker():
    """Check if Docker is available for database setup."""
    try:
        result = subprocess.run(["docker", "--version"], check=True, capture_output=True, text=True)
        print(f"✅ Docker is available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available")
        print("   Please install Docker to run PostgreSQL and Redis locally")
        return False


def start_databases():
    """Start PostgreSQL and Redis using Docker Compose."""
    compose_file = Path("docker-compose.yml")
    
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    # Start only databases
    if not run_command(["docker-compose", "up", "-d", "postgres", "redis"], "Starting databases"):
        return False
    
    # Wait a moment for databases to start
    print("⏳ Waiting for databases to start...")
    import time
    time.sleep(5)
    
    return True


def run_migrations():
    """Run database migrations."""
    # Determine the correct python path
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python.exe")
    else:
        python_path = Path("venv/bin/python")
    
    if not python_path.exists():
        print("❌ Virtual environment python not found")
        return False
    
    # Run migrations
    if not run_command([str(python_path), "-m", "alembic", "upgrade", "head"], "Running database migrations"):
        print("⚠️  Database migrations failed - you may need to set up the database manually")
        return False
    
    return True


def check_frontend():
    """Check if frontend is set up and ready."""
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    package_json = frontend_path / "package.json"
    if not package_json.exists():
        print("❌ Frontend package.json not found")
        return False
    
    node_modules = frontend_path / "node_modules"
    if not node_modules.exists():
        print("⚠️  Frontend dependencies not installed")
        return False
    
    dist_folder = frontend_path / "dist"
    if not dist_folder.exists():
        print("⚠️  Frontend not built (dist folder missing)")
        return False
    
    print("✅ Frontend is ready")
    return True


def run_application():
    """Run both backend and frontend servers together."""
    print("\n" + "="*60)
    print("🚀 Starting QuickScore Application")
    print("="*60)
    
    # Check if frontend is ready
    if not check_frontend():
        print("\n❌ Frontend not ready. Please ensure:")
        print("   1. Frontend dependencies are installed: cd frontend && npm install")
        print("   2. Frontend is built: cd frontend && npm run build")
        return False
    
    # Determine python path
    if platform.system() == "Windows":
        python_path = Path("venv/Scripts/python.exe")
    else:
        python_path = Path("venv/bin/python")
    
    if not python_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    print("\n🔄 Starting servers...")
    
    try:
        # Start backend server in a subprocess
        print("🔧 Starting FastAPI backend on http://localhost:8000...")
        backend_process = subprocess.Popen([
            str(python_path), "-m", "uvicorn", "app.main:app", 
            "--reload", "--port", "8000"
        ], cwd=Path.cwd())
        
        # Wait a moment for backend to start
        import time
        time.sleep(3)
        
        # Start frontend server in a subprocess
        print("🎨 Starting frontend on http://localhost:4173...")
        frontend_process = subprocess.Popen([
            "npm", "run", "preview"
        ], cwd=Path("frontend"))
        
        # Wait a moment for frontend to start
        time.sleep(2)
        
        print("\n" + "="*60)
        print("🎉 QuickScore Application Started Successfully!")
        print("="*60)
        print("🌐 Backend API: http://localhost:8000")
        print("🎨 Frontend UI: http://localhost:4173")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop both servers")
        print("="*60)
        
        # Wait for user interrupt
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Servers stopped")
            
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    return True


def print_quick_start_info():
    """Print quick start information."""
    print("\n" + "="*60)
    print("🎉 QuickScore MVP Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your API keys:")
    print("   - OPENAI_API_KEY=your_openai_api_key")
    print("   - DATABASE_URL=postgresql://postgres:password@localhost:5432/quickscore")
    print("   - REDIS_URL=redis://localhost:6379/0")
    
    print("\n🚀 To start the full application (backend + frontend):")
    print("   python setup.py --run")
    
    print("\n🔧 Or start services individually:")
    
    if platform.system() == "Windows":
        print("   # Backend only")
        print("   venv\\Scripts\\activate")
        print("   python -m uvicorn app.main:app --reload --port 8000")
        print("   ")
        print("   # Frontend only (in new terminal)")
        print("   cd frontend")
        print("   npm run preview")
    else:
        print("   # Backend only")
        print("   source venv/bin/activate")
        print("   python -m uvicorn app.main:app --reload --port 8000")
        print("   ")
        print("   # Frontend only (in new terminal)")
        print("   cd frontend")
        print("   npm run preview")
    
    print("\n🌐 Application URLs:")
    print("   - Frontend UI: http://localhost:4173")
    print("   - Backend API: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    
    print("\n📖 Example API Usage:")
    print('   curl -X POST "http://localhost:8000/api/v1/startups" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"name": "TechCorp", "industry": "SaaS", "description": "AI platform"}\'')


def main():
    """Main setup function."""
    # Check for run argument
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("🚀 QuickScore Application Launcher")
        print("="*40)
        
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        
        # Run the application
        if run_application():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Normal setup mode
    print("🚀 QuickScore MVP Setup")
    print("="*30)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        sys.exit(1)
    
    # Setup steps
    setup_steps = [
        (setup_environment, "Set up virtual environment"),
        (install_dependencies, "Install dependencies"),
        (setup_environment_file, "Create .env file"),
    ]
    
    for step_func, step_name in setup_steps:
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    # Optional Docker setup
    if check_docker():
        if input("\n🐳 Start databases with Docker? (y/n): ").lower().startswith('y'):
            start_databases()
            
            if input("🗄️  Run database migrations? (y/n): ").lower().startswith('y'):
                run_migrations()
    
    # Print completion info
    print_quick_start_info()


if __name__ == "__main__":
    main()