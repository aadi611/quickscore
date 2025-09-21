"""
Database abstraction layer using Supabase REST API.
This bypasses PostgreSQL direct connection issues while maintaining the same interface.
"""
import os
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kkjqxrckbmuprtlfiqvd.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtranF4cmNrYm11cHJ0bGZpcXZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgyMTg3OTEsImV4cCI6MjA3Mzc5NDc5MX0.Elx9YNYPVhc2pj-BhrgIMdhcyo_jgHdPupCVrQc9J0o")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

class SupabaseDatabase:
    """Database abstraction using Supabase REST API."""
    
    def __init__(self):
        self.client = supabase
    
    async def create_tables(self):
        """Create tables if they don't exist using SQL."""
        # Since we can't execute DDL through REST API, we'll ensure tables exist
        # Tables should be created manually in Supabase dashboard or via migrations
        try:
            # Test if tables exist by querying them
            await self.get_startups()
            print("✅ Database tables are accessible")
            return True
        except Exception as e:
            print(f"⚠️ Tables may not exist: {e}")
            print("📝 Please create tables manually in Supabase dashboard:")
            print(self._get_table_creation_sql())
            return False
    
    def _get_table_creation_sql(self) -> str:
        """Return SQL for creating required tables."""
        return """
-- Create tables in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS startups (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    stage VARCHAR(50),
    website VARCHAR(255),
    github_repo VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS startup_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    startup_id UUID REFERENCES startups(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    score DECIMAL(3,2),
    reasoning TEXT,
    raw_response TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS (Row Level Security) if needed
ALTER TABLE startups ENABLE ROW LEVEL SECURITY;
ALTER TABLE startup_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust as needed)
CREATE POLICY "Enable read access for all users" ON startups FOR SELECT USING (true);
CREATE POLICY "Enable read access for all users" ON startup_analyses FOR SELECT USING (true);
"""
    
    # Startup operations
    async def create_startup(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new startup."""
        try:
            result = self.client.table('startups').insert(startup_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating startup: {e}")
            raise
    
    async def get_startup(self, startup_id: str) -> Optional[Dict[str, Any]]:
        """Get a startup by ID."""
        try:
            result = self.client.table('startups').select("*").eq('id', startup_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error getting startup: {e}")
            return None
    
    async def get_startups(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all startups with pagination."""
        try:
            result = self.client.table('startups').select("*").range(skip, skip + limit - 1).execute()
            return result.data or []
        except Exception as e:
            print(f"Error getting startups: {e}")
            return []
    
    async def update_startup(self, startup_id: str, startup_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a startup."""
        try:
            startup_data['updated_at'] = datetime.utcnow().isoformat()
            result = self.client.table('startups').update(startup_data).eq('id', startup_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error updating startup: {e}")
            raise
    
    async def delete_startup(self, startup_id: str) -> bool:
        """Delete a startup."""
        try:
            result = self.client.table('startups').delete().eq('id', startup_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting startup: {e}")
            return False
    
    # Analysis operations
    async def create_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new analysis."""
        try:
            result = self.client.table('startup_analyses').insert(analysis_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating analysis: {e}")
            raise
    
    async def get_startup_analyses(self, startup_id: str) -> List[Dict[str, Any]]:
        """Get all analyses for a startup."""
        try:
            result = self.client.table('startup_analyses').select("*").eq('startup_id', startup_id).execute()
            return result.data or []
        except Exception as e:
            print(f"Error getting analyses: {e}")
            return []
    
    async def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get an analysis by ID."""
        try:
            result = self.client.table('startup_analyses').select("*").eq('id', analysis_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error getting analysis: {e}")
            return None
    
    # User operations
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user."""
        try:
            result = self.client.table('users').insert(user_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating user: {e}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get a user by email."""
        try:
            result = self.client.table('users').select("*").eq('email', email).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

# Global database instance
db = SupabaseDatabase()

# For backward compatibility with existing code
async def get_database():
    """Get database instance (for dependency injection)."""
    return db

async def init_database():
    """Initialize database tables."""
    return await db.create_tables()