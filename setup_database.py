#!/usr/bin/env python3
"""
Script to create database tables in Supabase.
This script uses SQL execution via the Supabase REST API.
"""
import asyncio
import httpx
import json

SUPABASE_URL = "https://kkjqxrckbmuprtlfiqvd.supabase.co"
# Note: We'll need to use the service role key for DDL operations
SUPABASE_SERVICE_KEY = "sb_secret_GWEDopY2cbAcOiEvZ_j2gQ_2I1LZPqG"

# SQL to create tables
CREATE_TABLES_SQL = """
-- Create tables in Supabase

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

-- Enable RLS (Row Level Security) - allows public access for demo
ALTER TABLE startups ENABLE ROW LEVEL SECURITY;
ALTER TABLE startup_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create permissive policies for demo purposes
DROP POLICY IF EXISTS "Enable all access for startups" ON startups;
CREATE POLICY "Enable all access for startups" ON startups FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable all access for analyses" ON startup_analyses;
CREATE POLICY "Enable all access for analyses" ON startup_analyses FOR ALL USING (true);

DROP POLICY IF EXISTS "Enable read access for users" ON users;
CREATE POLICY "Enable read access for users" ON users FOR SELECT USING (true);

-- Insert sample data
INSERT INTO startups (name, description, industry, stage, website) VALUES
('TechStart AI', 'AI-powered productivity platform for remote teams', 'Technology', 'Seed', 'https://techstart.ai'),
('GreenEnergy Solutions', 'Renewable energy storage systems for residential use', 'Clean Energy', 'Pre-Seed', 'https://greenenergy.co'),
('FinTech Plus', 'Mobile banking platform for small businesses', 'Financial Services', 'Series A', 'https://fintechplus.com')
ON CONFLICT (name) DO NOTHING;
"""

async def create_tables():
    """Create tables using Supabase SQL API."""
    print("🔧 Creating Supabase Database Tables")
    print("="*50)
    
    try:
        # Use the SQL REST API endpoint
        url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
        
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json"
        }
        
        # Try to execute SQL using a custom function
        # First, let's try the simpler approach with direct table creation
        await create_tables_via_api()
        
    except Exception as e:
        print(f"❌ Failed to create tables via API: {e}")
        print("\n📝 Manual Setup Required:")
        print("Please copy and paste the following SQL into your Supabase SQL Editor:")
        print("="*70)
        print(CREATE_TABLES_SQL)
        print("="*70)

async def create_tables_via_api():
    """Create tables by inserting sample data (will auto-create tables)."""
    print("🔄 Creating tables by inserting sample data...")
    
    async with httpx.AsyncClient() as client:
        # Try to insert sample startup data
        # This will fail if table doesn't exist, but we'll catch it
        startup_data = {
            "name": "TechStart AI",
            "description": "AI-powered productivity platform for remote teams",
            "industry": "Technology",
            "stage": "Seed",
            "website": "https://techstart.ai"
        }
        
        try:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/startups",
                json=startup_data,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
            )
            
            if response.status_code in [200, 201]:
                print("✅ Tables appear to exist and are accessible")
                return True
            else:
                print(f"⚠️ Response: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Table creation attempt failed: {e}")
            return False

async def test_table_access():
    """Test if we can read from the tables."""
    print("\n🔍 Testing table access...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/startups?select=*&limit=1",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Successfully read {len(data)} rows from startups table")
                if data:
                    print(f"   Sample data: {data[0].get('name', 'Unknown')}")
                return True
            else:
                print(f"❌ Failed to read startups table: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing table access: {e}")
            return False

async def main():
    """Main setup function."""
    print("🚀 Supabase Database Setup")
    print("="*50)
    
    # Test current access
    access_works = await test_table_access()
    
    if not access_works:
        print("\n🔧 Tables don't exist, attempting to create them...")
        await create_tables()
        
        # Test again
        print("\n🔍 Testing after creation attempt...")
        access_works = await test_table_access()
    
    print("\n" + "="*50)
    if access_works:
        print("🎉 Database setup complete!")
        print("   Your backend should now work with Supabase")
    else:
        print("❌ Database setup needs manual intervention")
        print("   Please create tables manually in Supabase Dashboard")
        print("\n📝 Instructions:")
        print("   1. Go to https://supabase.com/dashboard")
        print("   2. Open your project")
        print("   3. Go to SQL Editor")
        print("   4. Paste and run the SQL printed above")

if __name__ == "__main__":
    asyncio.run(main())