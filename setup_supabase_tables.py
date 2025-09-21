"""
Script to create required tables in Supabase database.
Run this script to set up the database schema.
"""
import asyncio
import os
from supabase import create_client, Client

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kkjqxrckbmuprtlfiqvd.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtranF4cmNrYm11cHJ0bGZpcXZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgyMTg3OTEsImV4cCI6MjA3Mzc5NDc5MX0.Elx9YNYPVhc2pj-BhrgIMdhcyo_jgHdPupCVrQc9J0o")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

def get_table_creation_sql():
    """Return SQL for creating required tables."""
    return """
-- Create startups table
CREATE TABLE IF NOT EXISTS public.startups (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    stage VARCHAR(50),
    website VARCHAR(255),
    github_repo VARCHAR(255),
    funding_amount DECIMAL(15, 2),
    team_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create startup_analyses table (renamed from analyses to be more specific)
CREATE TABLE IF NOT EXISTS public.startup_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    startup_id UUID REFERENCES public.startups(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL DEFAULT 'full_analysis',
    overall_score DECIMAL(4,2) CHECK (overall_score >= 0 AND overall_score <= 100),
    market_score DECIMAL(4,2) CHECK (market_score >= 0 AND market_score <= 100),
    team_score DECIMAL(4,2) CHECK (team_score >= 0 AND team_score <= 100),
    product_score DECIMAL(4,2) CHECK (product_score >= 0 AND product_score <= 100),
    financial_score DECIMAL(4,2) CHECK (financial_score >= 0 AND financial_score <= 100),
    reasoning TEXT,
    market_reasoning TEXT,
    team_reasoning TEXT,
    product_reasoning TEXT,
    financial_reasoning TEXT,
    raw_response TEXT,
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create users table
CREATE TABLE IF NOT EXISTS public.users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255),
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_startups_name ON public.startups(name);
CREATE INDEX IF NOT EXISTS idx_startups_industry ON public.startups(industry);
CREATE INDEX IF NOT EXISTS idx_startups_stage ON public.startups(stage);
CREATE INDEX IF NOT EXISTS idx_startup_analyses_startup_id ON public.startup_analyses(startup_id);
CREATE INDEX IF NOT EXISTS idx_startup_analyses_created_at ON public.startup_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- Enable Row Level Security (RLS)
ALTER TABLE public.startups ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.startup_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (adjust as needed for production)
-- For now, allow all operations for development
CREATE POLICY IF NOT EXISTS "Enable all operations for startups" ON public.startups
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY IF NOT EXISTS "Enable all operations for startup_analyses" ON public.startup_analyses
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY IF NOT EXISTS "Enable read access for users" ON public.users
    FOR SELECT USING (true);

CREATE POLICY IF NOT EXISTS "Enable insert access for users" ON public.users
    FOR INSERT WITH CHECK (true);

-- Add some sample data
INSERT INTO public.startups (name, description, industry, stage, website, github_repo, funding_amount, team_size) 
VALUES 
    ('TechStart Inc', 'AI-powered analytics platform for businesses', 'Technology', 'Seed', 'https://techstart.example', 'https://github.com/techstart/analytics', 500000.00, 8),
    ('GreenEco Solutions', 'Sustainable packaging solutions for e-commerce', 'Sustainability', 'Series A', 'https://greeneco.example', 'https://github.com/greeneco/packaging', 2000000.00, 15),
    ('HealthTech Pro', 'Telemedicine platform for rural areas', 'Healthcare', 'Pre-seed', 'https://healthtech.example', 'https://github.com/healthtech/platform', 100000.00, 3)
ON CONFLICT (id) DO NOTHING;

-- Print success message
SELECT 'Database tables created successfully!' as message;
"""

async def create_tables_via_api():
    """Create tables using Supabase client (limited functionality)."""
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    print("🔧 Setting up Supabase database tables...")
    print("\n📋 SQL to execute in Supabase SQL Editor:")
    print("=" * 60)
    print(get_table_creation_sql())
    print("=" * 60)
    
    print("\n📝 Instructions:")
    print("1. Go to your Supabase project dashboard: https://supabase.com/dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Copy and paste the SQL above")
    print("4. Click 'Run' to execute the SQL")
    print("5. The tables will be created with sample data")
    
    # Test connection
    try:
        # Try to access a non-existent table to test connection
        result = client.table('startups').select('count', count='exact').limit(1).execute()
        print("\n✅ Supabase connection is working!")
        print("ℹ️ If you see table errors, that's expected - run the SQL above to create them.")
    except Exception as e:
        print(f"\n⚠️ Connection test result: {e}")
        if "404" in str(e) or "table" in str(e).lower():
            print("✅ This is expected - tables don't exist yet. Run the SQL above to create them.")
        else:
            print("❌ There might be a connection issue.")

if __name__ == "__main__":
    asyncio.run(create_tables_via_api())