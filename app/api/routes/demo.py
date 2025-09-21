"""
Demo data API endpoints for showcasing QuickScore functionality.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Demo startup data
DEMO_STARTUPS = [
    {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "name": "TechFlow AI",
        "industry": "Artificial Intelligence",
        "stage": "Pre-Seed",
        "description": "AI-powered workflow automation platform for enterprise teams. Reduces manual work by 70% through intelligent process optimization.",
        "website": "https://techflow.ai",
        "linkedin_url": "https://linkedin.com/company/techflow-ai",
        "location": "San Francisco, CA",
        "founded_year": 2024,
        "funding_raised": 0,
        "team_size": 4,
        "overall_score": 85.2,
        "last_analysis": "2024-09-20T15:30:00Z",
        "status": "analyzed",
        "created_at": "2024-09-15T10:00:00Z"
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440002", 
        "name": "EcoCart Solutions",
        "industry": "E-commerce",
        "stage": "Seed",
        "description": "Sustainable e-commerce platform connecting eco-conscious consumers with verified green products. Already processing $2M ARR.",
        "website": "https://ecocart.solutions",
        "linkedin_url": "https://linkedin.com/company/ecocart-solutions",
        "location": "Austin, TX",
        "founded_year": 2023,
        "funding_raised": 1200000,
        "team_size": 12,
        "overall_score": 78.9,
        "last_analysis": "2024-09-19T14:20:00Z", 
        "status": "analyzed",
        "created_at": "2024-09-10T09:15:00Z"
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440003",
        "name": "MedAssist Pro",
        "industry": "Healthcare",
        "stage": "Pre-Seed", 
        "description": "Telemedicine platform with AI-powered diagnostic assistance for rural healthcare providers. Serving 50+ clinics.",
        "website": "https://medassist.pro",
        "linkedin_url": "https://linkedin.com/company/medassist-pro",
        "location": "Boston, MA",
        "founded_year": 2024,
        "funding_raised": 0,
        "team_size": 6,
        "overall_score": 82.1,
        "last_analysis": "2024-09-21T11:45:00Z",
        "status": "analyzed", 
        "created_at": "2024-09-12T16:30:00Z"
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440004",
        "name": "CryptoSecure Wallet",
        "industry": "Fintech",
        "stage": "Pre-Seed",
        "description": "Next-generation cryptocurrency wallet with quantum-resistant security and DeFi integration. 10K+ beta users.",
        "website": "https://cryptosecure.wallet",
        "linkedin_url": "https://linkedin.com/company/cryptosecure-wallet",
        "location": "Miami, FL",
        "founded_year": 2024,
        "funding_raised": 0,
        "team_size": 8,
        "overall_score": 73.4,
        "last_analysis": "2024-09-18T13:10:00Z",
        "status": "analyzed",
        "created_at": "2024-09-08T12:00:00Z"
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440005",
        "name": "EduPlatform Next", 
        "industry": "EdTech",
        "stage": "Pre-Seed",
        "description": "Personalized learning platform using adaptive AI to customize education paths for K-12 students. Piloting in 25 schools.",
        "website": "https://eduplatform.next",
        "linkedin_url": "https://linkedin.com/company/eduplatform-next",
        "location": "Denver, CO",
        "founded_year": 2024,
        "funding_raised": 0,
        "team_size": 7,
        "overall_score": 79.8,
        "last_analysis": "2024-09-17T10:25:00Z",
        "status": "analyzed",
        "created_at": "2024-09-05T14:45:00Z"
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440006",
        "name": "PineLabs",
        "industry": "Fintech",
        "stage": "Pre-Seed",
        "description": "Revolutionary payment processing platform for emerging markets with AI-powered fraud detection and seamless merchant onboarding.",
        "website": "https://pinelabs.com",
        "linkedin_url": "https://linkedin.com/company/pinelabs",
        "location": "Bangalore, India",
        "founded_year": 2024,
        "funding_raised": 0,
        "team_size": 15,
        "overall_score": 88.7,
        "last_analysis": "2024-09-21T16:00:00Z",
        "status": "analyzed",
        "created_at": "2024-09-21T15:30:00Z"
    }
]

# Demo analysis data
DEMO_ANALYSES = {
    "550e8400-e29b-41d4-a716-446655440001": {
        "overall_score": 85.2,
        "recommendation": "STRONG_INVEST",
        "confidence": 92,
        "detailed_scores": {
            "market_opportunity": 88,
            "team_strength": 90,
            "product_viability": 85,
            "business_model": 82,
            "competitive_advantage": 87,
            "execution_capability": 89,
            "financial_projections": 80,
            "risk_assessment": 75
        },
        "key_strengths": [
            "Exceptional founding team with previous AI/ML experience at Google and OpenAI",
            "Large addressable market ($50B+ workflow automation space)",
            "Early customer validation with 3 enterprise pilot programs",
            "Strong technical differentiation with proprietary NLP models",
            "Clear monetization strategy with proven enterprise sales approach"
        ],
        "risk_factors": [
            "Competitive market with established players like UiPath",
            "High customer acquisition costs in enterprise segment",
            "Dependency on third-party AI model providers",
            "Regulatory compliance challenges in certain industries"
        ],
        "market_analysis": {
            "market_size": "$50.8B",
            "growth_rate": "23.4% CAGR",
            "competition_level": "High",
            "barriers_to_entry": "Medium-High"
        },
        "financial_projections": {
            "year_1_revenue": 250000,
            "year_3_revenue": 15000000,
            "estimated_burn_rate": 75000,
            "runway_months": 18,
            "next_funding_needed": 2000000
        },
        "status": "completed",
        "created_at": "2024-09-20T15:30:00Z",
        "completed_at": "2024-09-20T15:45:00Z",
        "processing_time": 15
    },
    "550e8400-e29b-41d4-a716-446655440002": {
        "overall_score": 78.9,
        "recommendation": "INVEST",
        "confidence": 85,
        "detailed_scores": {
            "market_opportunity": 85,
            "team_strength": 78,
            "product_viability": 82,
            "business_model": 88,
            "competitive_advantage": 75,
            "execution_capability": 80,
            "financial_projections": 85,
            "risk_assessment": 70
        },
        "key_strengths": [
            "Proven traction with $2M ARR and 40% month-over-month growth",
            "Strong unit economics with 65% gross margins",
            "Growing market demand for sustainable products",
            "Established supplier network with 200+ verified eco-brands",
            "Experienced e-commerce team from Shopify and Amazon backgrounds"
        ],
        "risk_factors": [
            "Highly competitive e-commerce landscape",
            "Dependence on consumer trend towards sustainability",
            "Supply chain complexity with eco-verification requirements",
            "Customer acquisition costs increasing with market saturation"
        ],
        "market_analysis": {
            "market_size": "$15.7B",
            "growth_rate": "18.2% CAGR", 
            "competition_level": "Very High",
            "barriers_to_entry": "Low-Medium"
        },
        "financial_projections": {
            "year_1_revenue": 3200000,
            "year_3_revenue": 28000000,
            "estimated_burn_rate": 120000,
            "runway_months": 24,
            "next_funding_needed": 5000000
        },
        "status": "completed",
        "created_at": "2024-09-19T14:20:00Z",
        "completed_at": "2024-09-19T14:38:00Z",
        "processing_time": 18
    },
    "550e8400-e29b-41d4-a716-446655440006": {
        "overall_score": 88.7,
        "recommendation": "STRONG_INVEST",
        "confidence": 95,
        "detailed_scores": {
            "market_opportunity": 92,
            "team_strength": 89,
            "product_viability": 87,
            "business_model": 90,
            "competitive_advantage": 85,
            "execution_capability": 91,
            "financial_projections": 88,
            "risk_assessment": 78
        },
        "key_strengths": [
            "Massive untapped market in emerging economies ($2T+ payment processing)",
            "Exceptional founding team with deep fintech expertise from PayPal and Stripe",
            "Revolutionary AI fraud detection with 99.7% accuracy rate",
            "Already secured partnerships with 500+ merchants in pilot phase",
            "Strong regulatory compliance framework across 12 countries",
            "Proven traction with $500K monthly processing volume in beta"
        ],
        "risk_factors": [
            "Highly regulated fintech space with complex compliance requirements",
            "Intense competition from established players like Razorpay and Paytm",
            "Currency volatility in emerging markets affecting revenue predictability",
            "Dependency on banking partnerships and regulatory approvals",
            "High customer acquisition costs in price-sensitive markets"
        ],
        "market_analysis": {
            "market_size": "$2.1T",
            "growth_rate": "31.2% CAGR",
            "competition_level": "High",
            "barriers_to_entry": "Very High"
        },
        "financial_projections": {
            "year_1_revenue": 850000,
            "year_3_revenue": 45000000,
            "estimated_burn_rate": 95000,
            "runway_months": 21,
            "next_funding_needed": 3500000
        },
        "status": "completed",
        "created_at": "2024-09-21T16:00:00Z",
        "completed_at": "2024-09-21T16:12:00Z",
        "processing_time": 12
    }
}

# Demo founder data
DEMO_FOUNDERS = {
    "550e8400-e29b-41d4-a716-446655440001": [
        {
            "id": "founder-001",
            "name": "Sarah Chen",
            "title": "CEO & Co-Founder",
            "linkedin_url": "https://linkedin.com/in/sarah-chen-ai",
            "experience_years": 12,
            "previous_exits": 1,
            "domain_expert": True,
            "background": "Former Principal Engineer at Google AI, led team of 15 on AutoML platform",
            "score": 92,
            "key_achievements": [
                "Led $50M product at Google with 100M+ users",
                "Published 15 AI/ML papers with 2000+ citations",
                "Previous exit: AI startup acquired by Microsoft for $120M"
            ]
        },
        {
            "id": "founder-002", 
            "name": "Michael Rodriguez",
            "title": "CTO & Co-Founder",
            "linkedin_url": "https://linkedin.com/in/michael-rodriguez-ml",
            "experience_years": 10,
            "previous_exits": 0,
            "domain_expert": True,
            "background": "Former Senior Research Scientist at OpenAI, expert in NLP and automation",
            "score": 88,
            "key_achievements": [
                "Core contributor to GPT-3 training infrastructure",
                "Built scalable ML systems handling 1B+ requests/day",
                "Stanford PhD in Computer Science (AI specialization)"
            ]
        }
    ],
    "550e8400-e29b-41d4-a716-446655440002": [
        {
            "id": "founder-003",
            "name": "Jessica Park",
            "title": "CEO & Founder", 
            "linkedin_url": "https://linkedin.com/in/jessica-park-ecommerce",
            "experience_years": 8,
            "previous_exits": 0,
            "domain_expert": True,
            "background": "Former Director of Marketplace at Shopify, sustainability advocate",
            "score": 82,
            "key_achievements": [
                "Scaled Shopify's marketplace from $500M to $2B GMV",
                "Built eco-certification program used by 10K+ brands",
                "Featured in Forbes 30 Under 30 for Social Impact"
            ]
        }
    ],
    "550e8400-e29b-41d4-a716-446655440006": [
        {
            "id": "founder-007",
            "name": "Rajesh Sharma",
            "title": "CEO & Co-Founder",
            "linkedin_url": "https://linkedin.com/in/rajesh-sharma-fintech",
            "experience_years": 14,
            "previous_exits": 2,
            "domain_expert": True,
            "background": "Former VP of Engineering at PayPal India, led payment infrastructure for 200M+ users",
            "score": 94,
            "key_achievements": [
                "Built PayPal's fraud detection system processing $50B+ annually",
                "Founded and sold payment startup to Mastercard for $85M",
                "IIT Delhi graduate with 20+ patents in payment technology",
                "Led teams of 100+ engineers across 5 countries"
            ]
        },
        {
            "id": "founder-008",
            "name": "Priya Patel",
            "title": "CTO & Co-Founder",
            "linkedin_url": "https://linkedin.com/in/priya-patel-ai",
            "experience_years": 11,
            "previous_exits": 1,
            "domain_expert": True,
            "background": "Former Principal Engineer at Stripe, AI/ML expert specializing in fraud detection",
            "score": 90,
            "key_achievements": [
                "Designed Stripe's ML-powered risk assessment engine",
                "Published 12 papers on AI fraud detection with 1500+ citations",
                "Previous startup acquired by Google for $120M",
                "MIT PhD in Machine Learning and Financial Technology"
            ]
        },
        {
            "id": "founder-009",
            "name": "Amit Gupta",
            "title": "COO & Co-Founder",
            "linkedin_url": "https://linkedin.com/in/amit-gupta-operations",
            "experience_years": 12,
            "previous_exits": 1,
            "domain_expert": True,
            "background": "Former Director of Operations at Razorpay, expert in fintech scaling and compliance",
            "score": 87,
            "key_achievements": [
                "Scaled Razorpay from 1K to 8M merchants in 4 years",
                "Built regulatory compliance framework for 15+ countries",
                "Led $200M Series E funding round at previous startup",
                "Harvard MBA with focus on emerging market fintech"
            ]
        }
    ]
}


@router.get("/startups", response_model=List[Dict[str, Any]])
async def get_demo_startups():
    """Get demo startup data for showcasing."""
    return DEMO_STARTUPS


@router.get("/startups/{startup_id}", response_model=Dict[str, Any])
async def get_demo_startup(startup_id: str):
    """Get specific demo startup data."""
    startup = next((s for s in DEMO_STARTUPS if s["id"] == startup_id), None)
    if not startup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo startup not found"
        )
    return startup


@router.get("/startups/{startup_id}/analysis", response_model=Dict[str, Any])
async def get_demo_analysis(startup_id: str):
    """Get demo analysis data for a startup."""
    if startup_id not in DEMO_ANALYSES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo analysis not found"
        )
    return DEMO_ANALYSES[startup_id]


@router.get("/startups/{startup_id}/founders", response_model=List[Dict[str, Any]])
async def get_demo_founders(startup_id: str):
    """Get demo founder data for a startup."""
    if startup_id not in DEMO_FOUNDERS:
        return []
    return DEMO_FOUNDERS[startup_id]


@router.get("/dashboard/stats", response_model=Dict[str, Any])
async def get_demo_dashboard_stats():
    """Get demo dashboard statistics."""
    return {
        "total_startups": len(DEMO_STARTUPS),
        "analyzed_startups": len([s for s in DEMO_STARTUPS if s["status"] == "analyzed"]),
        "average_score": sum(s["overall_score"] for s in DEMO_STARTUPS) / len(DEMO_STARTUPS),
        "top_industries": [
            {"name": "Fintech", "count": 2, "avg_score": 81.05},
            {"name": "Artificial Intelligence", "count": 1, "avg_score": 85.2},
            {"name": "E-commerce", "count": 1, "avg_score": 78.9},
            {"name": "Healthcare", "count": 1, "avg_score": 82.1},
            {"name": "EdTech", "count": 1, "avg_score": 79.8}
        ],
        "recent_analyses": [
            {"startup_name": "PineLabs", "score": 88.7, "date": "2024-09-21"},
            {"startup_name": "TechFlow AI", "score": 85.2, "date": "2024-09-20"},
            {"startup_name": "MedAssist Pro", "score": 82.1, "date": "2024-09-21"},
            {"startup_name": "EcoCart Solutions", "score": 78.9, "date": "2024-09-19"}
        ],
        "investment_recommendations": {
            "strong_invest": 2,
            "invest": 3, 
            "consider": 1,
            "pass": 0
        }
    }


@router.post("/startups/demo/create", response_model=Dict[str, Any])
async def create_demo_startup():
    """Create a new demo startup for demonstration."""
    demo_startup = {
        "id": str(uuid.uuid4()),
        "name": "NewTech Innovations",
        "industry": "Clean Technology", 
        "stage": "Pre-Seed",
        "description": "Revolutionary solar panel technology with 40% higher efficiency using quantum dot enhancement.",
        "website": "https://newtech.innovations",
        "location": "Palo Alto, CA",
        "founded_year": 2024,
        "team_size": 5,
        "status": "pending_analysis",
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "success": True,
        "message": "Demo startup created successfully",
        "data": demo_startup
    }


@router.post("/startups/{startup_id}/analyze", response_model=Dict[str, Any])
async def trigger_demo_analysis(startup_id: str):
    """Trigger a demo analysis for a startup."""
    startup = next((s for s in DEMO_STARTUPS if s["id"] == startup_id), None)
    if not startup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo startup not found"
        )
    
    return {
        "success": True,
        "message": f"Analysis started for {startup['name']}",
        "analysis_id": str(uuid.uuid4()),
        "estimated_completion": (datetime.now() + timedelta(minutes=15)).isoformat(),
        "status": "processing"
    }


@router.get("/comparable/{startup_id}", response_model=List[Dict[str, Any]])
async def get_demo_comparable_startups(startup_id: str):
    """Get demo comparable startups."""
    startup = next((s for s in DEMO_STARTUPS if s["id"] == startup_id), None)
    if not startup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo startup not found"
        )
    
    # Return other startups in the same industry as comparables
    comparables = [s for s in DEMO_STARTUPS if s["id"] != startup_id and s["industry"] == startup["industry"]]
    
    # If no same industry, return top scoring startups
    if not comparables:
        comparables = sorted(DEMO_STARTUPS, key=lambda x: x["overall_score"], reverse=True)[:3]
        comparables = [s for s in comparables if s["id"] != startup_id]
    
    return comparables[:3]