"""
QuickScore Streamlit Application
AI-powered startup analysis and investment decision platform
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import uuid
import random
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="QuickScore - AI Startup Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .score-good { color: #10b981; font-weight: bold; }
    .score-medium { color: #f59e0b; font-weight: bold; }
    .score-poor { color: #ef4444; font-weight: bold; }
    
    .recommendation-strong { background-color: #10b981; color: white; padding: 0.5rem; border-radius: 5px; }
    .recommendation-invest { background-color: #3b82f6; color: white; padding: 0.5rem; border-radius: 5px; }
    .recommendation-consider { background-color: #f59e0b; color: white; padding: 0.5rem; border-radius: 5px; }
    .recommendation-pass { background-color: #ef4444; color: white; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Demo data
@st.cache_data
def get_demo_startups():
    return [
        {
            "id": "startup-001",
            "name": "TechFlow AI",
            "industry": "Artificial Intelligence",
            "stage": "Pre-Seed",
            "description": "AI-powered workflow automation platform for enterprise teams. Reduces manual work by 70% through intelligent process optimization.",
            "location": "San Francisco, CA",
            "founded_year": 2024,
            "team_size": 4,
            "funding_raised": 0,
            "overall_score": 85.2,
            "recommendation": "STRONG_INVEST",
            "created_at": "2024-09-15"
        },
        {
            "id": "startup-002",
            "name": "EcoCart Solutions",
            "industry": "E-commerce",
            "stage": "Seed",
            "description": "Sustainable e-commerce platform connecting eco-conscious consumers with verified green products. Already processing $2M ARR.",
            "location": "Austin, TX",
            "founded_year": 2023,
            "team_size": 12,
            "funding_raised": 1200000,
            "overall_score": 78.9,
            "recommendation": "INVEST",
            "created_at": "2024-09-10"
        },
        {
            "id": "startup-003",
            "name": "MedAssist Pro",
            "industry": "Healthcare",
            "stage": "Pre-Seed",
            "description": "Telemedicine platform with AI-powered diagnostic assistance for rural healthcare providers. Serving 50+ clinics.",
            "location": "Boston, MA",
            "founded_year": 2024,
            "team_size": 6,
            "funding_raised": 0,
            "overall_score": 82.1,
            "recommendation": "INVEST",
            "created_at": "2024-09-12"
        },
        {
            "id": "startup-004",
            "name": "PineLabs",
            "industry": "Fintech",
            "stage": "Pre-Seed",
            "description": "Revolutionary payment processing platform for emerging markets with AI-powered fraud detection and seamless merchant onboarding.",
            "location": "Bangalore, India",
            "founded_year": 2024,
            "team_size": 15,
            "funding_raised": 0,
            "overall_score": 88.7,
            "recommendation": "STRONG_INVEST",
            "created_at": "2024-09-21"
        }
    ]

@st.cache_data
def get_detailed_analysis(startup_id: str):
    analyses = {
        "startup-001": {
            "detailed_scores": {
                "Market Opportunity": 88,
                "Team Strength": 90,
                "Product Viability": 85,
                "Business Model": 82,
                "Competitive Advantage": 87,
                "Execution Capability": 89,
                "Financial Projections": 80,
                "Risk Assessment": 75
            },
            "key_strengths": [
                "Exceptional founding team with previous AI/ML experience at Google and OpenAI",
                "Large addressable market ($50B+ workflow automation space)",
                "Early customer validation with 3 enterprise pilot programs",
                "Strong technical differentiation with proprietary NLP models"
            ],
            "risk_factors": [
                "Competitive market with established players like UiPath",
                "High customer acquisition costs in enterprise segment",
                "Dependency on third-party AI model providers"
            ],
            "market_size": "$50.8B",
            "growth_rate": "23.4% CAGR",
            "confidence": 92
        },
        "startup-004": {
            "detailed_scores": {
                "Market Opportunity": 92,
                "Team Strength": 89,
                "Product Viability": 87,
                "Business Model": 90,
                "Competitive Advantage": 85,
                "Execution Capability": 91,
                "Financial Projections": 88,
                "Risk Assessment": 78
            },
            "key_strengths": [
                "Massive untapped market in emerging economies ($2T+ payment processing)",
                "Exceptional founding team with deep fintech expertise from PayPal and Stripe",
                "Revolutionary AI fraud detection with 99.7% accuracy rate",
                "Already secured partnerships with 500+ merchants in pilot phase"
            ],
            "risk_factors": [
                "Highly regulated fintech space with complex compliance requirements",
                "Intense competition from established players like Razorpay and Paytm",
                "Currency volatility in emerging markets affecting revenue predictability"
            ],
            "market_size": "$2.1T",
            "growth_rate": "31.2% CAGR",
            "confidence": 95
        }
    }
    
    # Default analysis for other startups
    return analyses.get(startup_id, {
        "detailed_scores": {
            "Market Opportunity": random.randint(70, 90),
            "Team Strength": random.randint(75, 85),
            "Product Viability": random.randint(70, 85),
            "Business Model": random.randint(75, 90),
            "Competitive Advantage": random.randint(70, 85),
            "Execution Capability": random.randint(75, 90),
            "Financial Projections": random.randint(70, 85),
            "Risk Assessment": random.randint(65, 80)
        },
        "key_strengths": [
            "Strong market positioning in target industry",
            "Experienced founding team",
            "Clear value proposition",
            "Scalable business model"
        ],
        "risk_factors": [
            "Market competition",
            "Regulatory challenges",
            "Customer acquisition costs",
            "Technology risks"
        ],
        "market_size": f"${random.randint(10, 100)}B",
        "growth_rate": f"{random.randint(15, 35)}% CAGR",
        "confidence": random.randint(80, 95)
    })

def get_score_color(score):
    if score >= 80:
        return "score-good"
    elif score >= 70:
        return "score-medium"
    else:
        return "score-poor"

def get_recommendation_class(recommendation):
    return f"recommendation-{recommendation.lower().replace('_', '-')}"

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚀 QuickScore</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">AI-Powered Pre-Seed Startup Analyzer</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Dashboard", "🏢 Startup Portfolio", "📈 Detailed Analysis", "✨ Create Demo Startup"]
    )
    
    # Get data
    startups = get_demo_startups()
    
    # Initialize session state
    if 'user_startups' not in st.session_state:
        st.session_state.user_startups = []
    
    # Combine demo and user startups
    all_startups = st.session_state.user_startups + startups
    
    if page == "🏠 Dashboard":
        show_dashboard(all_startups)
    elif page == "🏢 Startup Portfolio":
        show_startup_portfolio(all_startups)
    elif page == "📈 Detailed Analysis":
        show_detailed_analysis(all_startups)
    elif page == "✨ Startup":
        show_create_startup()

def show_dashboard(startups):
    st.header("📊 Investment Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Startups</h3>
            <h2>{}</h2>
        </div>
        """.format(len(startups)), unsafe_allow_html=True)
    
    with col2:
        analyzed = len([s for s in startups if s.get('overall_score')])
        st.markdown("""
        <div class="metric-card">
            <h3>Analyzed</h3>
            <h2>{}</h2>
        </div>
        """.format(analyzed), unsafe_allow_html=True)
    
    with col3:
        avg_score = sum(s.get('overall_score', 0) for s in startups) / max(len(startups), 1)
        st.markdown("""
        <div class="metric-card">
            <h3>Average Score</h3>
            <h2>{:.1f}</h2>
        </div>
        """.format(avg_score), unsafe_allow_html=True)
    
    with col4:
        strong_invest = len([s for s in startups if s.get('recommendation') == 'STRONG_INVEST'])
        st.markdown("""
        <div class="metric-card">
            <h3>Strong Investments</h3>
            <h2>{}</h2>
        </div>
        """.format(strong_invest), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Score Distribution")
        if startups:
            scores = [s.get('overall_score', 0) for s in startups if s.get('overall_score')]
            df_scores = pd.DataFrame({
                'Startup': [s['name'] for s in startups if s.get('overall_score')],
                'Score': scores
            })
            
            fig = px.bar(
                df_scores, 
                x='Startup', 
                y='Score',
                color='Score',
                color_continuous_scale='RdYlGn',
                title="Startup Scores"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏭 Industry Distribution")
        if startups:
            industry_counts = {}
            for startup in startups:
                industry = startup.get('industry', 'Unknown')
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
            
            fig = px.pie(
                values=list(industry_counts.values()),
                names=list(industry_counts.keys()),
                title="Startups by Industry"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🕒 Recent Analyses")
    recent_startups = sorted(startups, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
    
    for startup in recent_startups:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{startup['name']}** - {startup.get('industry', 'Unknown')}")
        with col2:
            score = startup.get('overall_score', 0)
            st.markdown(f'<span class="{get_score_color(score)}">{score}/100</span>', unsafe_allow_html=True)
        with col3:
            st.write(startup.get('created_at', 'N/A'))

def show_startup_portfolio(startups):
    st.header("🏢 Startup Portfolio")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industries = list(set([s.get('industry', 'Unknown') for s in startups]))
        selected_industry = st.selectbox("Filter by Industry", ["All"] + industries)
    
    with col2:
        stages = list(set([s.get('stage', 'Unknown') for s in startups]))
        selected_stage = st.selectbox("Filter by Stage", ["All"] + stages)
    
    with col3:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    
    # Filter startups
    filtered_startups = startups
    if selected_industry != "All":
        filtered_startups = [s for s in filtered_startups if s.get('industry') == selected_industry]
    if selected_stage != "All":
        filtered_startups = [s for s in filtered_startups if s.get('stage') == selected_stage]
    filtered_startups = [s for s in filtered_startups if s.get('overall_score', 0) >= min_score]
    
    st.markdown("---")
    
    # Display startups
    for i in range(0, len(filtered_startups), 2):
        col1, col2 = st.columns(2)
        
        for j, col in enumerate([col1, col2]):
            if i + j < len(filtered_startups):
                startup = filtered_startups[i + j]
                
                with col:
                    with st.container():
                        st.markdown(f"### {startup['name']}")
                        st.write(f"**Industry:** {startup.get('industry', 'Unknown')}")
                        st.write(f"**Stage:** {startup.get('stage', 'Unknown')}")
                        st.write(f"**Location:** {startup.get('location', 'Unknown')}")
                        st.write(f"**Team Size:** {startup.get('team_size', 'Unknown')} members")
                        
                        score = startup.get('overall_score', 0)
                        recommendation = startup.get('recommendation', 'CONSIDER')
                        
                        col_score, col_rec = st.columns(2)
                        with col_score:
                            st.markdown(f'**Score:** <span class="{get_score_color(score)}">{score}/100</span>', unsafe_allow_html=True)
                        with col_rec:
                            st.markdown(f'<div class="{get_recommendation_class(recommendation)}">{recommendation.replace("_", " ")}</div>', unsafe_allow_html=True)
                        
                        st.write(f"**Description:** {startup.get('description', 'No description available.')}")
                        
                        # Progress bar
                        st.progress(score / 100)
                        
                        # User created badge
                        if startup.get('id', '').startswith('user-'):
                            st.success("✨ Just Created")
                        
                        st.markdown("---")

def show_detailed_analysis(startups):
    st.header("📈 Detailed Analysis")
    
    if not startups:
        st.warning("No startups available for analysis. Create one in the Demo section!")
        return
    
    # Select startup
    startup_names = [f"{s['name']} (Score: {s.get('overall_score', 0)})" for s in startups]
    selected_startup = st.selectbox("Select a startup for detailed analysis:", startup_names)
    
    if selected_startup:
        # Get selected startup
        startup_index = startup_names.index(selected_startup)
        startup = startups[startup_index]
        
        # Get detailed analysis
        analysis = get_detailed_analysis(startup['id'])
        
        st.markdown("---")
        
        # Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{startup.get('overall_score', 0)}/100")
        
        with col2:
            recommendation = startup.get('recommendation', 'CONSIDER')
            st.markdown(f'**Recommendation:**<br><div class="{get_recommendation_class(recommendation)}">{recommendation.replace("_", " ")}</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric("Confidence", f"{analysis.get('confidence', 85)}%")
        
        st.markdown("---")
        
        # Score breakdown
        st.subheader("📊 Score Breakdown")
        
        scores = analysis.get('detailed_scores', {})
        if scores:
            # Radar chart
            categories = list(scores.keys())
            values = list(scores.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=startup['name'],
                fillcolor='rgba(102, 126, 234, 0.6)',
                line_color='rgba(102, 126, 234, 1)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Performance Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Score details
            col1, col2 = st.columns(2)
            
            with col1:
                for i, (category, score) in enumerate(scores.items()):
                    if i % 2 == 0:
                        st.metric(category, f"{score}/100")
            
            with col2:
                for i, (category, score) in enumerate(scores.items()):
                    if i % 2 == 1:
                        st.metric(category, f"{score}/100")
        
        st.markdown("---")
        
        # Strengths and risks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Key Strengths")
            strengths = analysis.get('key_strengths', [])
            for strength in strengths:
                st.write(f"• {strength}")
        
        with col2:
            st.subheader("⚠️ Risk Factors")
            risks = analysis.get('risk_factors', [])
            for risk in risks:
                st.write(f"• {risk}")
        
        st.markdown("---")
        
        # Market analysis
        st.subheader("🌍 Market Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Size", analysis.get('market_size', 'N/A'))
        
        with col2:
            st.metric("Growth Rate", analysis.get('growth_rate', 'N/A'))
        
        with col3:
            competition_level = "High" if startup.get('overall_score', 0) > 80 else "Medium"
            st.metric("Competition Level", competition_level)

def show_create_startup():
    st.header("✨ Startup")
    
    st.write("Create a new startup and see AI-powered analysis in action!")
    
    with st.form("create_startup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Startup Name", placeholder="Enter startup name")
            industry = st.selectbox("Industry", [
                "Artificial Intelligence",
                "Fintech",
                "Healthcare",
                "E-commerce",
                "EdTech",
                "Clean Technology",
                "SaaS",
                "Biotech"
            ])
            stage = st.selectbox("Stage", ["Pre-Seed", "Seed", "Series A"])
            location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
        
        with col2:
            team_size = st.slider("Team Size", 1, 50, 5)
            founded_year = st.selectbox("Founded Year", range(2020, 2025), index=4)
            funding_raised = st.number_input("Funding Raised ($)", min_value=0, value=0, step=50000)
            
        description = st.text_area("Description", placeholder="Brief description of the startup and its value proposition")
        
        submitted = st.form_submit_button("🚀 Create & Analyze Startup")
        
        if submitted:
            if name and description:
                # Simulate analysis
                with st.spinner("🔄 Creating startup and running AI analysis..."):
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    # Generate realistic score
                    base_score = random.randint(70, 85)
                    if industry in ["Artificial Intelligence", "Fintech", "Healthcare"]:
                        base_score += random.randint(5, 15)
                    
                    # Generate recommendation
                    if base_score >= 85:
                        recommendation = "STRONG_INVEST"
                    elif base_score >= 75:
                        recommendation = "INVEST"
                    else:
                        recommendation = "CONSIDER"
                    
                    # Create new startup
                    new_startup = {
                        "id": f"user-{uuid.uuid4()}",
                        "name": name,
                        "industry": industry,
                        "stage": stage,
                        "description": description,
                        "location": location,
                        "founded_year": founded_year,
                        "team_size": team_size,
                        "funding_raised": funding_raised,
                        "overall_score": base_score,
                        "recommendation": recommendation,
                        "created_at": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Add to session state
                    st.session_state.user_startups.append(new_startup)
                    
                    st.success(f"✅ Analysis complete! {name} scored {base_score}/100")
                    st.success(f"📊 Recommendation: {recommendation.replace('_', ' ')}")
                    st.info("💡 Check the 'Startup Portfolio' section to see your new startup!")
                    
                    # Show quick preview
                    st.markdown("### 📋 Quick Preview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Score", f"{base_score}/100")
                    with col2:
                        st.markdown(f'<div class="{get_recommendation_class(recommendation)}">{recommendation.replace("_", " ")}</div>', unsafe_allow_html=True)
                    with col3:
                        confidence = random.randint(85, 95)
                        st.metric("Confidence", f"{confidence}%")
                    
            else:
                st.error("Please fill in at least the startup name and description.")

if __name__ == "__main__":
    main()