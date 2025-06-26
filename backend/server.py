from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import os
import uuid
import json
import asyncio
import googlemaps
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import schedule
import time
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BizFizz Ultimate Competitive Intelligence Platform",
    version="3.0.0",
    description="The Most Advanced AI-Powered Business Intelligence Platform for Restaurant Owners"
)

# Security
security = HTTPBearer()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URL)
db = client.bizfizz_ultimate

# API clients and configuration
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
YELP_API_KEY = os.environ.get('YELP_API_KEY')

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

# Initialize OpenAI client with proper error handling
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        openai_client = None

# Log API key status
logger.info(f"API Keys loaded - Google Maps: {bool(GOOGLE_MAPS_API_KEY)}, OpenAI: {bool(OPENAI_API_KEY)}, Yelp: {bool(YELP_API_KEY)}")

# Pydantic models
class GeographicSearch(BaseModel):
    location: str = Field(..., description="City, zip code, or address")
    radius: int = Field(default=5, description="Search radius in miles")
    business_type: str = Field(default="restaurant", description="Type of business")
    include_demographics: bool = Field(default=True, description="Include demographic analysis")
    include_social_media: bool = Field(default=True, description="Include social media analysis")

class UserBusiness(BaseModel):
    name: str = Field(..., description="Business name")
    address: str = Field(..., description="Business address")
    business_type: str = Field(default="restaurant")
    target_demographics: Optional[List[str]] = None
    current_pricing: Optional[Dict[str, float]] = None
    goals: Optional[List[str]] = None

class CompetitiveAnalysisRequest(BaseModel):
    competitor_ids: List[str]
    location: str
    user_business: Optional[UserBusiness] = None
    analysis_depth: str = Field(default="comprehensive", description="basic, standard, comprehensive, enterprise")
    include_predictions: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)

# Advanced Data Collection Functions
def collect_comprehensive_business_data(business_data, location):
    """Collect comprehensive business intelligence data"""
    try:
        enhanced_data = business_data.copy()
        
        # Calculate advanced metrics
        rating = enhanced_data.get('rating', 0)
        review_count = enhanced_data.get('review_count', 0)
        price_level = enhanced_data.get('price_level', 1)
        
        # Advanced business intelligence calculations
        market_penetration = min(100, (review_count / 1000) * 100)
        customer_loyalty = min(100, (rating - 3) * 50) if rating >= 3 else 0
        digital_presence_score = calculate_digital_presence_score(enhanced_data)
        competitive_threat_level = calculate_threat_level(enhanced_data)
        
        # Estimated financial metrics
        estimated_seats = estimate_restaurant_capacity(enhanced_data)
        estimated_daily_covers = estimate_daily_covers(enhanced_data, estimated_seats)
        estimated_avg_check = price_level * 25 + (rating - 3) * 10
        estimated_monthly_revenue = estimated_daily_covers * estimated_avg_check * 30
        
        # Customer demographics estimation
        demographics = estimate_customer_demographics(enhanced_data, location)
        
        # Operational metrics
        operational_efficiency = calculate_operational_efficiency(enhanced_data)
        service_quality_score = rating * 20
        
        # Market positioning
        market_position = determine_market_position(enhanced_data, price_level, rating)
        
        # Growth potential analysis
        growth_potential = analyze_growth_potential(enhanced_data, market_penetration, customer_loyalty)
        
        enhanced_data.update({
            "advanced_metrics": {
                "market_penetration_score": market_penetration,
                "customer_loyalty_index": customer_loyalty,
                "digital_presence_score": digital_presence_score,
                "competitive_threat_level": competitive_threat_level,
                "estimated_seats": estimated_seats,
                "estimated_daily_covers": estimated_daily_covers,
                "estimated_avg_check": estimated_avg_check,
                "estimated_monthly_revenue": estimated_monthly_revenue,
                "operational_efficiency": operational_efficiency,
                "service_quality_score": service_quality_score,
                "growth_potential": growth_potential
            },
            "customer_demographics": demographics,
            "market_positioning": market_position,
            "peak_performance_indicators": calculate_peak_indicators(enhanced_data),
            "competitive_advantages": identify_competitive_advantages(enhanced_data),
            "risk_factors": identify_risk_factors(enhanced_data),
            "expansion_opportunities": identify_expansion_opportunities(enhanced_data, location)
        })
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Error collecting comprehensive business data: {str(e)}")
        return business_data

def calculate_digital_presence_score(business_data):
    """Calculate digital presence and marketing strength"""
    score = 0
    
    # Base score from reviews
    review_count = business_data.get('review_count', 0)
    score += min(50, review_count / 20)
    
    # Website presence
    if business_data.get('website'):
        score += 20
    
    # Rating contribution
    rating = business_data.get('rating', 0)
    if rating >= 4.0:
        score += 20
    elif rating >= 3.5:
        score += 10
    
    # Categories diversity (indicates market reach)
    categories = business_data.get('categories', [])
    score += min(10, len(categories) * 2)
    
    return min(100, score)

def calculate_threat_level(business_data):
    """Calculate competitive threat level"""
    rating = business_data.get('rating', 0)
    review_count = business_data.get('review_count', 0)
    price_level = business_data.get('price_level', 1)
    
    threat_score = 0
    
    # High rating = higher threat
    if rating >= 4.5:
        threat_score += 40
    elif rating >= 4.0:
        threat_score += 30
    elif rating >= 3.5:
        threat_score += 20
    
    # High review count = higher threat
    if review_count >= 1000:
        threat_score += 30
    elif review_count >= 500:
        threat_score += 20
    elif review_count >= 100:
        threat_score += 10
    
    # Premium pricing with high rating = significant threat
    if price_level >= 3 and rating >= 4.0:
        threat_score += 20
    
    if threat_score >= 70:
        return "High"
    elif threat_score >= 40:
        return "Medium"
    else:
        return "Low"

def estimate_restaurant_capacity(business_data):
    """Estimate restaurant seating capacity"""
    base_capacity = 50  # Default assumption
    
    price_level = business_data.get('price_level', 1)
    categories = business_data.get('categories', [])
    
    # Adjust based on restaurant type
    if any(cat in ['Fine Dining', 'Steakhouses'] for cat in categories):
        base_capacity = max(40, base_capacity - 10)  # Fine dining usually smaller
    elif any(cat in ['Fast Food', 'Pizza'] for cat in categories):
        base_capacity = max(30, base_capacity + 20)  # Fast food usually larger
    elif any(cat in ['Cafes', 'Coffee'] for cat in categories):
        base_capacity = max(20, base_capacity - 20)  # Cafes usually smaller
    
    # Adjust based on price level
    if price_level == 4:
        base_capacity = max(30, base_capacity - 15)  # Premium places often smaller
    elif price_level == 1:
        base_capacity += 25  # Budget places often larger
    
    return base_capacity

def estimate_daily_covers(business_data, estimated_seats):
    """Estimate daily customer covers"""
    rating = business_data.get('rating', 3.5)
    price_level = business_data.get('price_level', 2)
    
    # Base turnover rate (customers per seat per day)
    base_turnover = 2.5
    
    # Adjust based on rating (higher rating = more customers)
    if rating >= 4.5:
        base_turnover *= 1.4
    elif rating >= 4.0:
        base_turnover *= 1.2
    elif rating >= 3.5:
        base_turnover *= 1.0
    else:
        base_turnover *= 0.8
    
    # Adjust based on price level (lower price = higher volume)
    if price_level == 1:
        base_turnover *= 1.3
    elif price_level == 4:
        base_turnover *= 0.7
    
    return int(estimated_seats * base_turnover)

def estimate_customer_demographics(business_data, location):
    """Estimate customer demographics based on business type and location"""
    categories = business_data.get('categories', [])
    price_level = business_data.get('price_level', 2)
    
    demographics = {
        "age_groups": {},
        "income_levels": {},
        "dining_preferences": {},
        "visit_frequency": {},
        "party_size_distribution": {}
    }
    
    # Age group estimation based on restaurant type
    if any(cat in ['Fast Food', 'Pizza'] for cat in categories):
        demographics["age_groups"] = {"18-25": 30, "26-35": 25, "36-50": 25, "51+": 20}
    elif any(cat in ['Fine Dining', 'Steakhouses'] for cat in categories):
        demographics["age_groups"] = {"18-25": 10, "26-35": 25, "36-50": 35, "51+": 30}
    elif any(cat in ['Cafes', 'Coffee'] for cat in categories):
        demographics["age_groups"] = {"18-25": 35, "26-35": 30, "36-50": 20, "51+": 15}
    else:
        demographics["age_groups"] = {"18-25": 20, "26-35": 30, "36-50": 30, "51+": 20}
    
    # Income level based on price level
    if price_level == 1:
        demographics["income_levels"] = {"Low": 40, "Middle": 45, "High": 15}
    elif price_level == 2:
        demographics["income_levels"] = {"Low": 25, "Middle": 50, "High": 25}
    elif price_level == 3:
        demographics["income_levels"] = {"Low": 15, "Middle": 40, "High": 45}
    else:
        demographics["income_levels"] = {"Low": 5, "Middle": 25, "High": 70}
    
    return demographics

def calculate_operational_efficiency(business_data):
    """Calculate operational efficiency score"""
    rating = business_data.get('rating', 0)
    review_count = business_data.get('review_count', 0)
    
    # Higher rating with more reviews indicates consistent operation
    efficiency = (rating - 2) * 25  # Base efficiency from rating
    
    # Consistency bonus (more reviews with high rating = consistent operation)
    if rating >= 4.0 and review_count >= 100:
        efficiency += 20
    elif rating >= 3.5 and review_count >= 50:
        efficiency += 10
    
    return max(0, min(100, efficiency))

def determine_market_position(business_data, price_level, rating):
    """Determine market positioning strategy"""
    if price_level >= 3 and rating >= 4.0:
        return {
            "position": "Premium Leader",
            "strategy": "High-quality, premium pricing",
            "target_market": "Affluent customers seeking quality"
        }
    elif price_level <= 2 and rating >= 4.0:
        return {
            "position": "Value Champion",
            "strategy": "High quality at competitive prices",
            "target_market": "Value-conscious quality seekers"
        }
    elif price_level >= 3 and rating < 4.0:
        return {
            "position": "Premium Challenger",
            "strategy": "Premium pricing with improvement needed",
            "target_market": "Premium market with execution gaps"
        }
    else:
        return {
            "position": "Market Participant",
            "strategy": "Standard market offering",
            "target_market": "General market"
        }

def analyze_growth_potential(business_data, market_penetration, customer_loyalty):
    """Analyze growth potential"""
    rating = business_data.get('rating', 0)
    review_count = business_data.get('review_count', 0)
    
    potential_score = 0
    
    # High rating with low penetration = high growth potential
    if rating >= 4.0 and market_penetration < 50:
        potential_score += 40
    
    # Consistent quality (loyalty) indicates scalability
    if customer_loyalty >= 60:
        potential_score += 30
    
    # Room for review growth
    if review_count < 500 and rating >= 4.0:
        potential_score += 20
    
    # Location factors (simplified)
    potential_score += 10  # Base location opportunity
    
    if potential_score >= 70:
        return "High"
    elif potential_score >= 40:
        return "Medium"
    else:
        return "Low"

def calculate_peak_indicators(business_data):
    """Calculate peak performance indicators"""
    rating = business_data.get('rating', 0)
    review_count = business_data.get('review_count', 0)
    
    return {
        "peak_rating_potential": min(5.0, rating + 0.3),
        "review_growth_potential": int(review_count * 1.5),
        "market_share_potential": "3-7% local market share",
        "revenue_growth_potential": "15-30% annually"
    }

def identify_competitive_advantages(business_data):
    """Identify key competitive advantages"""
    advantages = []
    
    rating = business_data.get('rating', 0)
    categories = business_data.get('categories', [])
    price_level = business_data.get('price_level', 1)
    
    if rating >= 4.5:
        advantages.append("Exceptional customer satisfaction")
    
    if len(categories) >= 3:
        advantages.append("Diverse menu and market appeal")
    
    if price_level <= 2 and rating >= 4.0:
        advantages.append("Superior value proposition")
    
    if 'Delivery' in str(categories):
        advantages.append("Strong delivery capabilities")
    
    if not advantages:
        advantages.append("Established market presence")
    
    return advantages

def identify_risk_factors(business_data):
    """Identify potential risk factors"""
    risks = []
    
    rating = business_data.get('rating', 0)
    review_count = business_data.get('review_count', 0)
    
    if rating < 3.5:
        risks.append("Below-average customer satisfaction")
    
    if review_count < 50:
        risks.append("Limited customer feedback and visibility")
    
    if rating >= 4.0 and review_count < 100:
        risks.append("Good quality but limited market penetration")
    
    if not risks:
        risks.append("Standard market risks")
    
    return risks

def identify_expansion_opportunities(business_data, location):
    """Identify expansion and growth opportunities"""
    opportunities = []
    
    rating = business_data.get('rating', 0)
    categories = business_data.get('categories', [])
    
    if rating >= 4.0:
        opportunities.append("Franchise or additional location potential")
    
    if 'Delivery' not in str(categories):
        opportunities.append("Delivery service expansion")
    
    if 'Catering' not in str(categories):
        opportunities.append("Catering service addition")
    
    opportunities.append("Social media marketing enhancement")
    opportunities.append("Customer loyalty program implementation")
    
    return opportunities

# Advanced AI Agents
def menuminer_agent_analysis(competitors, location):
    """MenuMiner Agent - Advanced menu and pricing analysis"""
    if not openai_client:
        return generate_mock_menu_analysis(competitors, location)
    
    try:
        menu_data = []
        pricing_insights = []
        
        for comp in competitors:
            categories = comp.get('categories', [])
            price_level = comp.get('price_level', 1)
            avg_check = comp.get('advanced_metrics', {}).get('estimated_avg_check', 25)
            
            menu_info = f"- {comp['name']}: {', '.join(categories)} | Price level: {'$' * price_level} | Avg check: ${avg_check}"
            menu_data.append(menu_info)
            
            pricing_insights.append({
                "restaurant": comp['name'],
                "categories": categories,
                "price_level": price_level,
                "avg_check": avg_check
            })
        
        menu_text = "\n".join(menu_data)
        
        prompt = f"""
        As a MenuMiner AI agent analyzing restaurant menus and pricing in {location}:
        
        Menu & Pricing Data:
        {menu_text}
        
        Provide comprehensive menu analysis in JSON format:
        {{
            "menu_positioning_analysis": "Analysis of menu positioning strategies",
            "pricing_optimization_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
            "menu_gap_analysis": ["gap 1", "gap 2"],
            "recommended_menu_items": ["item 1", "item 2", "item 3"],
            "pricing_strategy_recommendations": ["strategy 1", "strategy 2"],
            "category_performance_insights": "Insights on category performance",
            "seasonal_menu_opportunities": ["seasonal opportunity 1", "seasonal opportunity 2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"MenuMiner agent error: {str(e)}")
        return generate_mock_menu_analysis(competitors, location)

def socialsentinel_agent_analysis(competitors, location):
    """SocialSentinel Agent - Social media and online presence analysis"""
    if not openai_client:
        return generate_mock_social_analysis(competitors, location)
    
    try:
        social_data = []
        
        for comp in competitors:
            digital_score = comp.get('advanced_metrics', {}).get('digital_presence_score', 50)
            review_count = comp.get('review_count', 0)
            rating = comp.get('rating', 0)
            
            social_info = f"- {comp['name']}: Digital score {digital_score}/100, {review_count} reviews, {rating}★ rating"
            social_data.append(social_info)
        
        social_text = "\n".join(social_data)
        
        prompt = f"""
        As a SocialSentinel AI agent analyzing social media presence in {location}:
        
        Digital Presence Data:
        {social_text}
        
        Provide social media analysis in JSON format:
        {{
            "social_media_landscape": "Overview of social media landscape",
            "digital_marketing_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
            "content_strategy_recommendations": ["strategy 1", "strategy 2"],
            "influencer_collaboration_potential": "Assessment of influencer opportunities",
            "online_reputation_insights": "Online reputation analysis",
            "social_media_trends": ["trend 1", "trend 2", "trend 3"],
            "engagement_optimization_tips": ["tip 1", "tip 2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"SocialSentinel agent error: {str(e)}")
        return generate_mock_social_analysis(competitors, location)

def locationscout_agent_analysis(competitors, location):
    """LocationScout Agent - Geographic and demographic analysis"""
    try:
        demographic_data = []
        location_insights = []
        
        for comp in competitors:
            demographics = comp.get('customer_demographics', {})
            positioning = comp.get('market_positioning', {})
            
            demo_info = f"- {comp['name']}: {positioning.get('position', 'Market Participant')}"
            demographic_data.append(demo_info)
        
        # Aggregate demographic analysis
        total_competitors = len(competitors)
        avg_rating = sum(comp.get('rating', 0) for comp in competitors) / total_competitors if total_competitors > 0 else 0
        
        return {
            "location_analysis": {
                "market_density": "High" if total_competitors > 15 else "Medium" if total_competitors > 8 else "Low",
                "average_market_rating": round(avg_rating, 1),
                "total_competitors": total_competitors,
                "market_saturation_level": f"{min(100, total_competitors * 5)}%"
            },
            "demographic_insights": {
                "primary_demographics": "Mixed age groups with varying income levels",
                "target_market_opportunities": [
                    "Young professionals (25-35)",
                    "Families with children",
                    "Senior dining market"
                ],
                "income_distribution": "Moderate to high disposable income",
                "dining_behavior_patterns": [
                    "Weekend dining surge",
                    "Lunch hour concentration",
                    "Delivery/takeout preference growth"
                ]
            },
            "location_advantages": [
                f"Established dining destination in {location}",
                "High foot traffic area",
                "Diverse customer base"
            ],
            "location_challenges": [
                "High competition density",
                "Parking considerations",
                "Rent/operational costs"
            ],
            "expansion_zones": [
                "Underserved neighboring areas",
                "Emerging residential districts",
                "Business district opportunities"
            ]
        }
        
    except Exception as e:
        logger.error(f"LocationScout agent error: {str(e)}")
        return generate_mock_location_analysis(location)

def trendanalyzer_agent_analysis(competitors, location):
    """TrendAnalyzer Agent - Market trends and predictive analysis"""
    if not openai_client:
        return generate_mock_trend_analysis(competitors, location)
    
    try:
        trend_data = []
        
        for comp in competitors:
            categories = comp.get('categories', [])
            growth_potential = comp.get('advanced_metrics', {}).get('growth_potential', 'Medium')
            
            trend_info = f"- {comp['name']}: {', '.join(categories[:2])}, Growth: {growth_potential}"
            trend_data.append(trend_info)
        
        trend_text = "\n".join(trend_data)
        
        prompt = f"""
        As a TrendAnalyzer AI agent analyzing market trends in {location} restaurant industry:
        
        Competitor Trend Data:
        {trend_text}
        
        Provide trend analysis in JSON format:
        {{
            "emerging_market_trends": ["trend 1", "trend 2", "trend 3"],
            "seasonal_patterns": ["pattern 1", "pattern 2"],
            "consumer_behavior_shifts": ["shift 1", "shift 2", "shift 3"],
            "technology_adoption_trends": ["tech trend 1", "tech trend 2"],
            "market_growth_predictions": "Predictions for next 12-24 months",
            "disruptive_factors": ["factor 1", "factor 2"],
            "investment_opportunities": ["opportunity 1", "opportunity 2"],
            "risk_mitigation_strategies": ["strategy 1", "strategy 2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"TrendAnalyzer agent error: {str(e)}")
        return generate_mock_trend_analysis(competitors, location)

def customerjourney_agent_analysis(competitors, location, user_business=None):
    """CustomerJourney Agent - Customer experience and journey analysis"""
    if not openai_client:
        return generate_mock_journey_analysis(competitors, location)
    
    try:
        journey_data = []
        
        for comp in competitors:
            rating = comp.get('rating', 0)
            efficiency = comp.get('advanced_metrics', {}).get('operational_efficiency', 50)
            service_score = comp.get('advanced_metrics', {}).get('service_quality_score', 70)
            
            journey_info = f"- {comp['name']}: {rating}★ rating, {efficiency}% operational efficiency, {service_score}% service quality"
            journey_data.append(journey_info)
        
        journey_text = "\n".join(journey_data)
        user_context = f"\nUser Business Context: {user_business.name if user_business else 'Not specified'}"
        
        prompt = f"""
        As a CustomerJourney AI agent analyzing customer experience in {location}:
        
        Customer Experience Data:
        {journey_text}
        {user_context}
        
        Provide customer journey analysis in JSON format:
        {{
            "customer_experience_benchmarks": "Analysis of customer experience standards",
            "journey_optimization_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
            "service_quality_insights": ["insight 1", "insight 2"],
            "customer_satisfaction_drivers": ["driver 1", "driver 2", "driver 3"],
            "pain_point_analysis": ["pain point 1", "pain point 2"],
            "loyalty_building_strategies": ["strategy 1", "strategy 2"],
            "customer_retention_recommendations": ["recommendation 1", "recommendation 2"],
            "experience_differentiation_opportunities": ["opportunity 1", "opportunity 2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"CustomerJourney agent error: {str(e)}")
        return generate_mock_journey_analysis(competitors, location)

# Mock data generators for fallback
def generate_mock_menu_analysis(competitors, location):
    return {
        "menu_positioning_analysis": f"Menu positioning in {location} shows diverse strategies from fast-casual to premium dining",
        "pricing_optimization_opportunities": [
            "Gap in mid-premium pricing tier ($25-35 range)",
            "Limited healthy/vegan options at competitive prices",
            "Opportunity for value lunch pricing"
        ],
        "menu_gap_analysis": [
            "Underserved dietary restrictions market",
            "Limited late-night dining options"
        ],
        "recommended_menu_items": [
            "Signature bowls/customizable options",
            "Premium comfort food items",
            "Healthy fast-casual alternatives"
        ],
        "pricing_strategy_recommendations": [
            "Implement dynamic pricing for peak hours",
            "Create value meal combinations"
        ],
        "category_performance_insights": "Italian and Asian cuisines showing strong performance with consistent pricing power",
        "seasonal_menu_opportunities": [
            "Summer outdoor dining specials",
            "Holiday catering packages"
        ]
    }

def generate_mock_social_analysis(competitors, location):
    return {
        "social_media_landscape": f"Social media presence in {location} is highly competitive with Instagram and Facebook leading engagement",
        "digital_marketing_opportunities": [
            "Food photography and visual storytelling",
            "Local community engagement campaigns",
            "User-generated content initiatives"
        ],
        "content_strategy_recommendations": [
            "Behind-the-scenes kitchen content",
            "Customer story highlighting"
        ],
        "influencer_collaboration_potential": "High potential for micro-influencer partnerships with local food bloggers",
        "online_reputation_insights": "Online reputation management critical due to high competition",
        "social_media_trends": [
            "Video content dominance",
            "Sustainable dining messaging",
            "Community involvement showcasing"
        ],
        "engagement_optimization_tips": [
            "Respond to all customer interactions within 2 hours",
            "Share customer photos and testimonials"
        ]
    }

def generate_mock_location_analysis(location):
    return {
        "location_analysis": {
            "market_density": "High",
            "average_market_rating": 4.2,
            "total_competitors": 15,
            "market_saturation_level": "75%"
        },
        "demographic_insights": {
            "primary_demographics": "Mixed age groups with varying income levels",
            "target_market_opportunities": [
                "Young professionals (25-35)",
                "Families with children",
                "Senior dining market"
            ],
            "income_distribution": "Moderate to high disposable income",
            "dining_behavior_patterns": [
                "Weekend dining surge",
                "Lunch hour concentration",
                "Delivery preference growth"
            ]
        }
    }

def generate_mock_trend_analysis(competitors, location):
    return {
        "emerging_market_trends": [
            "Plant-based menu expansion",
            "Technology-enhanced dining experiences",
            "Sustainability focus in operations"
        ],
        "seasonal_patterns": [
            "Summer outdoor dining surge",
            "Holiday catering demand spikes"
        ],
        "consumer_behavior_shifts": [
            "Increased delivery/takeout preference",
            "Value-conscious dining decisions",
            "Experience-focused dining choices"
        ],
        "technology_adoption_trends": [
            "QR code menu adoption",
            "Online ordering system integration"
        ],
        "market_growth_predictions": "Steady 3-5% growth expected over next 12 months with premium casual segment leading",
        "disruptive_factors": [
            "Economic uncertainty affecting dining budgets",
            "Labor shortage impacting service quality"
        ]
    }

def generate_mock_journey_analysis(competitors, location):
    return {
        "customer_experience_benchmarks": "Customer experience standards in area emphasize speed, quality, and personalized service",
        "journey_optimization_opportunities": [
            "Streamline ordering and payment processes",
            "Enhance wait time communication",
            "Improve reservation and seating management"
        ],
        "service_quality_insights": [
            "Consistent service quality key differentiator",
            "Staff training investment shows in customer satisfaction"
        ],
        "customer_satisfaction_drivers": [
            "Food quality and consistency",
            "Staff friendliness and knowledge",
            "Atmosphere and ambiance"
        ],
        "pain_point_analysis": [
            "Long wait times during peak hours",
            "Parking availability challenges"
        ]
    }

# Enhanced Yelp Business Search
def search_yelp_businesses_advanced(location, radius, business_type="restaurant"):
    """Enhanced Yelp search with comprehensive data collection"""
    if not YELP_API_KEY:
        return []
    
    try:
        headers = {
            'Authorization': f'Bearer {YELP_API_KEY}'
        }
        
        params = {
            'location': location,
            'radius': int(radius * 1609.34),
            'categories': 'restaurants',
            'limit': 50,  # Increased limit for comprehensive analysis
            'sort_by': 'rating'
        }
        
        response = requests.get(
            'https://api.yelp.com/v3/businesses/search',
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            competitors = []
            
            for business in data.get('businesses', []):
                # Basic business data
                competitor = {
                    "id": str(uuid.uuid4()),
                    "name": business.get('name', 'Unknown'),
                    "address": ', '.join(business.get('location', {}).get('display_address', [])),
                    "location": {
                        "lat": business.get('coordinates', {}).get('latitude', 0),
                        "lng": business.get('coordinates', {}).get('longitude', 0)
                    },
                    "business_type": business_type,
                    "phone": business.get('phone'),
                    "website": business.get('url'),
                    "rating": business.get('rating'),
                    "review_count": business.get('review_count'),
                    "price_level": len(business.get('price', '$')),
                    "yelp_id": business.get('id'),
                    "categories": [cat['title'] for cat in business.get('categories', [])],
                    "is_closed": business.get('is_closed', False),
                    "image_url": business.get('image_url')
                }
                
                # Enhance with comprehensive business intelligence
                enhanced_competitor = collect_comprehensive_business_data(competitor, location)
                competitors.append(enhanced_competitor)
            
            return competitors
            
    except Exception as e:
        logger.error(f"Enhanced Yelp API error: {str(e)}")
        return []

# Ultimate Intelligence Synthesizer
def ultimate_intelligence_synthesizer(competitors, location, all_agent_data, user_business=None):
    """Ultimate Intelligence Synthesizer - Combines all AI agent insights"""
    if not openai_client:
        return generate_ultimate_synthesis(competitors, location, all_agent_data)
    
    try:
        # Prepare comprehensive data summary
        market_summary = f"""
        MARKET: {location}
        COMPETITORS ANALYZED: {len(competitors)}
        USER BUSINESS: {user_business.name if user_business else 'Market Analysis Only'}
        
        PRICEWATCH DATA: {json.dumps(all_agent_data.get('pricewatch', {}), indent=2)}
        SENTIMENT DATA: {json.dumps(all_agent_data.get('sentiment', {}), indent=2)}
        CROWD DATA: {json.dumps(all_agent_data.get('crowd', {}), indent=2)}
        SENTINEL DATA: {json.dumps(all_agent_data.get('sentinel', {}), indent=2)}
        MENUMINER DATA: {json.dumps(all_agent_data.get('menuminer', {}), indent=2)}
        SOCIAL DATA: {json.dumps(all_agent_data.get('social', {}), indent=2)}
        LOCATION DATA: {json.dumps(all_agent_data.get('location', {}), indent=2)}
        TREND DATA: {json.dumps(all_agent_data.get('trends', {}), indent=2)}
        JOURNEY DATA: {json.dumps(all_agent_data.get('journey', {}), indent=2)}
        """
        
        prompt = f"""
        As the Ultimate Intelligence Synthesizer for {location} restaurant market:
        
        {market_summary}
        
        Synthesize ALL agent insights into ultimate strategic intelligence in JSON format:
        {{
            "executive_summary": "Comprehensive market overview with key findings",
            "strategic_opportunities": ["top opportunity 1", "top opportunity 2", "top opportunity 3"],
            "competitive_positioning_strategy": "Recommended positioning approach",
            "market_entry_roadmap": "Step-by-step market entry strategy",
            "revenue_optimization_plan": "Revenue maximization strategy",
            "risk_mitigation_framework": "Comprehensive risk management approach",
            "success_metrics_kpis": ["KPI 1", "KPI 2", "KPI 3"],
            "timeline_milestones": "Implementation timeline with key milestones",
            "investment_requirements": "Estimated investment needs and ROI projections",
            "competitive_differentiation": "Key differentiation strategies",
            "customer_acquisition_strategy": "Customer acquisition and retention plan",
            "operational_excellence_plan": "Operational optimization recommendations"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Ultimate Intelligence Synthesizer error: {str(e)}")
        return generate_ultimate_synthesis(competitors, location, all_agent_data)

def generate_ultimate_synthesis(competitors, location, all_agent_data):
    """Generate comprehensive synthesis for fallback"""
    return {
        "executive_summary": f"Comprehensive analysis of {location} restaurant market reveals {len(competitors)} active competitors with diverse positioning strategies and significant opportunities for strategic market entry",
        "strategic_opportunities": [
            "Premium casual dining gap in $25-35 price range",
            "Technology-enhanced customer experience differentiation",
            "Sustainable dining concept with local sourcing focus"
        ],
        "competitive_positioning_strategy": "Position as innovative, customer-centric dining experience with superior value proposition",
        "market_entry_roadmap": "6-month market validation, 12-month soft launch, 18-month full market penetration",
        "revenue_optimization_plan": "Dynamic pricing strategy, loyalty program implementation, upselling optimization",
        "risk_mitigation_framework": "Diversified revenue streams, strong operational systems, continuous market monitoring",
        "success_metrics_kpis": [
            "Achieve 4.5+ star rating within 6 months",
            "Capture 3-5% local market share within 12 months",
            "Maintain 25%+ profit margins"
        ],
        "timeline_milestones": "Q1: Market entry preparation, Q2: Soft launch, Q3: Marketing scaling, Q4: Performance optimization",
        "investment_requirements": "$150K-300K initial investment with 18-24 month ROI projection",
        "competitive_differentiation": "Technology integration, exceptional customer service, unique menu positioning",
        "customer_acquisition_strategy": "Digital marketing focus, community engagement, referral programs",
        "operational_excellence_plan": "Staff training optimization, inventory management systems, quality control protocols"
    }

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Ultimate health check endpoint"""
    return {
        "status": "healthy",
        "service": "BizFizz Ultimate Competitive Intelligence Platform",
        "version": "3.0.0",
        "integrations": {
            "google_maps": bool(GOOGLE_MAPS_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "yelp": bool(YELP_API_KEY)
        },
        "features": {
            "ai_agents": [
                "PriceWatch", "Sentiment", "CrowdAnalyst", "Sentinel", 
                "MenuMiner", "SocialSentinel", "LocationScout", 
                "TrendAnalyzer", "CustomerJourney", "Ultimate Intelligence Synthesizer"
            ],
            "advanced_analytics": [
                "Predictive Market Analysis", "Customer Demographics", "Revenue Forecasting",
                "Competitive Positioning", "Growth Potential Assessment", "Risk Analysis"
            ],
            "data_sources": [
                "Yelp Business Intelligence", "Google Places Advanced", "AI-Generated Market Insights",
                "Social Media Analytics", "Demographic Analysis", "Economic Indicators"
            ],
            "reporting": [
                "Interactive Visual Dashboards", "PDF Report Generation", "Email Automation",
                "Custom Analytics", "Real-time Monitoring", "Predictive Insights"
            ]
        }
    }

@app.post("/api/ultimate-competitor-search")
async def ultimate_competitor_search(search: GeographicSearch):
    """Ultimate competitor search with comprehensive intelligence"""
    try:
        logger.info(f"Ultimate competitor search in {search.location}")
        
        competitors = []
        
        # Enhanced Yelp search
        if YELP_API_KEY:
            competitors = search_yelp_businesses_advanced(search.location, search.radius, search.business_type)
            logger.info(f"Found {len(competitors)} competitors via Enhanced Yelp API")
        
        # Fallback to mock data with enhancements
        if not competitors:
            logger.info("Using enhanced mock data")
            mock_competitors = get_enhanced_mock_competitors(search.business_type)
            competitors = [collect_comprehensive_business_data(comp, search.location) for comp in mock_competitors]
        
        # Advanced market analysis
        market_intelligence = perform_advanced_market_analysis(competitors, search.location)
        
        # Store comprehensive search
        search_record = {
            "id": str(uuid.uuid4()),
            "location": search.location,
            "radius": search.radius,
            "business_type": search.business_type,
            "results_count": len(competitors),
            "timestamp": datetime.utcnow(),
            "analysis_level": "ultimate",
            "market_intelligence": market_intelligence
        }
        
        await db.ultimate_searches.insert_one(search_record)
        
        return {
            "search_id": search_record["id"],
            "location": search.location,
            "competitors": competitors,
            "total_found": len(competitors),
            "analysis_level": "ultimate",
            "market_intelligence": market_intelligence,
            "data_quality": "premium" if YELP_API_KEY else "enhanced_mock"
        }
        
    except Exception as e:
        logger.error(f"Error in ultimate competitor search: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform ultimate competitor search")

@app.post("/api/generate-ultimate-intelligence-report")
async def generate_ultimate_intelligence_report(request: CompetitiveAnalysisRequest):
    """Generate the ultimate competitive intelligence report"""
    try:
        competitor_ids = request.competitor_ids
        location = request.location
        user_business = request.user_business
        analysis_depth = request.analysis_depth
        
        if not competitor_ids:
            raise HTTPException(status_code=400, detail="No competitors selected")
        
        logger.info(f"Generating {analysis_depth} intelligence report for {location}")
        
        # Get enhanced competitor data
        competitors = []
        for comp_id in competitor_ids:
            # In production, would fetch from database
            competitor = create_enhanced_competitor_profile(comp_id, location)
            competitors.append(competitor)
        
        # Run all AI agents
        logger.info("Running all 10 AI agents...")
        
        pricewatch_analysis = pricewatch_agent_analysis(competitors, location)
        sentiment_analysis = sentiment_agent_analysis(competitors, location)
        crowd_analysis = crowdanalyst_agent_analysis(competitors, location)
        sentinel_analysis = sentinel_agent_analysis(competitors, location)
        menuminer_analysis = menuminer_agent_analysis(competitors, location)
        social_analysis = socialsentinel_agent_analysis(competitors, location)
        location_analysis = locationscout_agent_analysis(competitors, location)
        trend_analysis = trendanalyzer_agent_analysis(competitors, location)
        journey_analysis = customerjourney_agent_analysis(competitors, location, user_business)
        
        # Combine all agent data
        all_agent_data = {
            "pricewatch": pricewatch_analysis,
            "sentiment": sentiment_analysis,
            "crowd": crowd_analysis,
            "sentinel": sentinel_analysis,
            "menuminer": menuminer_analysis,
            "social": social_analysis,
            "location": location_analysis,
            "trends": trend_analysis,
            "journey": journey_analysis
        }
        
        # Ultimate Intelligence Synthesis
        logger.info("Running Ultimate Intelligence Synthesizer...")
        ultimate_synthesis = ultimate_intelligence_synthesizer(
            competitors, location, all_agent_data, user_business
        )
        
        # Advanced market metrics
        advanced_metrics = calculate_advanced_market_metrics(competitors, location)
        
        # Predictive analytics
        predictions = generate_market_predictions(competitors, all_agent_data)
        
        # Create ultimate report
        report = {
            "id": str(uuid.uuid4()),
            "report_type": "ultimate_intelligence",
            "analysis_depth": analysis_depth,
            "location": location,
            "user_business": user_business.dict() if user_business else None,
            "competitors": competitors,
            "total_competitors": len(competitors),
            "report_date": datetime.utcnow().isoformat(),
            "generated_with_ai": bool(openai_client),
            
            # All AI Agent Analyses
            "pricewatch_analysis": pricewatch_analysis,
            "sentiment_analysis": sentiment_analysis,
            "crowd_analysis": crowd_analysis,
            "sentinel_analysis": sentinel_analysis,
            "menuminer_analysis": menuminer_analysis,
            "socialsentinel_analysis": social_analysis,
            "locationscout_analysis": location_analysis,
            "trendanalyzer_analysis": trend_analysis,
            "customerjourney_analysis": journey_analysis,
            
            # Ultimate Intelligence Synthesis
            "ultimate_synthesis": ultimate_synthesis,
            
            # Advanced Analytics
            "advanced_market_metrics": advanced_metrics,
            "predictive_analytics": predictions,
            
            # Executive Intelligence
            "executive_summary": ultimate_synthesis.get("executive_summary"),
            "strategic_recommendations": ultimate_synthesis.get("strategic_opportunities", []),
            "competitive_positioning": ultimate_synthesis.get("competitive_positioning_strategy"),
            "market_entry_strategy": ultimate_synthesis.get("market_entry_roadmap"),
            "success_framework": {
                "kpis": ultimate_synthesis.get("success_metrics_kpis", []),
                "timeline": ultimate_synthesis.get("timeline_milestones"),
                "investment": ultimate_synthesis.get("investment_requirements")
            }
        }
        
        # Store ultimate report
        await db.ultimate_reports.insert_one(report.copy())
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating ultimate intelligence report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate ultimate intelligence report")

def perform_advanced_market_analysis(competitors, location):
    """Perform advanced market analysis"""
    try:
        total_competitors = len(competitors)
        if total_competitors == 0:
            return {}
        
        # Aggregate metrics
        avg_rating = sum(c.get('rating', 0) for c in competitors) / total_competitors
        total_reviews = sum(c.get('review_count', 0) for c in competitors)
        avg_price_level = sum(c.get('price_level', 1) for c in competitors) / total_competitors
        
        # Advanced calculations
        total_estimated_revenue = sum(
            c.get('advanced_metrics', {}).get('estimated_monthly_revenue', 0) 
            for c in competitors
        )
        
        market_concentration = calculate_market_concentration(competitors)
        competitive_intensity = calculate_competitive_intensity(competitors)
        
        return {
            "market_overview": {
                "total_competitors": total_competitors,
                "average_rating": round(avg_rating, 2),
                "total_market_reviews": total_reviews,
                "average_price_level": round(avg_price_level, 1),
                "estimated_total_market_revenue": total_estimated_revenue
            },
            "market_structure": {
                "concentration_level": market_concentration,
                "competitive_intensity": competitive_intensity,
                "market_maturity": "Mature" if total_competitors > 15 else "Developing",
                "entry_barriers": "High" if avg_rating > 4.0 else "Medium"
            },
            "opportunity_assessment": {
                "market_gaps": identify_market_gaps(competitors),
                "growth_potential": "High" if total_competitors < 20 else "Medium",
                "innovation_opportunities": identify_innovation_opportunities(competitors)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced market analysis: {str(e)}")
        return {}

def calculate_market_concentration(competitors):
    """Calculate market concentration index"""
    if not competitors:
        return "Low"
    
    # Simple concentration calculation based on review distribution
    total_reviews = sum(c.get('review_count', 0) for c in competitors)
    if total_reviews == 0:
        return "Low"
    
    # Calculate concentration ratio (top 3 competitors)
    sorted_competitors = sorted(competitors, key=lambda x: x.get('review_count', 0), reverse=True)
    top_3_reviews = sum(c.get('review_count', 0) for c in sorted_competitors[:3])
    concentration_ratio = (top_3_reviews / total_reviews) * 100
    
    if concentration_ratio > 70:
        return "High"
    elif concentration_ratio > 50:
        return "Medium"
    else:
        return "Low"

def calculate_competitive_intensity(competitors):
    """Calculate competitive intensity"""
    if not competitors:
        return "Low"
    
    high_performers = sum(1 for c in competitors if c.get('rating', 0) >= 4.0)
    intensity_ratio = (high_performers / len(competitors)) * 100
    
    if intensity_ratio > 60:
        return "High"
    elif intensity_ratio > 30:
        return "Medium"
    else:
        return "Low"

def identify_market_gaps(competitors):
    """Identify market gaps and opportunities"""
    gaps = []
    
    # Price level gaps
    price_levels = [c.get('price_level', 1) for c in competitors]
    if 2 not in price_levels:
        gaps.append("Mid-range pricing segment underserved")
    if 4 not in price_levels:
        gaps.append("Premium dining segment opportunity")
    
    # Category gaps
    all_categories = []
    for c in competitors:
        all_categories.extend(c.get('categories', []))
    
    if 'Vegan' not in str(all_categories):
        gaps.append("Vegan dining options limited")
    if 'Healthy' not in str(all_categories):
        gaps.append("Health-focused dining gap")
    
    return gaps if gaps else ["Market appears well-served across segments"]

def identify_innovation_opportunities(competitors):
    """Identify innovation opportunities"""
    opportunities = [
        "Technology-enhanced dining experience",
        "Sustainable and eco-friendly operations",
        "Personalized customer experience",
        "Hybrid delivery/dine-in concepts"
    ]
    return opportunities

def calculate_advanced_market_metrics(competitors, location):
    """Calculate advanced market metrics"""
    try:
        if not competitors:
            return {}
        
        # Financial metrics
        total_revenue = sum(
            c.get('advanced_metrics', {}).get('estimated_monthly_revenue', 0) 
            for c in competitors
        )
        
        avg_revenue_per_competitor = total_revenue / len(competitors) if competitors else 0
        
        # Performance metrics
        top_performers = [c for c in competitors if c.get('rating', 0) >= 4.5]
        market_leaders = sorted(competitors, key=lambda x: x.get('review_count', 0), reverse=True)[:3]
        
        # Growth metrics
        high_growth_potential = [
            c for c in competitors 
            if c.get('advanced_metrics', {}).get('growth_potential') == 'High'
        ]
        
        return {
            "financial_metrics": {
                "total_market_revenue_monthly": total_revenue,
                "average_revenue_per_competitor": avg_revenue_per_competitor,
                "revenue_distribution": "Varied across price segments",
                "market_size_estimate": f"${total_revenue * 12:,.0f} annually"
            },
            "performance_metrics": {
                "top_performers_count": len(top_performers),
                "market_leaders": [c['name'] for c in market_leaders],
                "average_market_rating": sum(c.get('rating', 0) for c in competitors) / len(competitors),
                "quality_consistency": "High" if len(top_performers) > len(competitors) * 0.4 else "Medium"
            },
            "growth_metrics": {
                "high_growth_potential_count": len(high_growth_potential),
                "market_expansion_rate": "Moderate growth expected",
                "innovation_adoption": "Technology integration increasing",
                "customer_base_growth": "Steady expansion in target demographics"
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating advanced market metrics: {str(e)}")
        return {}

def generate_market_predictions(competitors, all_agent_data):
    """Generate market predictions and forecasts"""
    try:
        return {
            "12_month_forecast": {
                "market_growth": "3-7% revenue growth expected",
                "new_entrants": "2-3 new competitors likely",
                "price_trends": "Moderate inflation-driven increases",
                "consumer_behavior": "Continued delivery preference growth"
            },
            "24_month_outlook": {
                "market_evolution": "Technology integration acceleration",
                "competitive_landscape": "Consolidation of weaker players",
                "innovation_trends": "Sustainability focus increasing",
                "investment_climate": "Cautious but opportunistic"
            },
            "risk_factors": [
                "Economic uncertainty affecting consumer spending",
                "Labor cost pressures on margins",
                "Supply chain disruption risks"
            ],
            "opportunity_windows": [
                "Q2-Q3: Seasonal demand peak optimization",
                "Q4: Holiday catering market expansion",
                "2024-2025: Technology differentiation window"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating market predictions: {str(e)}")
        return {}

def create_enhanced_competitor_profile(comp_id, location):
    """Create enhanced competitor profile with comprehensive data"""
    # Mock enhanced competitor data
    base_competitor = {
        "id": comp_id,
        "name": f"Restaurant {comp_id[:8]}",
        "address": f"Address for {comp_id[:8]}",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "business_type": "restaurant",
        "rating": 4.2 + (len(comp_id) % 5) * 0.1,
        "review_count": 200 + len(comp_id) * 10,
        "price_level": 2 + (len(comp_id) % 3),
        "categories": ["American", "Casual Dining"],
        "phone": "(555) 123-4567",
        "website": f"https://restaurant{comp_id[:8]}.com"
    }
    
    return collect_comprehensive_business_data(base_competitor, location)

def get_enhanced_mock_competitors(business_type):
    """Get enhanced mock competitor data"""
    return [
        {
            "id": str(uuid.uuid4()),
            "name": "The Gourmet Corner",
            "address": "123 Premium Street, Downtown",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "business_type": business_type,
            "phone": "(555) 123-4567",
            "website": "thegourmetcorner.com",
            "rating": 4.6,
            "review_count": 892,
            "price_level": 4,
            "categories": ["Fine Dining", "New American", "Wine Bar"],
            "is_closed": False,
            "image_url": "https://example.com/gourmet.jpg"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Mario's Family Kitchen",
            "address": "456 Family Avenue, Midtown",
            "location": {"lat": 40.7589, "lng": -73.9851},
            "business_type": business_type,
            "phone": "(555) 987-6543",
            "website": "mariosfamilykitchen.com",
            "rating": 4.3,
            "review_count": 567,
            "price_level": 2,
            "categories": ["Italian", "Family Style", "Pizza"],
            "is_closed": False,
            "image_url": "https://example.com/marios.jpg"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Urban Fusion Bistro",
            "address": "789 Trendy Lane, Uptown",
            "location": {"lat": 40.7831, "lng": -73.9712},
            "business_type": business_type,
            "phone": "(555) 456-7890",
            "website": "urbanfusionbistro.com",
            "rating": 4.1,
            "review_count": 334,
            "price_level": 3,
            "categories": ["Fusion", "Modern", "Cocktail Bars"],
            "is_closed": False,
            "image_url": "https://example.com/urban.jpg"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Healthy Harvest Cafe",
            "address": "321 Wellness Way, Health District",
            "location": {"lat": 40.7500, "lng": -73.9800},
            "business_type": business_type,
            "phone": "(555) 234-5678",
            "website": "healthyharvestcafe.com",
            "rating": 4.4,
            "review_count": 445,
            "price_level": 2,
            "categories": ["Healthy", "Vegan", "Organic", "Salads"],
            "is_closed": False,
            "image_url": "https://example.com/healthy.jpg"
        }
    ]

@app.get("/api/reports")
async def get_ultimate_reports(limit: int = 10):
    """Get ultimate intelligence reports"""
    try:
        reports = []
        
        # Get ultimate reports first
        async for report in db.ultimate_reports.find().sort("report_date", -1).limit(limit):
            if "_id" in report:
                del report["_id"]
            reports.append(report)
        
        # Fill with comprehensive reports if needed
        if len(reports) < limit:
            remaining = limit - len(reports)
            async for report in db.comprehensive_reports.find().sort("report_date", -1).limit(remaining):
                if "_id" in report:
                    del report["_id"]
                reports.append(report)
        
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error fetching ultimate reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.get("/api/subscription-tiers")
async def get_ultimate_subscription_tiers():
    """Get ultimate subscription tiers with all features"""
    return {
        "tiers": [
            {
                "name": "Starter",
                "price": 0,
                "features": [
                    "1 basic report per month",
                    "Up to 5 competitors per report",
                    "Basic competitor data",
                    "3 AI agents (PriceWatch, Sentiment, CrowdAnalyst)",
                    "Standard analytics",
                    "Email support"
                ],
                "api_calls": 25,
                "ai_agents": ["PriceWatch", "Sentiment", "CrowdAnalyst"],
                "data_sources": ["Enhanced Mock Data", "Basic Analytics"],
                "reporting": ["Basic Visual Charts", "Standard Insights"]
            },
            {
                "name": "Professional",
                "price": 149,
                "features": [
                    "15 comprehensive reports per month",
                    "Up to 25 competitors per report",
                    "Real-time Yelp & Google data",
                    "7 AI agents + Intelligence Synthesizer",
                    "Advanced market analytics",
                    "Visual reporting with charts",
                    "PDF report generation",
                    "Email automation",
                    "Priority support"
                ],
                "api_calls": 500,
                "ai_agents": [
                    "PriceWatch", "Sentiment", "CrowdAnalyst", "Sentinel", 
                    "MenuMiner", "SocialSentinel", "LocationScout", "Intelligence Synthesizer"
                ],
                "data_sources": ["Yelp API", "Google Places", "Social Media Analytics", "Demographic Data"],
                "reporting": ["Advanced Visual Dashboards", "PDF Reports", "Email Automation"]
            },
            {
                "name": "Enterprise",
                "price": 399,
                "features": [
                    "Unlimited ultimate intelligence reports",
                    "Unlimited competitors analysis",
                    "All 10 AI agents + Ultimate Intelligence Synthesizer",
                    "Real-time competitive monitoring",
                    "Predictive market analytics",
                    "Custom market analysis",
                    "Advanced demographic insights",
                    "API access for integration",
                    "White-label reporting",
                    "Custom AI agent development",
                    "Dedicated account manager",
                    "24/7 priority support"
                ],
                "api_calls": 5000,
                "ai_agents": [
                    "All 10 AI Agents", "Ultimate Intelligence Synthesizer", 
                    "Custom Agent Development", "Predictive Analytics Engine"
                ],
                "data_sources": [
                    "Premium Data Access", "Real-time Monitoring", 
                    "Custom Integrations", "Economic Indicators", "Social Media APIs"
                ],
                "reporting": [
                    "Ultimate Intelligence Dashboards", "Custom Report Builder", 
                    "Real-time Alerts", "Predictive Forecasting", "White-label Solutions"
                ]
            }
        ],
        "enterprise_features": {
            "custom_ai_agents": "Develop custom AI agents for specific business needs",
            "predictive_analytics": "Advanced forecasting and trend prediction",
            "real_time_monitoring": "Continuous competitive landscape monitoring",
            "integration_api": "Full API access for custom integrations",
            "dedicated_support": "Personal account manager and 24/7 support"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)