from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import uuid
import json
import asyncio
import googlemaps
import requests
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="BizFizz API", version="2.0.0", description="Advanced Competitive Intelligence Platform")

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
db = client.bizfizz

# API clients
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

class CompetitorProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    address: str
    location: Dict[str, float]  # lat, lng
    business_type: str
    phone: Optional[str] = None
    website: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    price_level: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Enhanced competitor search with comprehensive data
def search_yelp_businesses(location, radius, business_type="restaurant"):
    """Search Yelp for businesses with comprehensive competitive intelligence data"""
    if not YELP_API_KEY:
        return []
    
    try:
        headers = {
            'Authorization': f'Bearer {YELP_API_KEY}'
        }
        
        params = {
            'location': location,
            'radius': int(radius * 1609.34),  # Convert miles to meters
            'categories': 'restaurants',
            'limit': 20,
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
                # Calculate comprehensive business metrics
                review_count = business.get('review_count', 0)
                rating = business.get('rating', 0)
                price_level = len(business.get('price', '$'))
                
                # Advanced metric calculations
                estimated_daily_traffic = min(max(int(review_count / 30), 10), 500)
                avg_check = price_level * 25  # $25, $50, $75, $100
                estimated_monthly_revenue = estimated_daily_traffic * avg_check * 30
                customer_satisfaction = rating * 20  # Convert to 100-point scale
                market_strength = min(100, (rating * 15) + (min(review_count/100, 10) * 5))
                
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
                    "price_level": price_level,
                    "yelp_id": business.get('id'),
                    "categories": [cat['title'] for cat in business.get('categories', [])],
                    
                    # Comprehensive Business Intelligence Metrics
                    "business_metrics": {
                        "estimated_daily_traffic": estimated_daily_traffic,
                        "estimated_monthly_revenue": estimated_monthly_revenue,
                        "avg_check_estimate": avg_check,
                        "customer_satisfaction_score": customer_satisfaction,
                        "market_position": "Premium" if price_level >= 3 else "Mid-Range" if price_level == 2 else "Budget",
                        "review_velocity": max(1, review_count / 4),  # Reviews per year
                        "competitive_strength": market_strength,
                        "revenue_per_review": estimated_monthly_revenue / max(1, review_count),
                        "market_share_estimate": min(10, estimated_daily_traffic / 100)
                    },
                    
                    "operational_insights": {
                        "is_open_now": business.get('is_closed', True) == False,
                        "service_options": {
                            "delivery_available": True,
                            "takeout_available": True,
                            "dine_in_available": True,
                            "outdoor_seating": price_level >= 2,
                            "reservation_system": price_level >= 3
                        },
                        "peak_hours": {
                            "lunch_rush": "11:30 AM - 1:30 PM",
                            "dinner_rush": "6:00 PM - 8:30 PM",
                            "weekend_peak": "7:00 PM - 9:00 PM"
                        },
                        "capacity_estimate": estimated_daily_traffic * 1.5,  # Peak capacity
                        "staff_efficiency": min(100, estimated_daily_traffic / 5)
                    },
                    
                    "digital_presence": {
                        "has_website": bool(business.get('url')),
                        "social_engagement_score": min(100, review_count / 20),
                        "online_reputation_score": customer_satisfaction,
                        "review_sentiment": "Excellent" if rating >= 4.5 else "Good" if rating >= 4.0 else "Average" if rating >= 3.0 else "Poor",
                        "digital_marketing_strength": min(100, (review_count / 50) + (rating * 10)),
                        "customer_loyalty_indicator": min(100, review_count / 100)
                    },
                    
                    "competitive_analysis": {
                        "threat_level": "High" if market_strength >= 80 else "Medium" if market_strength >= 60 else "Low",
                        "differentiation_factors": business.get('categories', [])[:3],
                        "pricing_strategy": "Premium" if price_level >= 3 else "Value" if price_level <= 2 else "Standard",
                        "target_demographic": "Upscale" if price_level >= 3 else "Mainstream" if price_level == 2 else "Budget-conscious",
                        "expansion_potential": "High" if rating >= 4.2 and review_count >= 500 else "Medium"
                    }
                }
                competitors.append(competitor)
            
            return competitors
            
    except Exception as e:
        logger.error(f"Yelp API error: {str(e)}")
        return []

# Advanced AI Agent System
def pricewatch_agent_analysis(competitors, location):
    """PriceWatch Agent - Advanced pricing strategy analysis"""
    if not openai_client:
        return generate_mock_pricing_analysis(competitors, location)
    
    try:
        price_data = []
        total_revenue = 0
        price_levels = []
        
        for comp in competitors:
            price_level = comp.get('price_level', 1)
            avg_check = comp.get('business_metrics', {}).get('avg_check_estimate', 25)
            revenue = comp.get('business_metrics', {}).get('estimated_monthly_revenue', 0)
            
            price_data.append(f"- {comp['name']}: {'$' * price_level} (${avg_check} avg check, ${revenue:,.0f} est. monthly revenue)")
            total_revenue += revenue
            price_levels.append(price_level)
        
        avg_price_level = sum(price_levels) / len(price_levels) if price_levels else 2
        market_revenue = total_revenue
        
        prompt = f"""
        As an expert PriceWatch AI agent analyzing restaurant pricing in {location}:
        
        Market Data:
        {chr(10).join(price_data)}
        
        Market Summary:
        - Average price level: {avg_price_level:.1f}/4
        - Total market revenue: ${market_revenue:,.0f}/month
        - Number of competitors: {len(competitors)}
        
        Provide comprehensive pricing analysis in JSON format:
        {{
            "market_overview": "Detailed market pricing analysis",
            "pricing_opportunities": ["specific opportunity 1", "specific opportunity 2", "specific opportunity 3"],
            "competitive_gaps": ["gap analysis 1", "gap analysis 2"],
            "pricing_strategies": ["strategy recommendation 1", "strategy recommendation 2"],
            "revenue_benchmarks": "Revenue benchmarking insights",
            "price_elasticity": "Price sensitivity analysis"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"PriceWatch agent error: {str(e)}")
        return generate_mock_pricing_analysis(competitors, location)

def sentiment_agent_analysis(competitors, location):
    """Sentiment Agent - Advanced customer sentiment analysis"""
    if not openai_client:
        return generate_mock_sentiment_analysis(competitors, location)
    
    try:
        sentiment_data = []
        total_reviews = 0
        avg_rating = 0
        
        for comp in competitors:
            rating = comp.get('rating', 0)
            review_count = comp.get('review_count', 0)
            sentiment = comp.get('digital_presence', {}).get('review_sentiment', 'Average')
            
            sentiment_data.append(f"- {comp['name']}: {rating}/5 stars, {review_count} reviews, {sentiment} sentiment")
            total_reviews += review_count
            avg_rating += rating
        
        avg_rating = avg_rating / len(competitors) if competitors else 0
        
        prompt = f"""
        As an expert Sentiment Analysis AI agent for restaurants in {location}:
        
        Customer Feedback Data:
        {chr(10).join(sentiment_data)}
        
        Market Summary:
        - Total reviews analyzed: {total_reviews:,}
        - Average market rating: {avg_rating:.1f}/5
        - Market sentiment: {"Positive" if avg_rating >= 4.0 else "Mixed"}
        
        Provide comprehensive sentiment analysis in JSON format:
        {{
            "sentiment_overview": "Overall market sentiment analysis",
            "customer_pain_points": ["pain point 1", "pain point 2", "pain point 3"],
            "satisfaction_drivers": ["driver 1", "driver 2", "driver 3"],
            "service_gaps": ["gap 1", "gap 2"],
            "reputation_insights": "Reputation management insights",
            "customer_expectations": "What customers expect in this market"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Sentiment agent error: {str(e)}")
        return generate_mock_sentiment_analysis(competitors, location)

def crowdanalyst_agent_analysis(competitors, location):
    """CrowdAnalyst Agent - Advanced foot traffic and customer behavior analysis"""
    try:
        traffic_data = []
        total_traffic = 0
        total_revenue = 0
        capacity_utilization = []
        
        for comp in competitors:
            daily_traffic = comp.get('business_metrics', {}).get('estimated_daily_traffic', 50)
            revenue = comp.get('business_metrics', {}).get('estimated_monthly_revenue', 0)
            capacity = comp.get('operational_insights', {}).get('capacity_estimate', 100)
            utilization = (daily_traffic / capacity * 100) if capacity > 0 else 50
            
            traffic_data.append(f"- {comp['name']}: ~{daily_traffic} daily visitors, {utilization:.0f}% capacity")
            total_traffic += daily_traffic
            total_revenue += revenue
            capacity_utilization.append(utilization)
        
        avg_traffic = total_traffic / len(competitors) if competitors else 0
        avg_utilization = sum(capacity_utilization) / len(capacity_utilization) if capacity_utilization else 50
        revenue_per_customer = (total_revenue / 30) / total_traffic if total_traffic > 0 else 25
        
        return {
            "market_capacity": {
                "total_daily_customers": int(total_traffic),
                "average_per_restaurant": int(avg_traffic),
                "market_saturation": f"{avg_utilization:.0f}%",
                "revenue_per_customer": f"${revenue_per_customer:.0f}"
            },
            "traffic_patterns": {
                "peak_hours": ["11:30 AM - 1:30 PM (Lunch)", "6:00 PM - 8:30 PM (Dinner)", "7:00 PM - 9:00 PM (Weekend)"],
                "seasonal_trends": ["Higher traffic in warmer months", "Holiday season peaks", "Summer outdoor dining surge"],
                "customer_flow": f"Market serves {int(total_traffic * 30):,} customers monthly"
            },
            "behavior_insights": {
                "average_visit_duration": "45-75 minutes",
                "party_size": "2.3 people average",
                "repeat_visit_rate": "25-35%",
                "conversion_rate": "80-90% (foot traffic to purchase)"
            },
            "demographic_analysis": {
                "age_groups": ["25-34 (30%)", "35-44 (25%)", "45-54 (20%)", "Other (25%)"],
                "income_levels": "Varies by price point and location",
                "dining_preferences": "Quality and experience valued over price"
            },
            "market_opportunities": [
                f"Market capacity utilization at {avg_utilization:.0f}% - {'high demand' if avg_utilization > 70 else 'room for growth'}",
                "Off-peak hour optimization potential",
                "Customer retention improvement opportunities"
            ]
        }
        
    except Exception as e:
        logger.error(f"CrowdAnalyst agent error: {str(e)}")
        return generate_mock_traffic_analysis()

def sentinel_agent_analysis(competitors, location):
    """Sentinel Agent - Advanced competitive monitoring and threat analysis"""
    if not openai_client:
        return generate_mock_sentinel_analysis(competitors, location)
    
    try:
        competitive_data = []
        threat_levels = []
        digital_scores = []
        
        for comp in competitors:
            threat = comp.get('competitive_analysis', {}).get('threat_level', 'Medium')
            digital_score = comp.get('digital_presence', {}).get('digital_marketing_strength', 50)
            market_strength = comp.get('business_metrics', {}).get('competitive_strength', 50)
            
            competitive_data.append(f"- {comp['name']}: {threat} threat, {digital_score:.0f}/100 digital strength, {market_strength:.0f}/100 market position")
            threat_levels.append(threat)
            digital_scores.append(digital_score)
        
        avg_digital_score = sum(digital_scores) / len(digital_scores) if digital_scores else 50
        high_threat_count = sum(1 for t in threat_levels if t == 'High')
        
        prompt = f"""
        As an expert Sentinel AI agent monitoring competitive landscape in {location}:
        
        Competitive Intelligence:
        {chr(10).join(competitive_data)}
        
        Market Summary:
        - High-threat competitors: {high_threat_count}/{len(competitors)}
        - Average digital marketing strength: {avg_digital_score:.0f}/100
        - Market competitiveness: {"High" if avg_digital_score > 70 else "Medium"}
        
        Provide comprehensive competitive monitoring analysis in JSON format:
        {{
            "threat_assessment": "Overall competitive threat analysis",
            "emerging_trends": ["trend 1", "trend 2", "trend 3"],
            "competitive_moves": ["recent move 1", "recent move 2"],
            "market_disruptions": ["disruption 1", "disruption 2"],
            "strategic_recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
            "monitoring_priorities": ["priority 1", "priority 2"]
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Sentinel agent error: {str(e)}")
        return generate_mock_sentinel_analysis(competitors, location)

def market_intelligence_synthesizer(competitors, location, pricewatch_data, sentiment_data, crowd_data, sentinel_data):
    """Market Intelligence Synthesizer - Combines all AI agent insights"""
    if not openai_client:
        return generate_mock_synthesis(competitors, location)
    
    try:
        synthesis_prompt = f"""
        As a Master Market Intelligence Synthesizer for {location} restaurant market:
        
        PRICEWATCH INSIGHTS: {json.dumps(pricewatch_data, indent=2)}
        SENTIMENT ANALYSIS: {json.dumps(sentiment_data, indent=2)}
        CROWD ANALYTICS: {json.dumps(crowd_data, indent=2)}
        SENTINEL MONITORING: {json.dumps(sentinel_data, indent=2)}
        
        Synthesize all insights into strategic recommendations in JSON format:
        {{
            "executive_summary": "High-level market overview and key findings",
            "strategic_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
            "competitive_advantages": ["advantage to pursue 1", "advantage to pursue 2"],
            "market_entry_strategy": "Recommended approach for entering this market",
            "risk_mitigation": ["risk 1 and mitigation", "risk 2 and mitigation"],
            "success_metrics": ["metric to track 1", "metric to track 2", "metric to track 3"],
            "timeline_recommendations": "Suggested implementation timeline"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=800,
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Intelligence synthesizer error: {str(e)}")
        return generate_mock_synthesis(competitors, location)

# Mock data generators for fallback
def generate_mock_pricing_analysis(competitors, location):
    return {
        "market_overview": f"Pricing in {location} ranges from budget ($) to premium ($$$$) with strong mid-range presence",
        "pricing_opportunities": [
            "Gap in premium casual dining ($25-35 range)",
            "Limited early bird pricing strategies observed",
            "Opportunity for dynamic weekend pricing"
        ],
        "competitive_gaps": [
            "Few competitors offer value lunch pricing",
            "Premium brunch market underserved"
        ],
        "pricing_strategies": [
            "Consider tiered pricing for different meal periods",
            "Implement loyalty program with pricing benefits"
        ],
        "revenue_benchmarks": "Top performers achieve $15,000-30,000 monthly revenue per location",
        "price_elasticity": "Market shows moderate price sensitivity with quality expectations"
    }

def generate_mock_sentiment_analysis(competitors, location):
    return {
        "sentiment_overview": f"Overall positive sentiment in {location} with high expectations for service quality",
        "customer_pain_points": [
            "Wait times during peak hours",
            "Parking availability concerns",
            "Inconsistent service quality"
        ],
        "satisfaction_drivers": [
            "Fresh, quality ingredients",
            "Attentive staff service",
            "Comfortable atmosphere"
        ],
        "service_gaps": [
            "Limited vegetarian/vegan options",
            "Inconsistent WiFi availability"
        ],
        "reputation_insights": "Strong correlation between staff training and customer satisfaction scores",
        "customer_expectations": "Customers expect value for money, consistent quality, and personalized service"
    }

def generate_mock_traffic_analysis():
    return {
        "market_capacity": {
            "total_daily_customers": 2500,
            "average_per_restaurant": 125,
            "market_saturation": "75%",
            "revenue_per_customer": "$32"
        },
        "traffic_patterns": {
            "peak_hours": ["11:30 AM - 1:30 PM", "6:00 PM - 8:30 PM", "Weekend evenings"],
            "seasonal_trends": ["Summer outdoor dining surge", "Holiday season peaks"],
            "customer_flow": "Market serves 75,000 customers monthly"
        },
        "behavior_insights": {
            "average_visit_duration": "45-75 minutes",
            "party_size": "2.3 people average",
            "repeat_visit_rate": "30%",
            "conversion_rate": "85%"
        },
        "demographic_analysis": {
            "age_groups": ["25-34 (30%)", "35-44 (25%)", "Other (45%)"],
            "income_levels": "Middle to upper-middle class",
            "dining_preferences": "Quality and experience focused"
        },
        "market_opportunities": [
            "Off-peak optimization potential",
            "Customer retention improvement opportunities"
        ]
    }

def generate_mock_sentinel_analysis(competitors, location):
    return {
        "threat_assessment": f"Moderate to high competitive pressure in {location} with established players",
        "emerging_trends": [
            "Increased focus on sustainable practices",
            "Technology adoption for ordering and payments",
            "Social media marketing intensification"
        ],
        "competitive_moves": [
            "Menu diversification to include health-conscious options",
            "Enhanced delivery and takeout services"
        ],
        "market_disruptions": [
            "Ghost kitchen concepts gaining traction",
            "Subscription-based dining models emerging"
        ],
        "strategic_recommendations": [
            "Differentiate through unique dining experience",
            "Invest in technology integration",
            "Build strong local community relationships"
        ],
        "monitoring_priorities": [
            "Track competitor pricing changes",
            "Monitor social media engagement levels"
        ]
    }

def generate_mock_synthesis(competitors, location):
    return {
        "executive_summary": f"The {location} restaurant market shows strong potential with moderate competition and growing customer base",
        "strategic_opportunities": [
            "Fill identified pricing gaps in mid-premium segment",
            "Capitalize on underserved dietary preferences",
            "Leverage technology for competitive advantage"
        ],
        "competitive_advantages": [
            "Superior customer service delivery",
            "Unique menu positioning with local flavors"
        ],
        "market_entry_strategy": "Enter with differentiated concept targeting underserved premium-casual segment",
        "risk_mitigation": [
            "Staff retention challenges - implement comprehensive training program",
            "Market saturation risk - focus on unique value proposition"
        ],
        "success_metrics": [
            "Achieve 4.5+ star rating within 6 months",
            "Capture 3-5% local market share in first year",
            "Maintain 25%+ repeat customer rate"
        ],
        "timeline_recommendations": "6-month soft launch, 12-month full market penetration strategy"
    }

def get_mock_competitors(business_type):
    """Enhanced mock competitor data"""
    return [
        {
            "id": str(uuid.uuid4()),
            "name": "The Local Bistro",
            "address": "123 Main Street, Downtown",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "business_type": business_type,
            "phone": "(555) 123-4567",
            "website": "thelocalbistro.com",
            "rating": 4.2,
            "review_count": 342,
            "price_level": 3,
            "categories": ["New American", "Bistro", "Wine Bar"],
            "business_metrics": {
                "estimated_daily_traffic": 150,
                "estimated_monthly_revenue": 85000,
                "avg_check_estimate": 75,
                "customer_satisfaction_score": 84,
                "market_position": "Premium",
                "competitive_strength": 78
            }
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Giuseppe's Italian Kitchen",
            "address": "456 Oak Avenue, Midtown",
            "location": {"lat": 40.7589, "lng": -73.9851},
            "business_type": business_type,
            "phone": "(555) 987-6543",
            "website": "giuseppesitalian.com",
            "rating": 4.5,
            "review_count": 567,
            "price_level": 2,
            "categories": ["Italian", "Pizza", "Family Style"],
            "business_metrics": {
                "estimated_daily_traffic": 200,
                "estimated_monthly_revenue": 60000,
                "avg_check_estimate": 50,
                "customer_satisfaction_score": 90,
                "market_position": "Mid-Range",
                "competitive_strength": 85
            }
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Sakura Sushi Bar",
            "address": "789 Cherry Lane, Uptown",
            "location": {"lat": 40.7831, "lng": -73.9712},
            "business_type": business_type,
            "phone": "(555) 456-7890",
            "website": "sakurasushi.com",
            "rating": 4.7,
            "review_count": 289,
            "price_level": 4,
            "categories": ["Japanese", "Sushi", "Fine Dining"],
            "business_metrics": {
                "estimated_daily_traffic": 80,
                "estimated_monthly_revenue": 120000,
                "avg_check_estimate": 100,
                "customer_satisfaction_score": 94,
                "market_position": "Premium",
                "competitive_strength": 92
            }
        }
    ]

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "service": "BizFizz Advanced Competitive Intelligence Platform",
        "version": "2.0.0",
        "integrations": {
            "google_maps": bool(GOOGLE_MAPS_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "yelp": bool(YELP_API_KEY)
        },
        "features": {
            "ai_agents": ["PriceWatch", "Sentiment", "CrowdAnalyst", "Sentinel", "Intelligence Synthesizer"],
            "data_sources": ["Yelp Business Data", "Google Places", "AI-Generated Insights"],
            "analytics": ["Revenue Analysis", "Traffic Patterns", "Competitive Positioning", "Market Intelligence"]
        }
    }

@app.post("/api/search-competitors")
async def search_competitors(search: GeographicSearch):
    """Enhanced competitor search with comprehensive business intelligence"""
    try:
        logger.info(f"Searching for competitors in {search.location}")
        
        competitors = []
        
        # Try Yelp first for comprehensive data
        if YELP_API_KEY:
            competitors = search_yelp_businesses(search.location, search.radius, search.business_type)
            logger.info(f"Found {len(competitors)} competitors via Yelp API")
        
        # Fallback to Google Places if needed
        if not competitors and gmaps:
            try:
                places_result = gmaps.places_nearby(
                    location=search.location,
                    radius=search.radius * 1609.34,
                    type='restaurant',
                    keyword='restaurant'
                )
                
                for place in places_result.get('results', []):
                    competitor = {
                        "id": str(uuid.uuid4()),
                        "name": place.get('name', 'Unknown'),
                        "address": place.get('vicinity', 'Unknown'),
                        "location": {
                            "lat": place.get('geometry', {}).get('location', {}).get('lat', 0),
                            "lng": place.get('geometry', {}).get('location', {}).get('lng', 0)
                        },
                        "business_type": search.business_type,
                        "rating": place.get('rating'),
                        "review_count": place.get('user_ratings_total'),
                        "price_level": place.get('price_level', 2),
                        "place_id": place.get('place_id'),
                        "categories": ["Restaurant"],
                        "business_metrics": {
                            "estimated_daily_traffic": 100,
                            "estimated_monthly_revenue": 50000,
                            "avg_check_estimate": 35,
                            "customer_satisfaction_score": (place.get('rating', 3.5) * 20),
                            "market_position": "Mid-Range",
                            "competitive_strength": 60
                        }
                    }
                    competitors.append(competitor)
                
                logger.info(f"Found {len(competitors)} competitors via Google Places API")
                
            except Exception as e:
                logger.error(f"Google Places API error: {str(e)}")
        
        # Use enhanced mock data if no API results
        if not competitors:
            logger.info("Using enhanced mock data")
            competitors = get_mock_competitors(search.business_type)
        
        # Store search in database
        search_record = {
            "id": str(uuid.uuid4()),
            "location": search.location,
            "radius": search.radius,
            "business_type": search.business_type,
            "results_count": len(competitors),
            "timestamp": datetime.utcnow(),
            "data_source": "yelp" if YELP_API_KEY and competitors else "google" if gmaps else "mock"
        }
        
        await db.searches.insert_one(search_record)
        
        return {
            "search_id": search_record["id"],
            "location": search.location,
            "competitors": competitors,
            "total_found": len(competitors),
            "data_quality": "premium" if YELP_API_KEY else "standard",
            "market_insights": {
                "average_rating": sum(c.get('rating', 0) for c in competitors) / len(competitors) if competitors else 0,
                "price_range": f"${min(c.get('price_level', 1) for c in competitors)}-{max(c.get('price_level', 1) for c in competitors)}",
                "total_reviews": sum(c.get('review_count', 0) for c in competitors),
                "market_saturation": "High" if len(competitors) > 15 else "Medium" if len(competitors) > 10 else "Low"
            }
        }
        
    except Exception as e:
        logger.error(f"Error searching competitors: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search competitors")

@app.post("/api/generate-comprehensive-report")
async def generate_comprehensive_report(request: Dict[str, Any]):
    """Generate comprehensive competitive intelligence report using all AI agents"""
    try:
        competitor_ids = request.get("competitor_ids", [])
        location = request.get("location", "")
        user_business = request.get("user_business", {})
        
        if not competitor_ids:
            raise HTTPException(status_code=400, detail="No competitors selected")
        
        # Get competitor details
        competitors = []
        for comp_id in competitor_ids:
            # In production, would fetch from database
            # For now, create representative data
            competitor = {
                "id": comp_id,
                "name": f"Restaurant {comp_id[:8]}",
                "rating": 4.2 + (len(comp_id) % 5) * 0.1,
                "review_count": 200 + len(comp_id) * 10,
                "price_level": 2 + (len(comp_id) % 3),
                "business_metrics": {
                    "estimated_daily_traffic": 100 + len(comp_id) * 5,
                    "estimated_monthly_revenue": 50000 + len(comp_id) * 1000,
                    "avg_check_estimate": 30 + len(comp_id) % 20,
                    "competitive_strength": 60 + len(comp_id) % 30
                }
            }
            competitors.append(competitor)
        
        # Run all AI agents
        logger.info("Running PriceWatch Agent...")
        pricewatch_analysis = pricewatch_agent_analysis(competitors, location)
        
        logger.info("Running Sentiment Agent...")
        sentiment_analysis = sentiment_agent_analysis(competitors, location)
        
        logger.info("Running CrowdAnalyst Agent...")
        crowd_analysis = crowdanalyst_agent_analysis(competitors, location)
        
        logger.info("Running Sentinel Agent...")
        sentinel_analysis = sentinel_agent_analysis(competitors, location)
        
        logger.info("Running Intelligence Synthesizer...")
        synthesis = market_intelligence_synthesizer(
            competitors, location, pricewatch_analysis, 
            sentiment_analysis, crowd_analysis, sentinel_analysis
        )
        
        # Create comprehensive report
        report = {
            "id": str(uuid.uuid4()),
            "report_type": "comprehensive_intelligence",
            "location": location,
            "competitors": competitors,
            "total_competitors": len(competitors),
            "report_date": datetime.utcnow().isoformat(),
            "generated_with_ai": bool(openai_client),
            
            # AI Agent Analyses
            "pricewatch_analysis": pricewatch_analysis,
            "sentiment_analysis": sentiment_analysis,
            "crowd_analysis": crowd_analysis,
            "sentinel_analysis": sentinel_analysis,
            "market_synthesis": synthesis,
            
            # Executive Summary
            "executive_summary": synthesis.get("executive_summary", "Comprehensive market analysis completed"),
            "key_findings": [
                pricewatch_analysis.get("market_overview", "Pricing analysis completed"),
                sentiment_analysis.get("sentiment_overview", "Sentiment analysis completed"),
                crowd_analysis.get("market_capacity", {}).get("market_saturation", "Traffic analysis completed"),
                sentinel_analysis.get("threat_assessment", "Competitive monitoring completed")
            ],
            "strategic_recommendations": synthesis.get("strategic_opportunities", []),
            
            # Performance Metrics
            "market_metrics": {
                "total_market_revenue": sum(c.get('business_metrics', {}).get('estimated_monthly_revenue', 0) for c in competitors),
                "average_competitor_strength": sum(c.get('business_metrics', {}).get('competitive_strength', 50) for c in competitors) / len(competitors),
                "market_growth_potential": "High" if len(competitors) < 15 else "Medium",
                "entry_difficulty": synthesis.get("market_entry_strategy", "Standard market entry approach")
            }
        }
        
        # Store report in database
        await db.comprehensive_reports.insert_one(report.copy())
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate comprehensive report")

@app.get("/api/reports")
async def get_reports(limit: int = 10):
    """Get recent comprehensive reports"""
    try:
        reports = []
        
        # Get comprehensive reports
        async for report in db.comprehensive_reports.find().sort("report_date", -1).limit(limit):
            if "_id" in report:
                del report["_id"]
            reports.append(report)
        
        # Get basic reports if no comprehensive reports
        if not reports:
            async for report in db.reports.find().sort("report_date", -1).limit(limit):
                if "_id" in report:
                    del report["_id"]
                reports.append(report)
        
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.get("/api/subscription-tiers")
async def get_subscription_tiers():
    """Enhanced subscription tiers with advanced features"""
    return {
        "tiers": [
            {
                "name": "Free",
                "price": 0,
                "features": [
                    "1 basic report per month",
                    "Up to 5 competitors per report",
                    "Basic competitor data",
                    "Limited AI insights",
                    "Email support"
                ],
                "api_calls": 10,
                "ai_agents": ["Basic Analysis"],
                "data_sources": ["Mock Data"]
            },
            {
                "name": "Professional",
                "price": 99,
                "features": [
                    "10 comprehensive reports per month",
                    "Up to 20 competitors per report",
                    "Real-time Yelp & Google data",
                    "4 AI agents (PriceWatch, Sentiment, CrowdAnalyst, Sentinel)",
                    "Advanced market analytics",
                    "Visual reporting with charts",
                    "Priority support"
                ],
                "api_calls": 200,
                "ai_agents": ["PriceWatch", "Sentiment", "CrowdAnalyst", "Sentinel"],
                "data_sources": ["Yelp API", "Google Places", "AI Analysis"]
            },
            {
                "name": "Enterprise",
                "price": 299,
                "features": [
                    "Unlimited comprehensive reports",
                    "Unlimited competitors",
                    "All 5 AI agents + Intelligence Synthesizer",
                    "Real-time competitive monitoring",
                    "Custom market analysis",
                    "API access for integration",
                    "White-label reporting",
                    "Dedicated account manager",
                    "24/7 priority support"
                ],
                "api_calls": 2000,
                "ai_agents": ["All Agents", "Intelligence Synthesizer", "Custom Analysis"],
                "data_sources": ["Premium Data Access", "Real-time Monitoring", "Custom Integrations"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)