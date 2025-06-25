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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="BizFizz API", version="1.0.0")

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
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

class ReviewAnalysis(BaseModel):
    competitor_id: str
    total_reviews: int
    average_rating: float
    sentiment_breakdown: Dict[str, int]  # positive, negative, neutral
    key_themes: List[str]
    recent_trends: List[str]
    analysis_date: datetime = Field(default_factory=datetime.utcnow)

class CompetitorReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    search_location: str
    competitors: List[CompetitorProfile]
    review_analysis: List[ReviewAnalysis]
    insights: List[str]
    recommendations: List[str]
    report_date: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    business_name: Optional[str] = None
    subscription_tier: str = Field(default="free")  # free, basic, premium
    api_usage: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Helper functions
def get_mock_competitors(business_type):
    """Get mock competitor data"""
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
            "price_level": 3
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
            "price_level": 2
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
            "price_level": 4
        }
    ]

def search_yelp_businesses(location, radius, business_type="restaurant"):
    """Search Yelp for businesses"""
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
            'limit': 20
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
                    "categories": [cat['title'] for cat in business.get('categories', [])]
                }
                competitors.append(competitor)
            
            return competitors
            
    except Exception as e:
        logger.error(f"Yelp API error: {str(e)}")
        return []

def get_ai_insights(competitors, location):
    """Generate AI-powered insights from competitor data"""
    if not openai_client:
        return get_mock_insights(location)
    
    try:
        # Prepare competitor data for AI analysis
        competitor_summary = []
        for comp in competitors:
            comp_info = f"- {comp['name']}: {comp.get('rating', 'N/A')} stars, {comp.get('review_count', 'N/A')} reviews, Price level: {'$' * comp.get('price_level', 1)}"
            competitor_summary.append(comp_info)
        
        competitor_text = "\n".join(competitor_summary)
        
        prompt = f"""
        As a business intelligence analyst, analyze these restaurant competitors in {location}:
        
        {competitor_text}
        
        Provide 5 key insights about the competitive landscape that would help a restaurant owner make strategic decisions. Focus on:
        - Market positioning opportunities
        - Pricing strategies
        - Service quality benchmarks
        - Customer satisfaction trends
        - Competitive advantages to pursue
        
        Return only a JSON array of strings, no other text.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        insights_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            insights = json.loads(insights_text)
            if isinstance(insights, list):
                return insights[:5]
        except:
            # Fallback: split by lines and clean up
            lines = insights_text.split('\n')
            insights = []
            for line in lines:
                clean_line = line.strip('- ').strip().strip('"').strip("'")
                if clean_line and not clean_line.startswith('[') and not clean_line.startswith(']'):
                    insights.append(clean_line)
            return insights[:5] if insights else get_mock_insights(location)
            
    except Exception as e:
        logger.error(f"AI insights generation error: {str(e)}")
        return get_mock_insights(location)

def get_ai_recommendations(competitors, location):
    """Generate AI-powered recommendations"""
    if not openai_client:
        return get_mock_recommendations()
    
    try:
        competitor_summary = []
        for comp in competitors:
            comp_info = f"- {comp['name']}: {comp.get('rating', 'N/A')} stars, {comp.get('review_count', 'N/A')} reviews, Price level: {'$' * comp.get('price_level', 1)}"
            competitor_summary.append(comp_info)
        
        competitor_text = "\n".join(competitor_summary)
        
        prompt = f"""
        Based on this competitive analysis of restaurants in {location}:
        
        {competitor_text}
        
        Provide 5 specific, actionable recommendations for a new restaurant owner to succeed in this market. Each recommendation should be:
        - Specific and actionable
        - Based on competitive gaps or opportunities
        - Focused on differentiation and success
        
        Return only a JSON array of strings, no other text.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        recommendations_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            recommendations = json.loads(recommendations_text)
            if isinstance(recommendations, list):
                return recommendations[:5]
        except:
            # Fallback: split by lines and clean up
            lines = recommendations_text.split('\n')
            recommendations = []
            for line in lines:
                clean_line = line.strip('- ').strip().strip('"').strip("'")
                if clean_line and not clean_line.startswith('[') and not clean_line.startswith(']'):
                    recommendations.append(clean_line)
            return recommendations[:5] if recommendations else get_mock_recommendations()
            
    except Exception as e:
        logger.error(f"AI recommendations generation error: {str(e)}")
        return get_mock_recommendations()

def get_mock_insights(location):
    """Fallback mock insights"""
    return [
        f"Average competitor rating in {location}: 4.5/5 stars",
        f"Price range varies from $ to $$$$ across competitors",
        f"Most established competitors have 200+ customer reviews",
        f"Italian and Asian cuisines appear to dominate the market",
        f"Outdoor seating and delivery options are key differentiators"
    ]

def get_mock_recommendations():
    """Fallback mock recommendations"""
    return [
        "Consider expanding outdoor seating options to match competitor offerings",
        "Focus on achieving 4.5+ star rating through consistent service quality",
        "Implement robust online ordering system to compete with tech-savvy competitors",
        "Develop signature dishes or unique menu items to stand out from competition",
        "Monitor competitor pricing regularly and position strategically"
    ]

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "BizFizz API",
        "integrations": {
            "google_maps": bool(GOOGLE_MAPS_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "yelp": bool(YELP_API_KEY)
        }
    }

@app.post("/api/search-competitors")
async def search_competitors(search: GeographicSearch):
    """Search for competitors in a geographic area"""
    try:
        logger.info(f"Searching for competitors in {search.location}")
        
        competitors = []
        
        # Try Yelp first (more comprehensive restaurant data)
        if YELP_API_KEY:
            competitors = search_yelp_businesses(search.location, search.radius, search.business_type)
            logger.info(f"Found {len(competitors)} competitors via Yelp API")
        
        # If Yelp didn't return results, try Google Places
        if not competitors and gmaps:
            try:
                places_result = gmaps.places_nearby(
                    location=search.location,
                    radius=search.radius * 1609.34,  # Convert miles to meters
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
                        "price_level": place.get('price_level'),
                        "place_id": place.get('place_id')
                    }
                    competitors.append(competitor)
                
                logger.info(f"Found {len(competitors)} competitors via Google Places API")
                
            except Exception as e:
                logger.error(f"Google Places API error: {str(e)}")
        
        # Fallback to mock data if no API results
        if not competitors:
            logger.info("Using mock data - no API results or keys available")
            competitors = get_mock_competitors(search.business_type)
        
        # Store search in database
        search_record = {
            "id": str(uuid.uuid4()),
            "location": search.location,
            "radius": search.radius,
            "business_type": search.business_type,
            "results_count": len(competitors),
            "timestamp": datetime.utcnow()
        }
        
        await db.searches.insert_one(search_record)
        
        return {
            "search_id": search_record["id"],
            "location": search.location,
            "competitors": competitors,
            "total_found": len(competitors)
        }
        
    except Exception as e:
        logger.error(f"Error searching competitors: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search competitors")

@app.post("/api/analyze-reviews")
async def analyze_reviews(request: Dict[str, List[str]]):
    """Analyze reviews for selected competitors"""
    try:
        competitor_ids = request.get("competitor_ids", [])
        
        # Simulate API delay
        await asyncio.sleep(1)
        
        # Mock review analysis data (in production, would use real review data)
        analyses = []
        for comp_id in competitor_ids:
            analysis = {
                "competitor_id": comp_id,
                "total_reviews": 150 + len(comp_id) * 10,
                "average_rating": round(4.1 + (len(comp_id) % 3) * 0.2, 1),
                "sentiment_breakdown": {
                    "positive": 70 + (len(comp_id) % 4) * 5,
                    "neutral": 20,
                    "negative": 10 - (len(comp_id) % 4) * 2
                },
                "key_themes": ["food quality", "service speed", "atmosphere", "value for money"],
                "recent_trends": [
                    "Increased mentions of outdoor seating",
                    "Customers praising new menu items",
                    "Some concerns about wait times during peak hours"
                ],
                "analysis_date": datetime.utcnow().isoformat()
            }
            analyses.append(analysis)
        
        # Store analyses in database
        for analysis in analyses:
            await db.review_analyses.insert_one(analysis.copy())
        
        return {"analyses": analyses}
        
    except Exception as e:
        logger.error(f"Error analyzing reviews: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze reviews")

@app.post("/api/generate-report")
async def generate_report(request: Dict[str, Any]):
    """Generate comprehensive competitor report"""
    try:
        competitor_ids = request.get("competitor_ids", [])
        location = request.get("location", "")
        
        if not competitor_ids:
            raise HTTPException(status_code=400, detail="No competitors selected")
        
        # Simulate report generation delay
        await asyncio.sleep(1)
        
        # Get competitor details (in production, would fetch from database)
        competitors = []
        for comp_id in competitor_ids:
            competitor = {
                "id": comp_id,
                "name": f"Restaurant {comp_id[:8]}",
                "address": f"Address for {comp_id[:8]}",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "business_type": "restaurant",
                "rating": round(4.2 + (len(comp_id) % 5) * 0.1, 1),
                "review_count": 200 + len(comp_id) * 10,
                "price_level": 2 + (len(comp_id) % 3)
            }
            competitors.append(competitor)
        
        # Generate AI-powered insights and recommendations
        insights = get_ai_insights(competitors, location)
        recommendations = get_ai_recommendations(competitors, location)
        
        report = {
            "id": str(uuid.uuid4()),
            "search_location": location,
            "competitors": competitors,
            "insights": insights,
            "recommendations": recommendations,
            "report_date": datetime.utcnow().isoformat(),
            "total_competitors": len(competitors),
            "generated_with_ai": bool(openai_client)
        }
        
        # Store report in database
        await db.reports.insert_one(report.copy())
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.get("/api/reports")
async def get_reports(limit: int = 10):
    """Get recent reports"""
    try:
        reports = []
        async for report in db.reports.find().sort("report_date", -1).limit(limit):
            # Remove MongoDB ObjectId
            if "_id" in report:
                del report["_id"]
            reports.append(report)
        
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.get("/api/subscription-tiers")
async def get_subscription_tiers():
    """Get available subscription tiers"""
    return {
        "tiers": [
            {
                "name": "Free",
                "price": 0,
                "features": [
                    "1 report per month",
                    "Up to 5 competitors per report",
                    "Basic review analysis",
                    "Mock data insights",
                    "Email support"
                ],
                "api_calls": 10
            },
            {
                "name": "Basic",
                "price": 49,
                "features": [
                    "5 reports per month",
                    "Up to 15 competitors per report",
                    "Real-time competitor data",
                    "AI-powered sentiment analysis",
                    "Pricing trend alerts",
                    "Priority support"
                ],
                "api_calls": 100
            },
            {
                "name": "Premium",
                "price": 149,
                "features": [
                    "Unlimited reports",
                    "Unlimited competitors",
                    "Advanced AI insights",
                    "Real-time monitoring",
                    "Custom integrations",
                    "Yelp & Google data",
                    "24/7 support"
                ],
                "api_calls": 1000
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)