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

# Mock data for development (will be replaced with real APIs)
MOCK_COMPETITORS = [
    {
        "id": str(uuid.uuid4()),
        "name": "The Local Bistro",
        "address": "123 Main Street, Downtown",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "business_type": "restaurant",
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
        "business_type": "restaurant",
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
        "business_type": "restaurant",
        "phone": "(555) 456-7890",
        "website": "sakurasushi.com",
        "rating": 4.7,
        "review_count": 289,
        "price_level": 4
    }
]

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BizFizz API"}

@app.post("/api/search-competitors")
async def search_competitors(search: GeographicSearch):
    """Search for competitors in a geographic area"""
    try:
        # In production, this would use Google Places API
        # For now, return mock data
        
        # Simulate API delay
        await asyncio.sleep(1)
        
        # Filter mock competitors by business type
        filtered_competitors = [
            comp for comp in MOCK_COMPETITORS 
            if comp["business_type"] == search.business_type
        ]
        
        # Store search in database
        search_record = {
            "id": str(uuid.uuid4()),
            "location": search.location,
            "radius": search.radius,
            "business_type": search.business_type,
            "results_count": len(filtered_competitors),
            "timestamp": datetime.utcnow()
        }
        
        await db.searches.insert_one(search_record)
        
        return {
            "search_id": search_record["id"],
            "location": search.location,
            "competitors": filtered_competitors,
            "total_found": len(filtered_competitors)
        }
        
    except Exception as e:
        logger.error(f"Error searching competitors: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search competitors")

@app.post("/api/analyze-reviews")
async def analyze_reviews(competitor_ids: List[str]):
    """Analyze reviews for selected competitors"""
    try:
        # Simulate API delay
        await asyncio.sleep(2)
        
        # Mock review analysis data
        analyses = []
        for comp_id in competitor_ids:
            analysis = {
                "competitor_id": comp_id,
                "total_reviews": 150 + len(comp_id) * 10,  # Mock calculation
                "average_rating": 4.1 + (len(comp_id) % 3) * 0.2,
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
                "analysis_date": datetime.utcnow()
            }
            analyses.append(analysis)
        
        # Store analyses in database
        for analysis in analyses:
            await db.review_analyses.insert_one(analysis)
        
        return {"analyses": analyses}
        
    except Exception as e:
        logger.error(f"Error analyzing reviews: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze reviews")

@app.post("/api/generate-report")
async def generate_report(competitor_ids: List[str], location: str):
    """Generate comprehensive competitor report"""
    try:
        # Simulate report generation
        await asyncio.sleep(1)
        
        # Get competitor details
        competitors = [comp for comp in MOCK_COMPETITORS if comp["id"] in competitor_ids]
        
        # Generate insights and recommendations
        insights = [
            f"Average competitor rating in {location}: 4.5/5",
            f"Price range varies from $$ to $$$$",
            f"Most competitors have 200+ reviews",
            f"Italian and Asian cuisines dominate the area",
            f"Outdoor seating is becoming a key differentiator"
        ]
        
        recommendations = [
            "Consider expanding outdoor seating options",
            "Focus on consistent service quality to match 4.5+ rating",
            "Implement online ordering to compete with tech-savvy competitors",
            "Develop signature dishes to stand out from competition",
            "Monitor competitor pricing for strategic positioning"
        ]
        
        report = {
            "id": str(uuid.uuid4()),
            "search_location": location,
            "competitors": competitors,
            "insights": insights,
            "recommendations": recommendations,
            "report_date": datetime.utcnow(),
            "total_competitors": len(competitors)
        }
        
        # Store report in database
        await db.reports.insert_one(report)
        
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
            # Convert ObjectId to string if present
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
                    "Advanced sentiment analysis",
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
                    "AI-powered insights",
                    "Real-time monitoring",
                    "Custom integrations",
                    "24/7 support"
                ],
                "api_calls": 1000
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)