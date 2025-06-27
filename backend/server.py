from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, EmailStr
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
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionResponse, CheckoutStatusResponse, CheckoutSessionRequest
import tweepy
import asyncio
import httpx
from textblob import TextBlob
import feedparser
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BizFizz Ultimate Platform",
    version="4.0.0",
    description="Ultimate AI-Powered Business Intelligence & Consumer Marketplace Platform"
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
GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
YELP_API_KEY = os.environ.get('YELP_API_KEY')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_placeholder')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'pk_test_placeholder')

# Social Media API Keys
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY')
TWITTER_API_SECRET = os.environ.get('TWITTER_API_SECRET')
TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN')
TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

FACEBOOK_APP_ID = os.environ.get('FACEBOOK_APP_ID')
FACEBOOK_APP_SECRET = os.environ.get('FACEBOOK_APP_SECRET')

INSTAGRAM_ACCESS_TOKEN = os.environ.get('INSTAGRAM_ACCESS_TOKEN')

NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

# Notification API Keys
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        openai_client = None

# Initialize Stripe client
stripe_checkout = None
if STRIPE_SECRET_KEY and STRIPE_SECRET_KEY != 'sk_test_placeholder':
    try:
        stripe_checkout = StripeCheckout(api_key=STRIPE_SECRET_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Stripe client: {str(e)}")
        stripe_checkout = None

# Initialize Twitter client
twitter_client = None
if TWITTER_BEARER_TOKEN:
    try:
        twitter_client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize Twitter client: {str(e)}")
        twitter_client = None

# Log API key status
logger.info(f"API Keys loaded - Google Maps: {bool(GOOGLE_MAPS_API_KEY)}, Google Places: {bool(GOOGLE_PLACES_API_KEY)}, OpenAI: {bool(OPENAI_API_KEY)}, Yelp: {bool(YELP_API_KEY)}, Stripe: {bool(stripe_checkout)}, Twitter: {bool(twitter_client)}, Facebook: {bool(FACEBOOK_APP_ID)}, News: {bool(NEWS_API_KEY)}")

# WebSocket connection manager for real-time messaging
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_rooms: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast_to_room(self, message: str, room_id: str):
        if room_id in self.user_rooms:
            for user_id in self.user_rooms[room_id]:
                await self.send_personal_message(message, user_id)

manager = ConnectionManager()

# Pydantic models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    user_type: str = Field(..., description="business or consumer")
    business_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    user_type: str  # business or consumer
    business_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    subscription_tier: str = Field(default="starter")
    subscription_status: str = Field(default="active")
    credits: int = Field(default=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    profile_image: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

class BusinessProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    business_name: str
    business_type: str = Field(default="restaurant")
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    hours: Optional[Dict[str, str]] = None
    amenities: List[str] = Field(default_factory=list)
    photos: List[str] = Field(default_factory=list)
    menu_items: List[Dict[str, Any]] = Field(default_factory=list)
    is_verified: bool = Field(default=False)
    advertising_budget: float = Field(default=0.0)
    advertising_active: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConsumerReview(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    business_id: str
    rating: float = Field(..., ge=1, le=5)
    review_text: Optional[str] = None
    photos: List[str] = Field(default_factory=list)
    visit_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    helpful_votes: int = Field(default=0)
    verified_visit: bool = Field(default=False)

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    recipient_id: str
    message_type: str = Field(default="text")  # text, recommendation, business_inquiry
    content: str
    business_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    read_at: Optional[datetime] = None

class BusinessAdvertisement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str
    user_id: str
    ad_type: str = Field(..., description="banner, featured, sponsored_post")
    title: str
    description: str
    image_url: Optional[str] = None
    target_demographics: List[str] = Field(default_factory=list)
    budget_amount: float
    duration_days: int
    clicks: int = Field(default=0)
    impressions: int = Field(default=0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

class PaymentTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    transaction_type: str  # subscription, advertisement, credits
    amount: float
    currency: str = Field(default="usd")
    stripe_session_id: Optional[str] = None
    payment_status: str = Field(default="pending")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

# Subscription packages
class SocialMention(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: str  # twitter, facebook, instagram, google, news
    post_id: str
    content: str
    author_username: Optional[str] = None
    author_name: Optional[str] = None
    sentiment_score: float = Field(default=0.0)
    sentiment_label: str = Field(default="neutral")  # positive, negative, neutral
    business_id: Optional[str] = None
    business_name: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    published_at: datetime
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    is_alert_sent: bool = Field(default=False)
    engagement_metrics: Dict[str, int] = Field(default_factory=dict)  # likes, shares, etc.

class SocialMonitoringRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str
    business_name: str
    keywords: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)  # @username mentions
    hashtags: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)  # which platforms to monitor
    alert_settings: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_check: Optional[datetime] = None

class SocialAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str
    mention_id: str
    alert_type: str  # sentiment_negative, high_engagement, crisis, opportunity
    priority: str = Field(default="medium")  # low, medium, high, critical
    title: str
    description: str
    suggested_actions: List[str] = Field(default_factory=list)
    is_read: bool = Field(default=False)
    is_responded: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NewsArticle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    summary: Optional[str] = None
    source: str
    author: Optional[str] = None
    published_at: datetime
    url: str
    relevance_score: float = Field(default=0.0)
    keywords: List[str] = Field(default_factory=list)
    industry_tags: List[str] = Field(default_factory=list)
    business_impact: str = Field(default="neutral")  # positive, negative, neutral
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PaymentTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    transaction_type: str  # subscription, advertisement, credits
    amount: float
    currency: str = Field(default="usd")
    stripe_session_id: Optional[str] = None
    payment_status: str = Field(default="pending")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
SUBSCRIPTION_PACKAGES = {
    "starter": {"price": 0.0, "credits": 10, "features": ["Basic reports", "5 competitors", "Email support"]},
    "professional": {"price": 149.0, "credits": 500, "features": ["Unlimited reports", "25 competitors", "Advanced analytics", "Priority support"]},
    "enterprise": {"price": 399.0, "credits": 2000, "features": ["Enterprise features", "Unlimited competitors", "Custom integrations", "Dedicated support"]}
}

ADVERTISING_PACKAGES = {
    "basic": {"price": 29.0, "duration_days": 7, "features": ["Basic listing highlight", "1000 impressions"]},
    "featured": {"price": 99.0, "duration_days": 30, "features": ["Featured placement", "5000 impressions", "Analytics dashboard"]},
    "premium": {"price": 299.0, "duration_days": 30, "features": ["Premium placement", "Unlimited impressions", "Advanced targeting", "Dedicated support"]}
}

# Social Media Monitoring Functions
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Enhanced sentiment analysis using OpenAI and TextBlob"""
    try:
        # Basic TextBlob analysis
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # Enhanced OpenAI analysis if available
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Analyze sentiment of the following text. Return a JSON with sentiment_score (-1 to 1) and sentiment_label (positive/negative/neutral). Be precise and consider context."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                ai_analysis = json.loads(response.choices[0].message.content.strip())
                return {
                    "sentiment_score": ai_analysis.get("sentiment_score", textblob_score),
                    "sentiment_label": ai_analysis.get("sentiment_label", "neutral"),
                    "confidence": 0.9,
                    "method": "openai_enhanced"
                }
            except Exception as e:
                logger.error(f"OpenAI sentiment analysis failed: {e}")
        
        # Fallback to TextBlob
        sentiment_label = "positive" if textblob_score > 0.1 else "negative" if textblob_score < -0.1 else "neutral"
        return {
            "sentiment_score": textblob_score,
            "sentiment_label": sentiment_label,
            "confidence": 0.7,
            "method": "textblob"
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0, "method": "error"}

async def monitor_twitter(keywords: List[str], business_id: str, business_name: str):
    """Monitor Twitter for mentions and keywords"""
    if not twitter_client:
        logger.warning("Twitter client not initialized")
        return []
    
    try:
        mentions = []
        
        # Search for each keyword
        for keyword in keywords:
            query = f'"{keyword}" OR "{business_name}" -is:retweet'
            
            tweets = twitter_client.search_recent_tweets(
                query=query,
                max_results=10,
                expansions=["author_id"],
                tweet_fields=["created_at", "public_metrics", "context_annotations"],
                user_fields=["username", "public_metrics"]
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    # Analyze sentiment
                    sentiment = analyze_sentiment(tweet.text)
                    
                    # Extract author info
                    author_username = "unknown"
                    if tweets.includes and tweets.includes.get("users"):
                        for user in tweets.includes["users"]:
                            if user.id == tweet.author_id:
                                author_username = user.username
                                break
                    
                    mention = SocialMention(
                        platform="twitter",
                        post_id=str(tweet.id),
                        content=tweet.text,
                        author_username=author_username,
                        sentiment_score=sentiment["sentiment_score"],
                        sentiment_label=sentiment["sentiment_label"],
                        business_id=business_id,
                        business_name=business_name,
                        keywords=[keyword],
                        url=f"https://twitter.com/{author_username}/status/{tweet.id}",
                        published_at=tweet.created_at,
                        engagement_metrics=tweet.public_metrics or {}
                    )
                    
                    mentions.append(mention)
        
        return mentions
        
    except Exception as e:
        logger.error(f"Twitter monitoring error: {e}")
        return []

async def monitor_facebook(keywords: List[str], business_id: str, business_name: str):
    """Monitor Facebook for business mentions"""
    if not FACEBOOK_APP_ID:
        logger.warning("Facebook credentials not configured")
        return []
    
    try:
        mentions = []
        
        # Facebook Graph API search (simplified)
        async with httpx.AsyncClient() as client:
            for keyword in keywords:
                # Note: This is a simplified implementation
                # Real implementation would need proper OAuth flow and permissions
                url = f"https://graph.facebook.com/v18.0/search"
                params = {
                    "q": keyword,
                    "type": "post",
                    "access_token": f"{FACEBOOK_APP_ID}|{FACEBOOK_APP_SECRET}"  # App access token
                }
                
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get("data", []):
                        sentiment = analyze_sentiment(post.get("message", ""))
                        
                        mention = SocialMention(
                            platform="facebook",
                            post_id=post.get("id", ""),
                            content=post.get("message", ""),
                            sentiment_score=sentiment["sentiment_score"],
                            sentiment_label=sentiment["sentiment_label"],
                            business_id=business_id,
                            business_name=business_name,
                            keywords=[keyword],
                            url=f"https://facebook.com/{post.get('id', '')}",
                            published_at=datetime.fromisoformat(post.get("created_time", "").replace("Z", "+00:00")) if post.get("created_time") else datetime.utcnow()
                        )
                        
                        mentions.append(mention)
        
        return mentions
        
    except Exception as e:
        logger.error(f"Facebook monitoring error: {e}")
        return []

async def monitor_google_reviews(keywords: List[str], business_id: str, business_name: str):
    """Monitor Google Places/Reviews for business mentions"""
    if not GOOGLE_PLACES_API_KEY:
        logger.warning("Google Places API key not configured")
        return []
    
    try:
        mentions = []
        
        async with httpx.AsyncClient() as client:
            # Search for the business first
            search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            search_params = {
                "query": f"{business_name} restaurant",
                "type": "restaurant",
                "key": GOOGLE_PLACES_API_KEY
            }
            
            search_response = await client.get(search_url, params=search_params)
            if search_response.status_code == 200:
                search_data = search_response.json()
                
                for place in search_data.get("results", []):
                    place_id = place.get("place_id")
                    
                    if place_id:
                        # Get place details including reviews
                        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                        details_params = {
                            "place_id": place_id,
                            "fields": "name,reviews,rating,user_ratings_total",
                            "key": GOOGLE_PLACES_API_KEY
                        }
                        
                        details_response = await client.get(details_url, params=details_params)
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            place_details = details_data.get("result", {})
                            
                            for review in place_details.get("reviews", []):
                                # Analyze sentiment
                                sentiment = analyze_sentiment(review.get("text", ""))
                                
                                mention = SocialMention(
                                    platform="google",
                                    post_id=f"google_review_{review.get('time', '')}_{place_id}",
                                    content=review.get("text", ""),
                                    author_username=review.get("author_name", "Google User"),
                                    sentiment_score=sentiment["sentiment_score"],
                                    sentiment_label=sentiment["sentiment_label"],
                                    business_id=business_id,
                                    business_name=business_name,
                                    keywords=keywords,
                                    url=f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                                    published_at=datetime.fromtimestamp(review.get("time", 0)) if review.get("time") else datetime.utcnow(),
                                    engagement_metrics={
                                        "rating": review.get("rating", 0),
                                        "helpful": 0  # Google doesn't provide this in API
                                    }
                                )
                                
                                mentions.append(mention)
        
        return mentions
        
    except Exception as e:
        logger.error(f"Google Places monitoring error: {e}")
        return []

async def monitor_news(keywords: List[str], business_id: str, business_name: str):
    """Monitor news sources for industry and business mentions"""
    if not NEWS_API_KEY:
        logger.warning("News API key not configured")
        return []
    
    try:
        articles = []
        
        async with httpx.AsyncClient() as client:
            for keyword in keywords:
                # Search general news
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f'"{keyword}" AND restaurant',
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "language": "en",
                    "apiKey": NEWS_API_KEY
                }
                
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get("articles", []):
                        # Analyze business impact
                        content = f"{article.get('title', '')} {article.get('description', '')}"
                        sentiment = analyze_sentiment(content)
                        
                        news_article = NewsArticle(
                            title=article.get("title", ""),
                            content=article.get("description", ""),
                            source=article.get("source", {}).get("name", ""),
                            author=article.get("author"),
                            published_at=datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "+00:00")) if article.get("publishedAt") else datetime.utcnow(),
                            url=article.get("url", ""),
                            relevance_score=0.8,  # Calculate based on keyword matches
                            keywords=[keyword],
                            industry_tags=["restaurant", "food"],
                            business_impact=sentiment["sentiment_label"]
                        )
                        
                        articles.append(news_article)
        
        return articles
        
    except Exception as e:
        logger.error(f"News monitoring error: {e}")
        return []

async def create_smart_alert(mention: SocialMention) -> Optional[SocialAlert]:
    """Create intelligent alerts based on mention characteristics"""
    try:
        alert_type = "neutral"
        priority = "low"
        suggested_actions = []
        
        # Determine alert type and priority
        if mention.sentiment_score < -0.5:
            alert_type = "sentiment_negative"
            priority = "high" if mention.sentiment_score < -0.8 else "medium"
            suggested_actions = [
                "Respond promptly to address concerns",
                "Investigate the issue mentioned",
                "Offer to resolve the problem privately",
                "Monitor for additional negative mentions"
            ]
        elif mention.sentiment_score > 0.5:
            alert_type = "opportunity"
            priority = "medium"
            suggested_actions = [
                "Thank the customer for positive feedback",
                "Share the positive review on your channels",
                "Engage with the customer to build loyalty",
                "Use feedback for marketing testimonials"
            ]
        
        # Check engagement metrics for high impact
        if mention.engagement_metrics:
            total_engagement = sum(mention.engagement_metrics.values())
            if total_engagement > 100:  # High engagement threshold
                alert_type = "high_engagement"
                priority = "high"
                suggested_actions.insert(0, "High visibility content - immediate attention required")
        
        # Crisis detection
        if mention.sentiment_score < -0.8 and any(word in mention.content.lower() for word in ["terrible", "worst", "awful", "disgusting", "never again"]):
            alert_type = "crisis"
            priority = "critical"
            suggested_actions = [
                "URGENT: Immediate response required",
                "Contact customer directly",
                "Escalate to management",
                "Prepare public response strategy"
            ]
        
        if priority in ["medium", "high", "critical"]:
            alert = SocialAlert(
                business_id=mention.business_id,
                mention_id=mention.id,
                alert_type=alert_type,
                priority=priority,
                title=f"{alert_type.replace('_', ' ').title()} Alert - {mention.platform.title()}",
                description=f"Detected {alert_type.replace('_', ' ')} mention on {mention.platform}: {mention.content[:100]}...",
                suggested_actions=suggested_actions,
                metadata={
                    "platform": mention.platform,
                    "sentiment_score": mention.sentiment_score,
                    "engagement": mention.engagement_metrics,
                    "url": mention.url
                }
            )
            
            return alert
        
        return None
        
    except Exception as e:
        logger.error(f"Alert creation error: {e}")
        return None

# Enhanced Yelp Business Search (from previous implementation)
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
            'limit': 50,
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
                    "image_url": business.get('image_url'),
                    "photos": [business.get('image_url')] if business.get('image_url') else []
                }
                
                # Enhanced business metrics
                competitor["business_metrics"] = {
                    "estimated_daily_traffic": min(max(int(competitor["review_count"] / 30), 10), 500),
                    "estimated_monthly_revenue": min(max(int(competitor["review_count"] / 30), 10), 500) * competitor["price_level"] * 25 * 30,
                    "avg_check_estimate": competitor["price_level"] * 25,
                    "customer_satisfaction_score": competitor["rating"] * 20,
                    "market_position": "Premium" if competitor["price_level"] >= 3 else "Mid-Range" if competitor["price_level"] == 2 else "Budget",
                    "competitive_strength": min(100, (competitor["rating"] * 15) + (min(competitor["review_count"]/100, 10) * 5))
                }
                
                competitors.append(competitor)
            
            return competitors
            
    except Exception as e:
        logger.error(f"Enhanced Yelp API error: {str(e)}")
        return []

# AI Agent functions (simplified versions for space)
def pricewatch_agent_analysis(competitors, location):
    """PriceWatch Agent - Pricing analysis"""
    try:
        if not openai_client:
            return {
                "pricing_insights": ["Market shows diverse pricing strategies", "Premium segment opportunities exist"],
                "recommendations": ["Consider competitive pricing", "Monitor price changes"]
            }
        
        price_data = [f"- {comp['name']}: ${comp.get('business_metrics', {}).get('avg_check_estimate', 25)}" for comp in competitors]
        
        prompt = f"Analyze restaurant pricing in {location}:\n{chr(10).join(price_data)}\n\nProvide JSON with pricing_insights and recommendations arrays."
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"PriceWatch error: {e}")
        return {"pricing_insights": ["Analysis unavailable"], "recommendations": ["Retry analysis"]}

def sentiment_agent_analysis(competitors, location):
    """Sentiment Agent - Customer sentiment analysis"""
    try:
        avg_rating = sum(comp.get('rating', 0) for comp in competitors) / len(competitors) if competitors else 0
        
        return {
            "sentiment_overview": f"Average market sentiment: {avg_rating:.1f}/5 stars",
            "customer_satisfaction": "High" if avg_rating >= 4.0 else "Medium" if avg_rating >= 3.5 else "Low",
            "improvement_areas": ["Service speed", "Food quality", "Value for money"],
            "competitive_advantages": ["Location", "Menu variety", "Customer service"]
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"sentiment_overview": "Analysis unavailable"}

# Payment Processing Functions
async def create_payment_transaction(user_id: str, transaction_type: str, amount: float, metadata: Dict = None):
    """Create a payment transaction record"""
    transaction = PaymentTransaction(
        user_id=user_id,
        transaction_type=transaction_type,
        amount=amount,
        metadata=metadata or {}
    )
    
    await db.payment_transactions.insert_one(transaction.dict())
    return transaction

async def update_payment_status(session_id: str, status: str, payment_status: str):
    """Update payment transaction status"""
    await db.payment_transactions.update_one(
        {"stripe_session_id": session_id},
        {
            "$set": {
                "payment_status": payment_status,
                "completed_at": datetime.utcnow() if payment_status == "paid" else None
            }
        }
    )

# WebSocket endpoint for real-time messaging
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Store message in database
            message = Message(
                sender_id=user_id,
                recipient_id=message_data.get("recipient_id"),
                content=message_data.get("content"),
                message_type=message_data.get("type", "text"),
                business_id=message_data.get("business_id")
            )
            
            await db.messages.insert_one(message.dict())
            
            # Send to recipient
            await manager.send_personal_message(
                json.dumps({
                    "type": "new_message",
                    "message": message.dict(),
                    "sender_id": user_id
                }),
                message_data.get("recipient_id")
            )
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)

# Social Media Monitoring API Endpoints

@app.post("/api/social/monitoring/start")
async def start_social_monitoring(rule: SocialMonitoringRule):
    """Start social media monitoring for a business"""
    try:
        # Store monitoring rule
        await db.social_monitoring_rules.insert_one(rule.dict())
        
        # Start background monitoring task
        asyncio.create_task(run_social_monitoring(rule))
        
        return {"message": "Social monitoring started successfully", "rule_id": rule.id}
        
    except Exception as e:
        logger.error(f"Start monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")

@app.post("/api/social/monitoring/stop/{rule_id}")
async def stop_social_monitoring(rule_id: str):
    """Stop social media monitoring for a specific rule"""
    try:
        await db.social_monitoring_rules.update_one(
            {"id": rule_id},
            {"$set": {"is_active": False}}
        )
        
        return {"message": "Social monitoring stopped successfully"}
        
    except Exception as e:
        logger.error(f"Stop monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")

@app.get("/api/social/mentions/{business_id}")
async def get_social_mentions(
    business_id: str,
    platform: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get social media mentions for a business"""
    try:
        query = {"business_id": business_id}
        
        if platform:
            query["platform"] = platform
        if sentiment:
            query["sentiment_label"] = sentiment
        
        mentions = []
        async for mention in db.social_mentions.find(query).sort("detected_at", -1).skip(offset).limit(limit):
            if "_id" in mention:
                del mention["_id"]
            mentions.append(mention)
        
        return {"mentions": mentions, "total": len(mentions)}
        
    except Exception as e:
        logger.error(f"Get mentions error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get mentions")

@app.get("/api/social/alerts/{business_id}")
async def get_social_alerts(
    business_id: str,
    priority: Optional[str] = None,
    is_read: Optional[bool] = None,
    limit: int = 20
):
    """Get social media alerts for a business"""
    try:
        query = {"business_id": business_id}
        
        if priority:
            query["priority"] = priority
        if is_read is not None:
            query["is_read"] = is_read
        
        alerts = []
        async for alert in db.social_alerts.find(query).sort("created_at", -1).limit(limit):
            if "_id" in alert:
                del alert["_id"]
            alerts.append(alert)
        
        return {"alerts": alerts, "total": len(alerts)}
        
    except Exception as e:
        logger.error(f"Get alerts error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

@app.put("/api/social/alerts/{alert_id}/mark-read")
async def mark_alert_read(alert_id: str):
    """Mark an alert as read"""
    try:
        await db.social_alerts.update_one(
            {"id": alert_id},
            {"$set": {"is_read": True}}
        )
        
        return {"message": "Alert marked as read"}
        
    except Exception as e:
        logger.error(f"Mark alert read error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark alert as read")

@app.get("/api/social/analytics/{business_id}")
async def get_social_analytics(
    business_id: str,
    days: int = 7,
    platform: Optional[str] = None
):
    """Get social media analytics for a business"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        match_query = {
            "business_id": business_id,
            "detected_at": {"$gte": start_date, "$lte": end_date}
        }
        
        if platform:
            match_query["platform"] = platform
        
        # Sentiment distribution
        sentiment_pipeline = [
            {"$match": match_query},
            {"$group": {
                "_id": "$sentiment_label",
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$sentiment_score"}
            }}
        ]
        
        sentiment_data = await db.social_mentions.aggregate(sentiment_pipeline).to_list(None)
        
        # Platform distribution
        platform_pipeline = [
            {"$match": match_query},
            {"$group": {
                "_id": "$platform",
                "count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"}
            }}
        ]
        
        platform_data = await db.social_mentions.aggregate(platform_pipeline).to_list(None)
        
        # Time series data
        time_pipeline = [
            {"$match": match_query},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$detected_at"}},
                    "sentiment": "$sentiment_label"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        
        time_data = await db.social_mentions.aggregate(time_pipeline).to_list(None)
        
        # Alert summary
        alert_pipeline = [
            {"$match": {
                "business_id": business_id,
                "created_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": "$priority",
                "count": {"$sum": 1}
            }}
        ]
        
        alert_data = await db.social_alerts.aggregate(alert_pipeline).to_list(None)
        
        return {
            "sentiment_distribution": sentiment_data,
            "platform_distribution": platform_data,
            "time_series": time_data,
            "alert_summary": alert_data,
            "date_range": {"start": start_date, "end": end_date}
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@app.get("/api/news/articles")
async def get_news_articles(
    keywords: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """Get relevant news articles"""
    try:
        query = {}
        
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(",")]
            query["keywords"] = {"$in": keyword_list}
        
        articles = []
        async for article in db.news_articles.find(query).sort("published_at", -1).skip(offset).limit(limit):
            if "_id" in article:
                del article["_id"]
            articles.append(article)
        
        return {"articles": articles, "total": len(articles)}
        
    except Exception as e:
        logger.error(f"Get news error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get news articles")

@app.websocket("/api/social/live/{business_id}")
async def social_websocket_endpoint(websocket: WebSocket, business_id: str):
    """WebSocket endpoint for real-time social media updates"""
    await manager.connect(websocket, business_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages if needed
            
    except WebSocketDisconnect:
        manager.disconnect(business_id)

# Background monitoring task
async def run_social_monitoring(rule: SocialMonitoringRule):
    """Background task to continuously monitor social media"""
    try:
        while rule.is_active:
            logger.info(f"Running social monitoring for business: {rule.business_name}")
            
            all_mentions = []
            
            # Monitor Twitter
            if "twitter" in rule.platforms:
                twitter_mentions = await monitor_twitter(rule.keywords, rule.business_id, rule.business_name)
                all_mentions.extend(twitter_mentions)
            
            # Monitor Facebook
            if "facebook" in rule.platforms:
                facebook_mentions = await monitor_facebook(rule.keywords, rule.business_id, rule.business_name)
                all_mentions.extend(facebook_mentions)
            
            # Monitor News
            if "news" in rule.platforms:
                news_articles = await monitor_news(rule.keywords, rule.business_id, rule.business_name)
                # Store news articles
                for article in news_articles:
                    await db.news_articles.insert_one(article.dict())
            
            # Process mentions
            for mention in all_mentions:
                # Check if mention already exists
                existing = await db.social_mentions.find_one({"post_id": mention.post_id, "platform": mention.platform})
                
                if not existing:
                    # Store new mention
                    await db.social_mentions.insert_one(mention.dict())
                    
                    # Create alert if needed
                    alert = await create_smart_alert(mention)
                    if alert:
                        await db.social_alerts.insert_one(alert.dict())
                        
                        # Send real-time notification
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "new_alert",
                                "alert": alert.dict(),
                                "mention": mention.dict()
                            }),
                            rule.business_id
                        )
            
            # Update last check time
            await db.social_monitoring_rules.update_one(
                {"id": rule.id},
                {"$set": {"last_check": datetime.utcnow()}}
            )
            
            # Wait before next check (5 minutes)
            await asyncio.sleep(300)
            
            # Refresh rule status
            updated_rule = await db.social_monitoring_rules.find_one({"id": rule.id})
            if not updated_rule or not updated_rule.get("is_active", False):
                break
                
    except Exception as e:
        logger.error(f"Social monitoring error: {str(e)}")

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Ultimate health check endpoint"""
    return {
        "status": "healthy",
        "service": "BizFizz Ultimate Platform",
        "version": "4.0.0",
        "features": {
            "business_intelligence": True,
            "consumer_marketplace": True,
            "payment_processing": bool(stripe_checkout),
            "real_time_messaging": True,
            "advertising_platform": True,
            "social_media_monitoring": True,
            "news_monitoring": bool(NEWS_API_KEY),
            "sentiment_analysis": True
        },
        "integrations": {
            "google_maps": bool(GOOGLE_MAPS_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "yelp": bool(YELP_API_KEY),
            "stripe": bool(stripe_checkout),
            "twitter": bool(twitter_client),
            "facebook": bool(FACEBOOK_APP_ID),
            "news_api": bool(NEWS_API_KEY),
            "twilio": bool(TWILIO_ACCOUNT_SID),
            "sendgrid": bool(SENDGRID_API_KEY)
        }
    }

# User Management
@app.post("/api/users/register")
async def register_user(user: UserRegistration):
    """Register a new user"""
    try:
        # Check if user exists
        existing_user = await db.users.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user profile
        user_profile = UserProfile(
            email=user.email,
            user_type=user.user_type,
            business_name=user.business_name,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        await db.users.insert_one(user_profile.dict())
        
        return {"message": "User registered successfully", "user_id": user_profile.id}
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.get("/api/users/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile"""
    try:
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if "_id" in user:
            del user["_id"]
        
        return user
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user")

# Business Management
@app.post("/api/businesses")
async def create_business_profile(business: BusinessProfile):
    """Create business profile"""
    try:
        await db.businesses.insert_one(business.dict())
        return {"message": "Business profile created", "business_id": business.id}
        
    except Exception as e:
        logger.error(f"Create business error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create business")

@app.get("/api/businesses")
async def get_businesses(
    location: Optional[str] = None,
    business_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """Get businesses for consumer marketplace"""
    try:
        query = {}
        if business_type:
            query["business_type"] = business_type
        
        businesses = []
        async for business in db.businesses.find(query).skip(offset).limit(limit):
            if "_id" in business:
                del business["_id"]
            
            # Add average rating from reviews
            pipeline = [
                {"$match": {"business_id": business["id"]}},
                {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}, "review_count": {"$sum": 1}}}
            ]
            
            rating_data = await db.consumer_reviews.aggregate(pipeline).to_list(1)
            if rating_data:
                business["avg_rating"] = round(rating_data[0]["avg_rating"], 1)
                business["review_count"] = rating_data[0]["review_count"]
            else:
                business["avg_rating"] = 0
                business["review_count"] = 0
            
            businesses.append(business)
        
        return {"businesses": businesses}
        
    except Exception as e:
        logger.error(f"Get businesses error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get businesses")

@app.get("/api/businesses/{business_id}")
async def get_business_details(business_id: str):
    """Get detailed business information"""
    try:
        business = await db.businesses.find_one({"id": business_id})
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        
        if "_id" in business:
            del business["_id"]
        
        # Get reviews
        reviews = []
        async for review in db.consumer_reviews.find({"business_id": business_id}).limit(10):
            if "_id" in review:
                del review["_id"]
            
            # Get reviewer info
            reviewer = await db.users.find_one({"id": review["user_id"]})
            if reviewer:
                review["reviewer_name"] = reviewer.get("first_name", "Anonymous")
            
            reviews.append(review)
        
        business["reviews"] = reviews
        
        return business
        
    except Exception as e:
        logger.error(f"Get business details error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get business details")

# Consumer Reviews
@app.post("/api/reviews")
async def create_review(review: ConsumerReview):
    """Create a consumer review"""
    try:
        await db.consumer_reviews.insert_one(review.dict())
        return {"message": "Review created successfully", "review_id": review.id}
        
    except Exception as e:
        logger.error(f"Create review error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create review")

@app.get("/api/reviews/business/{business_id}")
async def get_business_reviews(business_id: str, limit: int = 10, offset: int = 0):
    """Get reviews for a business"""
    try:
        reviews = []
        async for review in db.consumer_reviews.find({"business_id": business_id}).skip(offset).limit(limit):
            if "_id" in review:
                del review["_id"]
            
            # Get reviewer info
            reviewer = await db.users.find_one({"id": review["user_id"]})
            if reviewer:
                review["reviewer_name"] = reviewer.get("first_name", "Anonymous")
                review["reviewer_type"] = reviewer.get("user_type", "consumer")
            
            reviews.append(review)
        
        return {"reviews": reviews}
        
    except Exception as e:
        logger.error(f"Get reviews error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get reviews")

# Messaging System
@app.get("/api/messages/{user_id}")
async def get_user_messages(user_id: str, limit: int = 50):
    """Get messages for a user"""
    try:
        messages = []
        async for message in db.messages.find({
            "$or": [{"sender_id": user_id}, {"recipient_id": user_id}]
        }).sort("created_at", -1).limit(limit):
            if "_id" in message:
                del message["_id"]
            messages.append(message)
        
        return {"messages": messages}
        
    except Exception as e:
        logger.error(f"Get messages error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get messages")

@app.post("/api/messages")
async def send_message(message: Message):
    """Send a message"""
    try:
        await db.messages.insert_one(message.dict())
        
        # Send real-time notification
        await manager.send_personal_message(
            json.dumps({
                "type": "new_message",
                "message": message.dict()
            }),
            message.recipient_id
        )
        
        return {"message": "Message sent successfully", "message_id": message.id}
        
    except Exception as e:
        logger.error(f"Send message error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")

# Payment Processing
@app.get("/api/stripe/config")
async def get_stripe_config():
    """Get Stripe publishable key"""
    return {"publishableKey": STRIPE_PUBLISHABLE_KEY}

@app.post("/api/payments/create-checkout-session")
async def create_checkout_session(request: Dict[str, Any]):
    """Create Stripe checkout session"""
    try:
        if not stripe_checkout:
            raise HTTPException(status_code=503, detail="Payment processing not available")
        
        package_type = request.get("package_type")  # subscription, advertisement
        package_id = request.get("package_id")
        user_id = request.get("user_id")
        origin_url = request.get("origin_url")
        
        # Get package details
        if package_type == "subscription":
            if package_id not in SUBSCRIPTION_PACKAGES:
                raise HTTPException(status_code=400, detail="Invalid subscription package")
            package = SUBSCRIPTION_PACKAGES[package_id]
        elif package_type == "advertisement":
            if package_id not in ADVERTISING_PACKAGES:
                raise HTTPException(status_code=400, detail="Invalid advertising package")
            package = ADVERTISING_PACKAGES[package_id]
        else:
            raise HTTPException(status_code=400, detail="Invalid package type")
        
        amount = package["price"]
        
        if amount == 0:
            # Free package, update user directly
            await db.users.update_one(
                {"id": user_id},
                {"$set": {"subscription_tier": package_id, "credits": package.get("credits", 0)}}
            )
            return {"message": "Free package activated", "session_id": None}
        
        # Create checkout session
        success_url = f"{origin_url}/payment-success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{origin_url}/payment-cancel"
        
        checkout_request = CheckoutSessionRequest(
            amount=amount,
            currency="usd",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": user_id,
                "package_type": package_type,
                "package_id": package_id
            }
        )
        
        session = await stripe_checkout.create_checkout_session(checkout_request)
        
        # Create transaction record
        transaction = await create_payment_transaction(
            user_id=user_id,
            transaction_type=package_type,
            amount=amount,
            metadata={
                "package_id": package_id,
                "stripe_session_id": session.session_id
            }
        )
        
        # Update transaction with session ID
        await db.payment_transactions.update_one(
            {"id": transaction.id},
            {"$set": {"stripe_session_id": session.session_id}}
        )
        
        return {
            "session_id": session.session_id,
            "checkout_url": session.url
        }
        
    except Exception as e:
        logger.error(f"Create checkout session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@app.get("/api/payments/checkout-status/{session_id}")
async def get_checkout_status(session_id: str):
    """Get checkout session status"""
    try:
        if not stripe_checkout:
            raise HTTPException(status_code=503, detail="Payment processing not available")
        
        status = await stripe_checkout.get_checkout_status(session_id)
        
        # Update local transaction
        await update_payment_status(session_id, status.status, status.payment_status)
        
        # If payment successful, update user
        if status.payment_status == "paid":
            transaction = await db.payment_transactions.find_one({"stripe_session_id": session_id})
            if transaction and transaction.get("payment_status") != "paid":
                user_id = transaction["metadata"]["user_id"]
                package_type = transaction["metadata"]["package_type"]
                package_id = transaction["metadata"]["package_id"]
                
                if package_type == "subscription":
                    package = SUBSCRIPTION_PACKAGES[package_id]
                    await db.users.update_one(
                        {"id": user_id},
                        {
                            "$set": {
                                "subscription_tier": package_id,
                                "credits": package.get("credits", 0),
                                "subscription_status": "active"
                            }
                        }
                    )
                
                # Mark transaction as processed
                await update_payment_status(session_id, status.status, "paid")
        
        return {
            "status": status.status,
            "payment_status": status.payment_status,
            "amount_total": status.amount_total,
            "currency": status.currency
        }
        
    except Exception as e:
        logger.error(f"Get checkout status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get checkout status")

# Business Advertising
@app.post("/api/advertisements")
async def create_advertisement(ad: BusinessAdvertisement):
    """Create business advertisement"""
    try:
        # Set expiration date
        ad.expires_at = ad.created_at + timedelta(days=ad.duration_days)
        
        await db.advertisements.insert_one(ad.dict())
        return {"message": "Advertisement created successfully", "ad_id": ad.id}
        
    except Exception as e:
        logger.error(f"Create advertisement error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create advertisement")

@app.get("/api/advertisements")
async def get_active_advertisements(ad_type: Optional[str] = None, limit: int = 10):
    """Get active advertisements for display"""
    try:
        query = {
            "is_active": True,
            "expires_at": {"$gt": datetime.utcnow()}
        }
        
        if ad_type:
            query["ad_type"] = ad_type
        
        ads = []
        async for ad in db.advertisements.find(query).limit(limit):
            if "_id" in ad:
                del ad["_id"]
            
            # Get business info
            business = await db.businesses.find_one({"id": ad["business_id"]})
            if business:
                ad["business_name"] = business.get("business_name")
                ad["business_address"] = business.get("address")
            
            # Track impression
            await db.advertisements.update_one(
                {"id": ad["id"]},
                {"$inc": {"impressions": 1}}
            )
            
            ads.append(ad)
        
        return {"advertisements": ads}
        
    except Exception as e:
        logger.error(f"Get advertisements error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get advertisements")

@app.post("/api/advertisements/{ad_id}/click")
async def track_ad_click(ad_id: str):
    """Track advertisement click"""
    try:
        await db.advertisements.update_one(
            {"id": ad_id},
            {"$inc": {"clicks": 1}}
        )
        
        return {"message": "Click tracked successfully"}
        
    except Exception as e:
        logger.error(f"Track click error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track click")

# Enhanced competitor search with consumer data
@app.post("/api/ultimate-competitor-search")
async def ultimate_competitor_search(search_request: Dict[str, Any]):
    """Ultimate competitor search with market intelligence"""
    try:
        location = search_request.get("location")
        radius = search_request.get("radius", 5)
        business_type = search_request.get("business_type", "restaurant")
        
        # Get Yelp data
        yelp_competitors = search_yelp_businesses_advanced(location, radius, business_type)
        
        # Enhance with local business data
        for competitor in yelp_competitors:
            # Check if business exists in our platform
            local_business = await db.businesses.find_one({"name": competitor["name"]})
            if local_business:
                competitor["is_platform_member"] = True
                competitor["platform_business_id"] = local_business["id"]
                
                # Add consumer reviews
                reviews = []
                async for review in db.consumer_reviews.find({"business_id": local_business["id"]}).limit(5):
                    if "_id" in review:
                        del review["_id"]
                    reviews.append(review)
                
                competitor["platform_reviews"] = reviews
            else:
                competitor["is_platform_member"] = False
        
        # Market intelligence
        market_intelligence = {
            "total_competitors": len(yelp_competitors),
            "platform_members": len([c for c in yelp_competitors if c.get("is_platform_member")]),
            "average_rating": sum(c.get("rating", 0) for c in yelp_competitors) / len(yelp_competitors) if yelp_competitors else 0,
            "price_distribution": {},
            "competitive_intensity": "High" if len(yelp_competitors) > 15 else "Medium"
        }
        
        return {
            "competitors": yelp_competitors,
            "market_intelligence": market_intelligence,
            "location": location,
            "total_found": len(yelp_competitors)
        }
        
    except Exception as e:
        logger.error(f"Ultimate competitor search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search competitors")

# Generate ultimate intelligence report
@app.post("/api/generate-ultimate-intelligence-report")
async def generate_ultimate_intelligence_report(request: Dict[str, Any]):
    """Generate ultimate intelligence report"""
    try:
        competitor_ids = request.get("competitor_ids", [])
        location = request.get("location", "")
        user_business = request.get("user_business")
        
        if not competitor_ids:
            raise HTTPException(status_code=400, detail="No competitors selected")
        
        # Mock competitor data for now
        competitors = []
        for comp_id in competitor_ids:
            competitor = {
                "id": comp_id,
                "name": f"Restaurant {comp_id[:8]}",
                "rating": 4.2,
                "review_count": 200,
                "price_level": 2,
                "business_metrics": {
                    "estimated_daily_traffic": 150,
                    "estimated_monthly_revenue": 45000,
                    "competitive_strength": 75
                }
            }
            competitors.append(competitor)
        
        # Run AI analysis
        pricing_analysis = pricewatch_agent_analysis(competitors, location)
        sentiment_analysis = sentiment_agent_analysis(competitors, location)
        
        # Generate comprehensive report
        report = {
            "id": str(uuid.uuid4()),
            "report_type": "ultimate_intelligence",
            "location": location,
            "competitors": competitors,
            "total_competitors": len(competitors),
            "report_date": datetime.utcnow().isoformat(),
            "pricing_analysis": pricing_analysis,
            "sentiment_analysis": sentiment_analysis,
            "executive_summary": f"Comprehensive analysis of {len(competitors)} competitors in {location}",
            "strategic_recommendations": [
                "Focus on customer experience differentiation",
                "Optimize pricing strategy based on market analysis",
                "Enhance digital presence and marketing"
            ],
            "market_opportunities": [
                "Underserved customer segments identified",
                "Technology integration opportunities",
                "Expansion potential in adjacent markets"
            ]
        }
        
        # Store report
        await db.ultimate_reports.insert_one(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Generate ultimate report error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

# Get subscription tiers
@app.get("/api/subscription-tiers")
async def get_subscription_tiers():
    """Get subscription tiers with pricing"""
    return {
        "tiers": [
            {
                "id": "starter",
                "name": "Starter",
                "price": 0,
                "features": [
                    "Basic competitor search",
                    "10 searches per month",
                    "Limited analytics",
                    "Email support"
                ],
                "credits": 10,
                "popular": False
            },
            {
                "id": "professional",
                "name": "Professional",
                "price": 149,
                "features": [
                    "Advanced competitor intelligence",
                    "500 searches per month",
                    "Full AI agent suite",
                    "Advanced analytics",
                    "Priority support",
                    "Business listing on consumer marketplace"
                ],
                "credits": 500,
                "popular": True
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": 399,
                "features": [
                    "Unlimited intelligence reports",
                    "2000+ searches per month",
                    "Custom AI agents",
                    "White-label solutions",
                    "Dedicated account manager",
                    "Premium marketplace placement",
                    "Advanced advertising tools"
                ],
                "credits": 2000,
                "popular": False
            }
        ],
        "advertising_packages": [
            {
                "id": "basic",
                "name": "Basic Advertising",
                "price": 29,
                "duration_days": 7,
                "features": ["Basic listing highlight", "1000 impressions"]
            },
            {
                "id": "featured",
                "name": "Featured Placement",
                "price": 99,
                "duration_days": 30,
                "features": ["Featured placement", "5000 impressions", "Analytics"]
            },
            {
                "id": "premium",
                "name": "Premium Advertising",
                "price": 299,
                "duration_days": 30,
                "features": ["Premium placement", "Unlimited impressions", "Advanced targeting"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)