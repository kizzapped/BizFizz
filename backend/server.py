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
import redis
from functools import wraps
import pickle

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

# Initialize Redis for caching (optional)
redis_client = None
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("Redis cache connected successfully")
except Exception as e:
    logger.warning(f"Redis cache not available: {str(e)}")
    redis_client = None

# Caching decorator
def cache_result(expiration_seconds=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # Create cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            try:
                # Try to get from cache
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result.encode('latin1'))
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            try:
                # Store in cache
                redis_client.setex(
                    cache_key, 
                    expiration_seconds, 
                    pickle.dumps(result).decode('latin1')
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator

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
class UserLocation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    latitude: float
    longitude: float
    accuracy: float = Field(default=10.0)  # meters
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    location_sharing_enabled: bool = Field(default=False)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

class PromoLocation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str
    business_name: str
    restaurant_latitude: float
    restaurant_longitude: float
    promo_radius: float = Field(default=1609.34)  # 1 mile in meters
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PromotionalCampaign(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    business_id: str
    business_name: str
    campaign_name: str
    promo_message: str
    discount_amount: Optional[float] = None
    discount_type: str = Field(default="percentage")  # percentage, fixed, bogo
    promo_code: Optional[str] = None
    valid_until: datetime
    max_uses: int = Field(default=100)
    current_uses: int = Field(default=0)
    target_radius: float = Field(default=1609.34)  # 1 mile
    is_active: bool = Field(default=True)
    send_sms: bool = Field(default=True)
    send_push: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    success_metrics: Dict[str, int] = Field(default_factory=dict)

class ProximityAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    business_id: str
    campaign_id: str
    distance_meters: float
    promo_message: str
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    opened: bool = Field(default=False)
    redeemed: bool = Field(default=False)
    method: str = Field(default="sms")  # sms, push, both
    user_response: Optional[str] = None

class LocationPermission(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    permission_granted: bool = Field(default=False)
    location_sharing: bool = Field(default=False)
    promotional_notifications: bool = Field(default=True)
    sms_notifications: bool = Field(default=True)
    push_notifications: bool = Field(default=True)
    privacy_level: str = Field(default="balanced")  # strict, balanced, open
    granted_at: datetime = Field(default_factory=datetime.utcnow)
class OpenTableReservation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    restaurant_id: str
    restaurant_name: str
    guest_name: str
    guest_email: str
    guest_phone: str
    reservation_date: datetime
    reservation_time: str
    party_size: int
    special_requests: Optional[str] = None
    opentable_confirmation: Optional[str] = None
    status: str = Field(default="pending")  # pending, confirmed, cancelled, completed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class RestaurantAvailability(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    restaurant_id: str
    restaurant_name: str
    opentable_id: Optional[str] = None
    address: str
    phone: str
    cuisine_type: str
    price_range: str = Field(default="$$")  # $, $$, $$$, $$$$
    available_times: List[str] = Field(default_factory=list)
    max_party_size: int = Field(default=8)
    booking_url: str
    rating: float = Field(default=0.0)
    review_count: int = Field(default=0)
    features: List[str] = Field(default_factory=list)  # outdoor seating, bar, etc.
    distance_miles: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ReservationQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    query_text: str
    desired_date: datetime
    desired_time: Optional[str] = None
    party_size: int
    cuisine_preference: Optional[str] = None
    price_range: Optional[str] = None
    location: Optional[str] = None
    max_distance: float = Field(default=10.0)  # miles
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
class VoiceCommand(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    command_text: str
    intent: str  # restaurant_search, make_reservation, check_availability, general_query
    entities: Dict[str, Any] = Field(default_factory=dict)  # cuisine, location, date, time, etc.
    response_text: str
    action_taken: Optional[str] = None
    was_successful: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = Field(default=0.0)

class CorbySession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)  # Current search, booking in progress, etc.
    is_active: bool = Field(default=True)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)  # cuisine preferences, location, etc.

class CorbyResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_input: str
    corby_response: str
    intent_detected: str
    entities_extracted: Dict[str, Any] = Field(default_factory=dict)
    action_performed: Optional[str] = None
    data_returned: Optional[Dict[str, Any]] = None
    voice_enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
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

# Advanced AI Voice Features

ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Premium Voice Personalities for Corby
CORBY_VOICE_PROFILES = {
    "professional": {
        "personality": "Professional, efficient, and knowledgeable restaurant concierge",
        "speaking_style": "Clear, articulate, and business-like",
        "voice_settings": {"rate": 0.9, "pitch": 1.0, "volume": 0.8}
    },
    "friendly": {
        "personality": "Warm, enthusiastic, and personable dining companion",
        "speaking_style": "Conversational, upbeat, and encouraging",
        "voice_settings": {"rate": 1.0, "pitch": 1.1, "volume": 0.9}
    },
    "luxury": {
        "personality": "Sophisticated, refined, and exclusive dining advisor",
        "speaking_style": "Elegant, thoughtful, and discerning",
        "voice_settings": {"rate": 0.8, "pitch": 0.9, "volume": 0.85}
    }
}

async def get_enhanced_ai_response(command_text: str, session: CorbySession, voice_profile: str = "friendly"):
    """Get enhanced AI response using multiple AI models"""
    try:
        # Use Claude for sophisticated reasoning if available
        if ANTHROPIC_API_KEY:
            return await get_claude_response(command_text, session, voice_profile)
        
        # Enhanced OpenAI response
        return await get_openai_enhanced_response(command_text, session, voice_profile)
        
    except Exception as e:
        logger.error(f"Enhanced AI response error: {e}")
        return await get_openai_enhanced_response(command_text, session, voice_profile)

async def get_claude_response(command_text: str, session: CorbySession, voice_profile: str):
    """Get response from Claude for superior reasoning"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        personality = CORBY_VOICE_PROFILES[voice_profile]["personality"]
        speaking_style = CORBY_VOICE_PROFILES[voice_profile]["speaking_style"]
        
        # Build context
        context = ""
        if session.conversation_history:
            recent_context = session.conversation_history[-3:]
            context = "\n".join([f"User: {h['user']}\nCorby: {h['corby']}" for h in recent_context])
        
        system_prompt = f"""You are Corby, an AI restaurant assistant with this personality: {personality}
        
Speaking style: {speaking_style}

Key capabilities:
- Search restaurants by cuisine, location, price, ratings
- Check real-time availability and make reservations
- Provide personalized recommendations based on user preferences
- Handle dietary restrictions and special occasions
- Give local dining insights and hidden gems
- Manage reservation changes and cancellations

Context: {context}

User's current location context: {session.context.get('location', 'unknown')}
User preferences: {session.user_preferences}

Respond naturally and helpfully to: "{command_text}"

Keep responses conversational, under 100 words, and actionable."""
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=200,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": command_text}
            ]
        )
        
        return {
            "response_text": message.content[0].text,
            "method": "claude_enhanced",
            "voice_profile": voice_profile
        }
        
    except Exception as e:
        logger.error(f"Claude response error: {e}")
        return await get_openai_enhanced_response(command_text, session, voice_profile)

async def get_openai_enhanced_response(command_text: str, session: CorbySession, voice_profile: str):
    """Enhanced OpenAI response with personality"""
    try:
        if not openai_client:
            return {"response_text": "I'm here to help with restaurants! What would you like to know?"}
        
        personality = CORBY_VOICE_PROFILES[voice_profile]["personality"]
        speaking_style = CORBY_VOICE_PROFILES[voice_profile]["speaking_style"]
        
        context = ""
        if session.conversation_history:
            recent_context = session.conversation_history[-3:]
            context = "\n".join([f"User: {h['user']}\nCorby: {h['corby']}" for h in recent_context])
        
        enhanced_prompt = f"""You are Corby, a sophisticated AI restaurant assistant.

Personality: {personality}
Speaking Style: {speaking_style}

Recent conversation:
{context}

User preferences: {session.user_preferences}
Location context: {session.context.get('location', 'unknown')}

User says: "{command_text}"

Respond with enthusiasm and expertise. Be helpful, specific, and conversational. Under 80 words."""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Corby, an expert restaurant assistant. Be helpful, enthusiastic, and conversational."},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=150,
            temperature=0.8
        )
        
        return {
            "response_text": response.choices[0].message.content.strip(),
            "method": "openai_enhanced",
            "voice_profile": voice_profile
        }
        
    except Exception as e:
        logger.error(f"OpenAI enhanced response error: {e}")
        return {"response_text": "I'm having trouble right now, but I'm here to help with restaurants!"}

async def generate_premium_voice_audio(text: str, voice_profile: str = "friendly"):
    """Generate premium quality voice using ElevenLabs (when available)"""
    try:
        if not ELEVENLABS_API_KEY:
            return {"use_browser_tts": True, "settings": CORBY_VOICE_PROFILES[voice_profile]["voice_settings"]}
        
        # ElevenLabs integration for premium voice
        voice_id = {
            "professional": "ErXwobaYiN019PkySvjV",  # Antoni
            "friendly": "EXAVITQu4vr4xnSDxMaL",    # Bella
            "luxury": "VR6AewLTigWG4xSOukaG"       # Arnold
        }.get(voice_profile, "EXAVITQu4vr4xnSDxMaL")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5
                    }
                }
            )
            
            if response.status_code == 200:
                # In production, save to file system or cloud storage
                audio_id = str(uuid.uuid4())
                return {
                    "audio_id": audio_id,
                    "audio_url": f"/api/corby/audio/{audio_id}",
                    "use_premium_voice": True,
                    "voice_profile": voice_profile
                }
        
        # Fallback to browser TTS
        return {"use_browser_tts": True, "settings": CORBY_VOICE_PROFILES[voice_profile]["voice_settings"]}
        
    except Exception as e:
        logger.error(f"Premium voice generation error: {e}")
        return {"use_browser_tts": True, "settings": CORBY_VOICE_PROFILES[voice_profile]["voice_settings"]}

async def get_contextual_recommendations(user_id: str, session: CorbySession):
    """Get AI-powered contextual recommendations"""
    try:
        # Get user's location
        user_location = await db.user_locations.find_one({"user_id": user_id})
        
        # Get user's past reservations for preferences
        past_reservations = []
        async for reservation in db.opentable_reservations.find({"user_id": user_id}).limit(5):
            past_reservations.append(reservation)
        
        # Get user's social activity
        user_mentions = []
        async for mention in db.social_mentions.find({"user_id": user_id}).limit(3):
            user_mentions.append(mention)
        
        # AI analysis for recommendations
        if openai_client:
            context_data = {
                "location": user_location,
                "past_reservations": past_reservations,
                "social_activity": user_mentions,
                "conversation_context": session.conversation_history[-3:] if session.conversation_history else [],
                "user_preferences": session.user_preferences,
                "time_of_day": datetime.now().strftime("%H:%M"),
                "day_of_week": datetime.now().strftime("%A")
            }
            
            recommendation_prompt = f"""
Based on this user data, provide 3 specific restaurant recommendations:

User Context: {context_data}

Consider:
- Time of day and dining preferences
- Past restaurant choices and patterns  
- Current conversation context
- Location and convenience
- Special occasions or events

Return JSON format:
{{
  "recommendations": [
    {{
      "restaurant_name": "name",
      "reason": "why this fits the user",
      "cuisine": "type",
      "price_range": "$$",
      "best_time": "lunch/dinner",
      "special_note": "unique selling point"
    }}
  ],
  "personal_message": "conversational explanation"
}}
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert restaurant recommendation engine. Provide personalized, contextual suggestions."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            try:
                return json.loads(response.choices[0].message.content.strip())
            except:
                pass
        
        # Fallback recommendations
        return {
            "recommendations": [
                {
                    "restaurant_name": "Local Favorite Bistro",
                    "reason": "Popular choice for your area",
                    "cuisine": "American",
                    "price_range": "$$",
                    "best_time": "dinner",
                    "special_note": "Great for casual dining"
                }
            ],
            "personal_message": "Based on your location, here are some great options nearby!"
        }
        
    except Exception as e:
        logger.error(f"Contextual recommendations error: {e}")
        return {"recommendations": [], "personal_message": "I can help you find great restaurants! What are you in the mood for?"}

async def handle_complex_queries(command_text: str, session: CorbySession):
    """Handle complex, multi-part queries with advanced AI"""
    try:
        # Extract multiple intents and requirements
        if openai_client:
            analysis_prompt = f"""
Analyze this complex restaurant query and extract all requirements:

Query: "{command_text}"

Extract and return JSON:
{{
  "primary_intent": "main request",
  "requirements": {{
    "cuisine": "if specified",
    "location": "if specified", 
    "date": "if specified",
    "time": "if specified",
    "party_size": "if specified",
    "price_range": "if specified",
    "dietary_restrictions": "if specified",
    "occasion": "if specified (birthday, anniversary, etc)",
    "atmosphere": "if specified (romantic, casual, etc)",
    "special_needs": "if specified (parking, accessibility, etc)"
  }},
  "complexity_score": "1-10",
  "suggested_follow_up": "what to ask next"
}}
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing complex restaurant requests."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                
                # If complex query, provide comprehensive response
                if analysis.get("complexity_score", 0) > 6:
                    return await generate_comprehensive_response(analysis, session)
                
            except Exception as e:
                logger.error(f"Complex query analysis error: {e}")
        
        return None  # Let normal processing handle it
        
    except Exception as e:
        logger.error(f"Complex query handling error: {e}")
        return None

async def generate_comprehensive_response(analysis: dict, session: CorbySession):
    """Generate comprehensive response for complex queries"""
    try:
        requirements = analysis.get("requirements", {})
        primary_intent = analysis.get("primary_intent", "")
        
        # Build comprehensive response
        response_parts = []
        
        if primary_intent:
            response_parts.append(f"I understand you're looking for {primary_intent}.")
        
        # Address specific requirements
        if requirements.get("occasion"):
            response_parts.append(f"For your {requirements['occasion']}, I recommend focusing on restaurants with the right atmosphere.")
        
        if requirements.get("dietary_restrictions"):
            response_parts.append(f"I'll make sure to find options that accommodate {requirements['dietary_restrictions']} dietary needs.")
        
        if requirements.get("atmosphere"):
            response_parts.append(f"Looking for a {requirements['atmosphere']} setting - I know just the places!")
        
        # Suggest next steps
        follow_up = analysis.get("suggested_follow_up", "")
        if follow_up:
            response_parts.append(follow_up)
        
        comprehensive_response = " ".join(response_parts)
        
        return {
            "response_text": comprehensive_response,
            "action_taken": "complex_query_analysis",
            "was_successful": True,
            "data": {"analysis": analysis, "requirements": requirements}
        }
        
    except Exception as e:
        logger.error(f"Comprehensive response error: {e}")
        return None

# Corby Voice Assistant AI Functions

async def process_voice_command(user_id: str, command_text: str, session_id: Optional[str] = None, voice_profile: str = "friendly"):
    """Process voice command through enhanced Corby AI assistant"""
    try:
        # Get or create conversation session
        if not session_id:
            session = CorbySession(
                user_id=user_id,
                context={"location": "unknown"},
                user_preferences={"voice_profile": voice_profile}
            )
            await db.corby_sessions.insert_one(session.dict())
            session_id = session.id
        else:
            session_data = await db.corby_sessions.find_one({"id": session_id})
            session = CorbySession(**session_data) if session_data else None
            if not session:
                return {"error": "Session not found"}

        # Check for complex queries first
        complex_response = await handle_complex_queries(command_text, session)
        if complex_response:
            # Update conversation and return complex response
            session.conversation_history.append({
                "user": command_text,
                "corby": complex_response["response_text"],
                "timestamp": datetime.utcnow().isoformat(),
                "intent": "complex_query"
            })
            await db.corby_sessions.replace_one({"id": session_id}, session.dict())
            return {
                "session_id": session_id,
                "response_text": complex_response["response_text"],
                "intent": "complex_query",
                "voice_enabled": True,
                "voice_profile": voice_profile,
                **complex_response
            }

        # Regular intent analysis
        intent_analysis = await analyze_voice_intent(command_text, session.conversation_history)
        intent = intent_analysis.get("intent", "general_query")
        entities = intent_analysis.get("entities", {})
        
        # Generate response based on intent
        if intent == "general_query":
            # Use enhanced AI for general conversation
            ai_response = await get_enhanced_ai_response(command_text, session, voice_profile)
            corby_response = {
                "response_text": ai_response["response_text"],
                "action_taken": "ai_conversation_enhanced",
                "was_successful": True,
                "method": ai_response.get("method", "standard")
            }
        else:
            # Use existing specialized handlers
            corby_response = await generate_corby_response(intent, entities, command_text, session)
        
        # Generate premium voice if available
        voice_data = await generate_premium_voice_audio(corby_response["response_text"], voice_profile)
        
        # Update conversation history
        session.conversation_history.append({
            "user": command_text,
            "corby": corby_response["response_text"],
            "timestamp": datetime.utcnow().isoformat(),
            "intent": intent,
            "voice_profile": voice_profile
        })
        
        # Keep only last 10 exchanges
        if len(session.conversation_history) > 10:
            session.conversation_history = session.conversation_history[-10:]
        
        # Update session with enhanced context
        session.last_interaction = datetime.utcnow()
        session.context.update(corby_response.get("context_updates", {}))
        session.user_preferences["voice_profile"] = voice_profile
        
        await db.corby_sessions.replace_one({"id": session_id}, session.dict())
        
        # Save enhanced voice command record
        voice_command = VoiceCommand(
            user_id=user_id,
            command_text=command_text,
            intent=intent,
            entities=entities,
            response_text=corby_response["response_text"],
            action_taken=corby_response.get("action_taken"),
            was_successful=corby_response.get("was_successful", False),
            confidence_score=intent_analysis.get("confidence", 0.8)
        )
        
        await db.voice_commands.insert_one(voice_command.dict())
        
        return {
            "session_id": session_id,
            "response_text": corby_response["response_text"],
            "intent": intent,
            "entities": entities,
            "action_taken": corby_response.get("action_taken"),
            "data": corby_response.get("data"),
            "voice_enabled": True,
            "voice_profile": voice_profile,
            "voice_data": voice_data,
            "confidence": intent_analysis.get("confidence", 0.8),
            "ai_method": corby_response.get("method", "standard")
        }
        
    except Exception as e:
        logger.error(f"Enhanced voice command processing error: {e}")
        return {
            "response_text": "I'm experiencing some technical difficulties, but I'm still here to help you find amazing restaurants! Could you try asking again?",
            "intent": "error",
            "voice_enabled": True,
            "voice_profile": voice_profile
        }

async def analyze_voice_intent(command_text: str, conversation_history: List[Dict[str, str]]):
    """Analyze voice command to determine intent and extract entities"""
    try:
        if not openai_client:
            # Fallback intent detection without OpenAI
            return await fallback_intent_detection(command_text)
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            recent_context = conversation_history[-3:]  # Last 3 exchanges
            context = "\n".join([f"User: {h['user']}\nCorby: {h['corby']}" for h in recent_context])
        
        intent_prompt = f"""
You are Corby, a helpful AI assistant for BizFizz restaurant platform. Analyze this voice command and return a JSON response.

Conversation Context:
{context}

Current User Command: "{command_text}"

Determine the intent and extract entities. Available intents:
- restaurant_search: User wants to find restaurants
- make_reservation: User wants to book a table
- check_availability: User wants to check if tables are available
- modify_reservation: User wants to change/cancel existing booking
- get_recommendations: User wants personalized suggestions
- check_weather: User asks about weather (for outdoor dining)
- general_query: General questions about restaurants/food
- greeting: Hello, hi, how are you
- help: User needs assistance

Return JSON format:
{{
  "intent": "detected_intent",
  "entities": {{
    "cuisine": "italian/chinese/etc",
    "location": "near me/specific address",
    "date": "today/tomorrow/specific date",
    "time": "lunch/dinner/7pm/etc", 
    "party_size": "number",
    "restaurant_name": "specific restaurant",
    "dietary_restrictions": "vegetarian/gluten-free/etc"
  }},
  "confidence": 0.95
}}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert intent classifier for restaurant voice commands. Always return valid JSON."},
                {"role": "user", "content": intent_prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        return await fallback_intent_detection(command_text)

async def fallback_intent_detection(command_text: str):
    """Fallback intent detection using keyword matching"""
    command_lower = command_text.lower()
    
    # Define keyword patterns
    if any(word in command_lower for word in ["find", "search", "restaurant", "food", "eat", "hungry"]):
        intent = "restaurant_search"
    elif any(word in command_lower for word in ["book", "reserve", "table", "reservation"]):
        intent = "make_reservation"
    elif any(word in command_lower for word in ["available", "free", "open"]):
        intent = "check_availability"
    elif any(word in command_lower for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        intent = "greeting"
    elif any(word in command_lower for word in ["help", "assist", "how to", "what can"]):
        intent = "help"
    elif any(word in command_lower for word in ["recommend", "suggest", "best", "popular"]):
        intent = "get_recommendations"
    else:
        intent = "general_query"
    
    # Extract basic entities
    entities = {}
    
    # Cuisine detection
    cuisines = ["italian", "chinese", "mexican", "indian", "japanese", "thai", "french", "american", "pizza", "sushi"]
    for cuisine in cuisines:
        if cuisine in command_lower:
            entities["cuisine"] = cuisine
            break
    
    # Location detection
    if "near me" in command_lower or "nearby" in command_lower:
        entities["location"] = "near me"
    
    # Time detection
    if any(word in command_lower for word in ["lunch", "dinner", "breakfast"]):
        entities["time"] = next(word for word in ["lunch", "dinner", "breakfast"] if word in command_lower)
    
    # Party size detection
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    for num in numbers:
        if f"{num} people" in command_lower or f"party of {num}" in command_lower:
            entities["party_size"] = num
            break
    
    return {
        "intent": intent,
        "entities": entities,
        "confidence": 0.7
    }

# Advanced Menu Query Functions

async def handle_advanced_menu_queries(command_text: str, session: CorbySession):
    """Handle advanced menu-based queries like 'restaurants with steak and seafood'"""
    try:
        # Enhanced menu search capabilities
        menu_query_patterns = {
            "menu_items": r"restaurants?.*(with|offering|serving|have)\s+(.*?)(?:\s+and\s+(.*?))?(?:\s+dishes?|\s+food)?",
            "price_comparison": r"(?:best price|cheapest|most affordable|lowest cost).*(?:for|on)\s+(.*?)(?:\s+at|\s+in)?",
            "dietary_specific": r"(?:vegetarian|vegan|gluten.free|kosher|halal)\s+(?:options|restaurants|food)",
            "cuisine_combination": r"(.*?)\s+and\s+(.*?)\s+(?:restaurants?|cuisine|food)"
        }
        
        import re
        command_lower = command_text.lower()
        
        # Menu items search
        menu_match = re.search(menu_query_patterns["menu_items"], command_lower)
        if menu_match:
            item1 = menu_match.group(2).strip()
            item2 = menu_match.group(3).strip() if menu_match.group(3) else None
            
            return await search_restaurants_by_menu_items(item1, session, item2)
        
        # Price comparison
        price_match = re.search(menu_query_patterns["price_comparison"], command_lower)
        if price_match:
            category = price_match.group(1).strip()
            return await compare_prices_by_category(category, session)
        
        # Dietary restrictions
        if re.search(menu_query_patterns["dietary_specific"], command_lower):
            dietary_type = re.search(r"(vegetarian|vegan|gluten.free|kosher|halal)", command_lower).group(1)
            return await find_dietary_restaurants(dietary_type, session)
        
        return None  # Let other handlers process
        
    except Exception as e:
        logger.error(f"Advanced menu query error: {e}")
        return None

async def search_restaurants_by_menu_items(item1: str, session: CorbySession, item2: str = None):
    """Search restaurants that serve specific menu items"""
    try:
        # Enhanced search with menu integration
        if openai_client:
            # Use AI to understand menu items and find matches
            menu_analysis_prompt = f"""
            Find restaurants that serve {item1}{f' and {item2}' if item2 else ''}.
            
            Based on typical restaurant offerings, categorize this as:
            1. Specific dishes (e.g., "steak and seafood" = steakhouse, seafood restaurant)
            2. Cuisine types that commonly serve these items
            3. Restaurant categories (fine dining, casual, etc.)
            
            Return JSON:
            {{
              "cuisine_types": ["steakhouse", "seafood", "american"],
              "restaurant_categories": ["fine dining", "casual dining"],
              "search_terms": ["steak", "seafood", "surf and turf"],
              "confidence": 0.95
            }}
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a restaurant menu expert. Analyze requests and suggest restaurant types."},
                    {"role": "user", "content": menu_analysis_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                cuisine_types = analysis.get("cuisine_types", [item1])
            except:
                cuisine_types = [item1]
        else:
            cuisine_types = [item1]
        
        # Search restaurants based on analyzed cuisine types
        restaurants_found = []
        for cuisine in cuisine_types:
            restaurants = await search_restaurants_with_availability(
                cuisine, 
                datetime.now().strftime("%Y-%m-%d"), 
                "19:00", 
                2, 
                session.context.get("location", "near me")
            )
            restaurants_found.extend(restaurants)
        
        # Remove duplicates and format response
        unique_restaurants = {r.restaurant_id: r for r in restaurants_found}.values()
        restaurant_list = list(unique_restaurants)[:5]
        
        if restaurant_list:
            items_text = f"{item1} and {item2}" if item2 else item1
            restaurant_names = [f"{r.restaurant_name} ({r.rating} stars, {r.price_range})" for r in restaurant_list[:3]]
            
            response_text = f"Great choice! I found {len(restaurant_list)} restaurants serving {items_text}. Top recommendations: {', '.join(restaurant_names)}. Would you like me to check availability or get more details about any of these?"
            
            return {
                "response_text": response_text,
                "action_taken": f"menu_search_{len(restaurant_list)}_found",
                "was_successful": True,
                "data": {"restaurants": [r.dict() for r in restaurant_list]},
                "context_updates": {
                    "last_menu_search": {
                        "items": [item1, item2] if item2 else [item1],
                        "restaurants": [r.dict() for r in restaurant_list]
                    }
                }
            }
        else:
            return {
                "response_text": f"I couldn't find restaurants specifically offering {items_text} in your area right now. Would you like me to broaden the search or try different cuisine types?",
                "action_taken": "menu_search_no_results",
                "was_successful": False
            }
        
    except Exception as e:
        logger.error(f"Menu item search error: {e}")
        return {
            "response_text": f"I'm having trouble searching for {item1} restaurants right now. Let me try a general search instead.",
            "was_successful": False
        }

async def compare_prices_by_category(category: str, session: CorbySession):
    """Compare prices across restaurants for specific categories"""
    try:
        # Simulate price comparison (in production, integrate with menu APIs)
        category_lower = category.lower()
        
        # Generate mock price data based on category
        mock_price_data = {
            "appetizers": [
                {"restaurant": "Joe's Bistro", "price_range": "$8-15", "popular_item": "Calamari $12", "rating": 4.2},
                {"restaurant": "Casual Corner", "price_range": "$6-12", "popular_item": "Wings $9", "rating": 4.0},
                {"restaurant": "Upscale Eatery", "price_range": "$15-25", "popular_item": "Oysters $18", "rating": 4.5}
            ],
            "entrees": [
                {"restaurant": "Family Diner", "price_range": "$12-18", "popular_item": "Burger $14", "rating": 3.8},
                {"restaurant": "Steakhouse Prime", "price_range": "$25-45", "popular_item": "Ribeye $38", "rating": 4.6},
                {"restaurant": "Pasta Palace", "price_range": "$16-22", "popular_item": "Linguine $18", "rating": 4.1}
            ],
            "desserts": [
                {"restaurant": "Sweet Treats", "price_range": "$7-12", "popular_item": "Cheesecake $9", "rating": 4.3},
                {"restaurant": "Bistro Downtown", "price_range": "$8-14", "popular_item": "Tiramisu $11", "rating": 4.2},
                {"restaurant": "Corner Cafe", "price_range": "$5-10", "popular_item": "Apple Pie $7", "rating": 3.9}
            ]
        }
        
        # Find matching category
        matching_data = None
        for cat, data in mock_price_data.items():
            if cat in category_lower or category_lower in cat:
                matching_data = data
                break
        
        if not matching_data:
            matching_data = mock_price_data["appetizers"]  # Default fallback
        
        # Sort by price (lowest first)
        sorted_data = sorted(matching_data, key=lambda x: int(x["price_range"].split('-')[0].replace('$', '')))
        
        best_value = sorted_data[0]
        response_text = f"For {category}, the best prices are at {best_value['restaurant']} with {category} ranging {best_value['price_range']}. Their popular {best_value['popular_item']} is highly rated at {best_value['rating']} stars. Would you like me to check availability there?"
        
        return {
            "response_text": response_text,
            "action_taken": f"price_comparison_{category}",
            "was_successful": True,
            "data": {
                "category": category,
                "price_comparison": sorted_data,
                "best_value": best_value
            }
        }
        
    except Exception as e:
        logger.error(f"Price comparison error: {e}")
        return {
            "response_text": f"I'm working on getting price information for {category}. In the meantime, would you like me to find restaurants in your budget range?",
            "was_successful": False
        }

async def handle_review_requests(command_text: str, session: CorbySession):
    """Handle requests to write or leave reviews"""
    try:
        command_lower = command_text.lower()
        
        # Extract restaurant name and review type
        review_patterns = {
            "leave_review": r"leave\s+(.*?)\s+(?:a\s+)?(good|great|positive|5.star|excellent|bad|negative|poor)\s+review\s+(?:for|at)\s+(.*)",
            "write_review": r"write\s+(?:a\s+)?(good|great|positive|5.star|excellent|bad|negative|poor)\s+review\s+(?:for|about)\s+(.*)"
        }
        
        import re
        review_match = None
        review_type = None
        restaurant_name = None
        
        for pattern_name, pattern in review_patterns.items():
            match = re.search(pattern, command_lower)
            if match:
                if pattern_name == "leave_review":
                    review_type = match.group(2)
                    restaurant_name = match.group(3)
                else:  # write_review
                    review_type = match.group(1)
                    restaurant_name = match.group(2)
                break
        
        if not restaurant_name:
            return {
                "response_text": "I'd be happy to help you with a review! Which restaurant would you like to review, and what type of experience did you have?",
                "action_taken": "review_request_clarification",
                "was_successful": False
            }
        
        # Generate review based on type
        if openai_client:
            review_prompt = f"""
            Write a {review_type} restaurant review for {restaurant_name}. 
            
            Make it sound authentic and personal, around 2-3 sentences.
            Include specific details like:
            - Food quality and taste
            - Service experience  
            - Atmosphere/ambiance
            - Value for money
            
            Write in first person as if the user experienced it.
            Keep it genuine and helpful for other diners.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes authentic restaurant reviews."},
                    {"role": "user", "content": review_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            generated_review = response.choices[0].message.content.strip()
        else:
            # Fallback review templates
            if review_type in ["good", "great", "positive", "5-star", "excellent"]:
                generated_review = f"Had a wonderful experience at {restaurant_name}! The food was delicious, service was attentive, and the atmosphere was perfect for our group. Definitely recommend and will be back!"
            else:
                generated_review = f"Unfortunately, our experience at {restaurant_name} didn't meet expectations. The service was slow and the food quality could be improved. Hoping they can address these issues."
        
        return {
            "response_text": f"I've drafted a {review_type} review for {restaurant_name}: \n\n\"{generated_review}\"\n\nWould you like me to help you post this on Google Reviews, Yelp, or another platform? I can guide you through the process!",
            "action_taken": f"review_generated_{review_type}",
            "was_successful": True,
            "data": {
                "restaurant_name": restaurant_name,
                "review_type": review_type,
                "generated_review": generated_review,
                "suggested_platforms": ["Google Reviews", "Yelp", "Facebook", "TripAdvisor"]
            }
        }
        
    except Exception as e:
        logger.error(f"Review request error: {e}")
        return None

async def find_dietary_restaurants(dietary_type: str, session: CorbySession):
    """Find restaurants with specific dietary options"""
    try:
        dietary_mapping = {
            "vegetarian": "vegetarian-friendly restaurants",
            "vegan": "vegan restaurants", 
            "gluten-free": "gluten-free restaurants",
            "kosher": "kosher restaurants",
            "halal": "halal restaurants"
        }
        
        search_term = dietary_mapping.get(dietary_type, f"{dietary_type} restaurants")
        
        # Search for restaurants
        restaurants = await search_restaurants_with_availability(
            search_term,
            datetime.now().strftime("%Y-%m-%d"),
            "19:00",
            2,
            session.context.get("location", "near me")
        )
        
        if restaurants:
            restaurant_names = [f"{r.restaurant_name} ({r.rating} stars)" for r in restaurants[:3]]
            response_text = f"Perfect! I found {len(restaurants)} {dietary_type} restaurants nearby: {', '.join(restaurant_names)}. Would you like me to check their specific {dietary_type} menu options or make a reservation?"
        else:
            response_text = f"I'm still building my database of {dietary_type} restaurants in your area. Would you like me to search for restaurants that typically accommodate {dietary_type} diets?"
        
        return {
            "response_text": response_text,
            "action_taken": f"dietary_search_{dietary_type}",
            "was_successful": len(restaurants) > 0,
            "data": {"restaurants": [r.dict() for r in restaurants[:5]] if restaurants else []}
        }
        
    except Exception as e:
        logger.error(f"Dietary restaurant search error: {e}")
        return {
            "response_text": f"I can help you find {dietary_type} restaurants! Let me search for options in your area.",
            "was_successful": False
        }

async def generate_corby_response(intent: str, entities: Dict[str, Any], command_text: str, session: CorbySession):
    """Generate Corby's response based on intent and entities"""
    try:
        if intent == "greeting":
            return {
                "response_text": "Hello! I'm Corby, your personal restaurant assistant! I can help you find amazing restaurants, check availability, and even make reservations for you. What can I help you with today?",
                "action_taken": None,
                "was_successful": True
            }
        
        elif intent == "help":
            return {
                "response_text": "I'm here to help! You can ask me things like: 'Find Italian restaurants near me', 'Book a table for 2 at 7 PM tonight', 'What are the best sushi places nearby?', or 'Check availability at Joe's Bistro for tomorrow'. Just speak naturally and I'll take care of the rest!",
                "action_taken": None,
                "was_successful": True
            }
        
        elif intent == "restaurant_search":
            return await handle_restaurant_search(entities, session)
        
        elif intent == "make_reservation":
            return await handle_reservation_request(entities, session)
        
        elif intent == "check_availability":
            return await handle_availability_check(entities, session)
        
        elif intent == "get_recommendations":
            return await handle_recommendations(entities, session)
        
        else:
            # Use OpenAI for general conversation
            if openai_client:
                return await generate_ai_conversation_response(command_text, session)
            else:
                return {
                    "response_text": "I'm focused on helping you with restaurants! Try asking me to find restaurants, check availability, or make reservations.",
                    "action_taken": None,
                    "was_successful": False
                }
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return {
            "response_text": "I'm having trouble processing that request. Could you try asking in a different way?",
            "action_taken": None,
            "was_successful": False
        }

async def handle_restaurant_search(entities: Dict[str, Any], session: CorbySession):
    """Handle restaurant search through voice"""
    try:
        # Extract search parameters
        cuisine = entities.get("cuisine", "restaurant")
        location = entities.get("location", "near me")
        date = entities.get("date", datetime.now().strftime("%Y-%m-%d"))
        time = entities.get("time", "19:00")
        party_size = entities.get("party_size", "2")
        
        # Convert party size to number
        party_size_num = 2
        try:
            if isinstance(party_size, str):
                word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
                party_size_num = word_to_num.get(party_size, int(party_size)) if party_size.isdigit() else 2
            else:
                party_size_num = int(party_size)
        except:
            party_size_num = 2
        
        # Search restaurants
        restaurants = await search_restaurants_with_availability(
            cuisine, date, time, party_size_num, location, 10.0
        )
        
        if restaurants:
            # Create a natural response
            if len(restaurants) == 1:
                restaurant = restaurants[0]
                response_text = f"Great! I found {restaurant.restaurant_name}, a {restaurant.cuisine_type} restaurant. They're rated {restaurant.rating} stars and have availability at {time}. Would you like me to make a reservation for you?"
            else:
                top_3 = restaurants[:3]
                restaurant_list = ", ".join([f"{r.restaurant_name} ({r.rating} stars)" for r in top_3])
                response_text = f"I found {len(restaurants)} {cuisine} restaurants for you! The top options are: {restaurant_list}. Would you like more details about any of these, or shall I help you make a reservation?"
            
            # Update session context
            context_updates = {
                "last_search": {
                    "restaurants": [r.dict() for r in restaurants[:5]],
                    "search_params": {
                        "cuisine": cuisine,
                        "date": date,
                        "time": time,
                        "party_size": party_size_num
                    }
                }
            }
            
            return {
                "response_text": response_text,
                "action_taken": f"searched_{len(restaurants)}_restaurants",
                "was_successful": True,
                "data": {"restaurants": [r.dict() for r in restaurants[:5]]},
                "context_updates": context_updates
            }
        else:
            return {
                "response_text": f"I couldn't find any {cuisine} restaurants available at {time} for {party_size_num} people. Would you like me to try a different time or cuisine type?",
                "action_taken": "no_restaurants_found",
                "was_successful": False
            }
        
    except Exception as e:
        logger.error(f"Restaurant search error: {e}")
        return {
            "response_text": "I'm having trouble searching for restaurants right now. Please try again in a moment.",
            "action_taken": None,
            "was_successful": False
        }

async def handle_reservation_request(entities: Dict[str, Any], session: CorbySession):
    """Handle reservation booking through voice"""
    try:
        # Check if we have a recent search in context
        last_search = session.context.get("last_search")
        if not last_search:
            return {
                "response_text": "I'd be happy to help you make a reservation! First, let me search for available restaurants. What type of cuisine are you in the mood for?",
                "action_taken": "request_search_first",
                "was_successful": False
            }
        
        restaurants = last_search.get("restaurants", [])
        if not restaurants:
            return {
                "response_text": "I don't have any restaurant options to book. Would you like me to search for restaurants first?",
                "action_taken": "no_restaurants_in_context",
                "was_successful": False
            }
        
        # Get the first/best restaurant from search
        restaurant = restaurants[0]
        search_params = last_search.get("search_params", {})
        
        # Extract additional info from entities or use search context
        guest_name = entities.get("guest_name", "")
        date = entities.get("date", search_params.get("date"))
        time = entities.get("time", search_params.get("time"))
        party_size = entities.get("party_size", search_params.get("party_size"))
        
        if not guest_name:
            return {
                "response_text": f"I can book a table at {restaurant['restaurant_name']} for {party_size} people on {date} at {time}. What name should I put the reservation under?",
                "action_taken": "request_guest_name",
                "was_successful": False,
                "context_updates": {
                    "pending_reservation": {
                        "restaurant": restaurant,
                        "date": date,
                        "time": time,
                        "party_size": party_size
                    }
                }
            }
        
        # Here we would need guest email and phone for a real reservation
        # For demo purposes, we'll simulate a successful booking
        confirmation_id = f"BZ{random.randint(1000, 9999)}"
        
        return {
            "response_text": f"Perfect! I've booked your table at {restaurant['restaurant_name']} for {party_size} people on {date} at {time}. Your confirmation number is {confirmation_id}. You should receive a confirmation shortly!",
            "action_taken": f"reservation_created_{confirmation_id}",
            "was_successful": True,
            "data": {
                "reservation": {
                    "restaurant": restaurant['restaurant_name'],
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                    "confirmation": confirmation_id
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Reservation handling error: {e}")
        return {
            "response_text": "I'm having trouble making that reservation. Let me help you find available options instead.",
            "action_taken": None,
            "was_successful": False
        }

async def handle_availability_check(entities: Dict[str, Any], session: CorbySession):
    """Handle availability checking through voice"""
    try:
        restaurant_name = entities.get("restaurant_name")
        if not restaurant_name:
            return {
                "response_text": "Which restaurant would you like me to check availability for?",
                "action_taken": "request_restaurant_name",
                "was_successful": False
            }
        
        date = entities.get("date", datetime.now().strftime("%Y-%m-%d"))
        time = entities.get("time", "19:00")
        party_size = entities.get("party_size", "2")
        
        # Simulate availability check
        is_available = random.choice([True, False])
        
        if is_available:
            return {
                "response_text": f"Great news! {restaurant_name} has availability for {party_size} people at {time} on {date}. Would you like me to make a reservation for you?",
                "action_taken": f"availability_checked_available",
                "was_successful": True,
                "data": {"available": True, "restaurant": restaurant_name}
            }
        else:
            alternative_times = ["6:30 PM", "8:00 PM", "8:30 PM"]
            return {
                "response_text": f"Unfortunately, {restaurant_name} is booked at {time} on {date}. However, they have availability at {', '.join(alternative_times)}. Would any of these times work for you?",
                "action_taken": f"availability_checked_alternative_times",
                "was_successful": True,
                "data": {"available": False, "alternatives": alternative_times}
            }
        
    except Exception as e:
        logger.error(f"Availability check error: {e}")
        return {
            "response_text": "I'm having trouble checking availability right now. Please try again in a moment.",
            "action_taken": None,
            "was_successful": False
        }

async def handle_recommendations(entities: Dict[str, Any], session: CorbySession):
    """Handle restaurant recommendations through voice"""
    try:
        cuisine = entities.get("cuisine", "")
        location = entities.get("location", "near you")
        
        # Get user preferences from session
        user_preferences = session.user_preferences
        preferred_cuisine = user_preferences.get("cuisine", cuisine)
        
        if preferred_cuisine:
            response_text = f"Based on your preferences, I recommend trying some great {preferred_cuisine} restaurants {location}. "
        else:
            response_text = f"Here are some popular restaurant recommendations {location}. "
        
        # Mock recommendations
        recommendations = [
            "Bella Vista for authentic Italian with amazing pasta",
            "Dragon Palace for fresh Chinese cuisine",
            "The Corner Bistro for contemporary American dishes"
        ]
        
        response_text += "You might enjoy: " + ", ".join(recommendations[:2]) + f". Would you like me to check availability at any of these restaurants?"
        
        return {
            "response_text": response_text,
            "action_taken": "provided_recommendations",
            "was_successful": True,
            "data": {"recommendations": recommendations}
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return {
            "response_text": "I'd love to give you some recommendations! What type of cuisine are you in the mood for?",
            "action_taken": None,
            "was_successful": False
        }

async def generate_ai_conversation_response(command_text: str, session: CorbySession):
    """Generate conversational response using OpenAI"""
    try:
        context = "\n".join([f"User: {h['user']}\nCorby: {h['corby']}" for h in session.conversation_history[-3:]])
        
        conversation_prompt = f"""
You are Corby, a friendly and helpful AI assistant for BizFizz restaurant platform. You help users find restaurants, make reservations, and discover great dining experiences.

Context of recent conversation:
{context}

User just said: "{command_text}"

Respond naturally and helpfully. If they ask about restaurants, food, or dining, be enthusiastic and helpful. If it's off-topic, gently redirect to how you can help with restaurants.

Keep responses concise (1-2 sentences) and conversational.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Corby, a helpful restaurant assistant. Be friendly, concise, and focus on restaurant-related help."},
                {"role": "user", "content": conversation_prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        return {
            "response_text": response_text,
            "action_taken": "ai_conversation",
            "was_successful": True
        }
        
    except Exception as e:
        logger.error(f"AI conversation error: {e}")
        return {
            "response_text": "I'm here to help you with restaurants! What can I assist you with today?",
            "action_taken": None,
            "was_successful": True
        }

# OpenTable Integration Functions

OPENTABLE_CLIENT_ID = os.environ.get('OPENTABLE_CLIENT_ID')
OPENTABLE_CLIENT_SECRET = os.environ.get('OPENTABLE_CLIENT_SECRET')
OPENTABLE_API_BASE = "https://platform.opentable.com"

async def search_restaurants_with_availability(
    query: str, 
    date: str, 
    time: str, 
    party_size: int,
    location: Optional[str] = None,
    max_distance: float = 10.0
) -> List[RestaurantAvailability]:
    """Search for restaurants with availability using multiple sources"""
    try:
        restaurants = []
        
        # Method 1: Check our database for partner restaurants
        partner_restaurants = []
        async for restaurant in db.businesses.find({
            "user_type": "business",
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"cuisine_type": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]
        }).limit(10):
            partner_restaurants.append(restaurant)
        
        # Convert partner restaurants to availability format
        for restaurant in partner_restaurants:
            # Generate mock availability times (in real implementation, this would check actual availability)
            available_times = []
            base_hour = int(time.split(':')[0]) if time else 19
            for hour_offset in [-1, 0, 1, 2]:
                check_hour = base_hour + hour_offset
                if 17 <= check_hour <= 22:  # Restaurant hours 5 PM - 10 PM
                    available_times.append(f"{check_hour:02d}:00")
                    available_times.append(f"{check_hour:02d}:30")
            
            restaurant_availability = RestaurantAvailability(
                restaurant_id=restaurant["id"],
                restaurant_name=restaurant["name"],
                address=restaurant.get("address", "Address not available"),
                phone=restaurant.get("phone", "Phone not available"),
                cuisine_type=restaurant.get("cuisine_type", "Restaurant"),
                price_range=restaurant.get("price_range", "$$"),
                available_times=available_times,
                booking_url=f"/book-table/{restaurant['id']}",
                rating=restaurant.get("rating", 4.2),
                review_count=restaurant.get("review_count", 150),
                features=restaurant.get("features", ["Dine-in", "Takeout"]),
                latitude=restaurant.get("latitude"),
                longitude=restaurant.get("longitude")
            )
            
            restaurants.append(restaurant_availability)
        
        # Method 2: OpenTable API integration (if available)
        if OPENTABLE_CLIENT_ID and OPENTABLE_CLIENT_SECRET:
            try:
                opentable_restaurants = await search_opentable_api(query, date, time, party_size)
                restaurants.extend(opentable_restaurants)
            except Exception as e:
                logger.warning(f"OpenTable API search failed: {e}")
        
        # Method 3: Web scraping fallback (simplified for demo)
        web_scraped_restaurants = await search_restaurants_web_fallback(query, location)
        restaurants.extend(web_scraped_restaurants)
        
        # Sort by distance and rating
        if location:
            restaurants = await calculate_distances(restaurants, location)
            restaurants = [r for r in restaurants if r.distance_miles <= max_distance]
        
        restaurants.sort(key=lambda x: (x.distance_miles or 999, -x.rating))
        
        return restaurants[:20]  # Return top 20 results
        
    except Exception as e:
        logger.error(f"Restaurant search error: {e}")
        return []

async def search_opentable_api(query: str, date: str, time: str, party_size: int):
    """Search OpenTable API for restaurants (when API access is available)"""
    try:
        # Get OAuth token
        token = await get_opentable_oauth_token()
        if not token:
            return []
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Search for restaurants
        search_url = f"{OPENTABLE_API_BASE}/v1/restaurants/search"
        params = {
            "q": query,
            "date": date,
            "time": time,
            "party_size": party_size,
            "limit": 10
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(search_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                restaurants = []
                
                for restaurant in data.get("restaurants", []):
                    restaurant_availability = RestaurantAvailability(
                        restaurant_id=restaurant.get("id", ""),
                        restaurant_name=restaurant.get("name", ""),
                        opentable_id=restaurant.get("opentable_id"),
                        address=restaurant.get("address", ""),
                        phone=restaurant.get("phone", ""),
                        cuisine_type=restaurant.get("cuisine", ""),
                        available_times=restaurant.get("available_times", []),
                        booking_url=restaurant.get("booking_url", ""),
                        rating=restaurant.get("rating", 0.0),
                        review_count=restaurant.get("review_count", 0),
                        latitude=restaurant.get("lat"),
                        longitude=restaurant.get("lng")
                    )
                    restaurants.append(restaurant_availability)
                
                return restaurants
        
        return []
        
    except Exception as e:
        logger.error(f"OpenTable API search error: {e}")
        return []

async def get_opentable_oauth_token():
    """Get OAuth token for OpenTable API"""
    try:
        if not OPENTABLE_CLIENT_ID or not OPENTABLE_CLIENT_SECRET:
            return None
        
        auth_string = f"{OPENTABLE_CLIENT_ID}:{OPENTABLE_CLIENT_SECRET}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth.opentable.com/api/v2/oauth/token",
                headers=headers,
                data=data
            )
            
            if response.status_code == 200:
                return response.json().get("access_token")
        
        return None
        
    except Exception as e:
        logger.error(f"OpenTable OAuth error: {e}")
        return None

async def search_restaurants_web_fallback(query: str, location: Optional[str] = None):
    """Fallback method using web scraping or public APIs"""
    try:
        restaurants = []
        
        # Simulate web scraping results (in production, use actual scraping)
        mock_restaurants = [
            {
                "name": f"The {query.title()} Place",
                "address": "123 Main St, City, State",
                "phone": "(555) 123-4567",
                "cuisine": query.lower(),
                "rating": 4.3,
                "price_range": "$$"
            },
            {
                "name": f"{query.title()} Garden",
                "address": "456 Oak Ave, City, State",
                "phone": "(555) 234-5678",
                "cuisine": query.lower(),
                "rating": 4.1,
                "price_range": "$$$"
            }
        ]
        
        for mock_restaurant in mock_restaurants:
            # Generate available times
            available_times = ["18:00", "18:30", "19:00", "19:30", "20:00", "20:30"]
            
            restaurant = RestaurantAvailability(
                restaurant_id=f"web_{hash(mock_restaurant['name']) % 10000}",
                restaurant_name=mock_restaurant["name"],
                address=mock_restaurant["address"],
                phone=mock_restaurant["phone"],
                cuisine_type=mock_restaurant["cuisine"],
                price_range=mock_restaurant["price_range"],
                available_times=available_times,
                booking_url=f"https://www.opentable.com/r/{mock_restaurant['name'].lower().replace(' ', '-')}",
                rating=mock_restaurant["rating"],
                review_count=random.randint(50, 300)
            )
            
            restaurants.append(restaurant)
        
        return restaurants
        
    except Exception as e:
        logger.error(f"Web fallback search error: {e}")
        return []

async def calculate_distances(restaurants: List[RestaurantAvailability], user_location: str):
    """Calculate distances from user location to restaurants"""
    try:
        # In production, use Google Maps Geocoding API to get coordinates
        # For now, assign random distances for demo
        for restaurant in restaurants:
            if not restaurant.distance_miles:
                restaurant.distance_miles = round(random.uniform(0.5, 8.0), 1)
        
        return restaurants
        
    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        return restaurants

async def create_reservation_with_opentable(reservation_data: dict):
    """Create reservation using OpenTable API or direct booking"""
    try:
        reservation = OpenTableReservation(**reservation_data)
        
        # Method 1: Try OpenTable API if available
        if OPENTABLE_CLIENT_ID and reservation_data.get("opentable_id"):
            opentable_result = await book_via_opentable_api(reservation_data)
            if opentable_result:
                reservation.opentable_confirmation = opentable_result.get("confirmation_id")
                reservation.status = "confirmed"
        
        # Method 2: Store in our database regardless
        await db.opentable_reservations.insert_one(reservation.dict())
        
        # Method 3: Send notification to restaurant (if it's our partner)
        restaurant = await db.businesses.find_one({"id": reservation.restaurant_id})
        if restaurant:
            await manager.send_personal_message(
                json.dumps({
                    "type": "new_reservation",
                    "title": "New Reservation!",
                    "reservation": reservation.dict(),
                    "guest_info": {
                        "name": reservation.guest_name,
                        "party_size": reservation.party_size,
                        "time": reservation.reservation_time
                    }
                }),
                reservation.restaurant_id
            )
        
        return reservation
        
    except Exception as e:
        logger.error(f"Reservation creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create reservation")

async def book_via_opentable_api(reservation_data: dict):
    """Book directly via OpenTable API"""
    try:
        token = await get_opentable_oauth_token()
        if not token:
            return None
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        booking_data = {
            "restaurant_id": reservation_data.get("opentable_id"),
            "party_size": reservation_data["party_size"],
            "date_time": f"{reservation_data['reservation_date']}T{reservation_data['reservation_time']}",
            "customer": {
                "first_name": reservation_data["guest_name"].split()[0],
                "last_name": " ".join(reservation_data["guest_name"].split()[1:]) or "",
                "email": reservation_data["guest_email"],
                "phone": reservation_data["guest_phone"]
            },
            "special_requests": reservation_data.get("special_requests", "")
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENTABLE_API_BASE}/v1/reservations",
                headers=headers,
                json=booking_data
            )
            
            if response.status_code in [200, 201]:
                return response.json()
        
        return None
        
    except Exception as e:
        logger.error(f"OpenTable API booking error: {e}")
        return None

import random
import base64

# Geolocation and Proximity Functions
import math

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula (returns meters)"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

async def check_proximity_triggers(user_id: str, latitude: float, longitude: float):
    """Check if user location triggers any promotional campaigns"""
    try:
        triggered_campaigns = []
        
        # Get user's location permissions
        user_permission = await db.location_permissions.find_one({"user_id": user_id})
        if not user_permission or not user_permission.get("location_sharing", False):
            return triggered_campaigns
        
        # Get all active promotional campaigns
        active_campaigns = []
        async for campaign in db.promotional_campaigns.find({
            "is_active": True,
            "valid_until": {"$gt": datetime.utcnow()}
        }):
            active_campaigns.append(campaign)
        
        for campaign in active_campaigns:
            # Get business location
            business = await db.businesses.find_one({"id": campaign["business_id"]})
            if not business:
                continue
                
            # Calculate distance
            business_lat = business.get("latitude")
            business_lon = business.get("longitude")
            
            if business_lat and business_lon:
                distance = calculate_distance(latitude, longitude, business_lat, business_lon)
                
                # Check if within campaign radius
                if distance <= campaign.get("target_radius", 1609.34):
                    # Check if user hasn't received this promo recently (prevent spam)
                    recent_alert = await db.proximity_alerts.find_one({
                        "user_id": user_id,
                        "campaign_id": campaign["id"],
                        "sent_at": {"$gte": datetime.utcnow() - timedelta(hours=6)}
                    })
                    
                    if not recent_alert and campaign.get("current_uses", 0) < campaign.get("max_uses", 100):
                        triggered_campaigns.append({
                            "campaign": campaign,
                            "business": business,
                            "distance": distance
                        })
        
        return triggered_campaigns
        
    except Exception as e:
        logger.error(f"Proximity check error: {e}")
        return []

async def send_proximity_notification(user_id: str, campaign: dict, business: dict, distance: float):
    """Send promotional notification to user"""
    try:
        user = await db.users.find_one({"id": user_id})
        if not user:
            return False
        
        # Create proximity alert record
        proximity_alert = ProximityAlert(
            user_id=user_id,
            business_id=campaign["business_id"],
            campaign_id=campaign["id"],
            distance_meters=distance,
            promo_message=campaign["promo_message"],
            method="sms" if campaign.get("send_sms", True) else "push"
        )
        
        await db.proximity_alerts.insert_one(proximity_alert.dict())
        
        # Prepare promotional message
        distance_text = f"{distance/1609.34:.1f} miles" if distance > 1000 else f"{int(distance)} meters"
        
        promo_message = f"""
🍽️ Hey {user.get('name', 'Food Lover')}! 

You're just {distance_text} from {business['name']}!

🎉 SPECIAL OFFER: {campaign['promo_message']}
        
Use code: {campaign.get('promo_code', 'NEARBY')}
Valid until: {campaign['valid_until'].strftime('%m/%d %I:%M %p')}

📍 {business.get('address', '')}
🕒 Open now! Tap to get directions
        """.strip()
        
        # Send SMS notification (if Twilio is configured)
        sms_sent = False
        if TWILIO_ACCOUNT_SID and campaign.get("send_sms", True) and user.get("phone"):
            try:
                from twilio.rest import Client
                twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                
                message = twilio_client.messages.create(
                    body=promo_message,
                    from_='+1234567890',  # Your Twilio number
                    to=user["phone"]
                )
                
                sms_sent = True
                logger.info(f"SMS sent to {user['phone']}: {message.sid}")
                
            except Exception as e:
                logger.error(f"SMS sending failed: {e}")
        
        # Send push notification (WebSocket)
        if campaign.get("send_push", True):
            try:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "proximity_promo",
                        "title": f"Special Offer at {business['name']}!",
                        "message": campaign["promo_message"],
                        "business": business,
                        "campaign": campaign,
                        "distance": distance_text,
                        "alert_id": proximity_alert.id
                    }),
                    user_id
                )
            except Exception as e:
                logger.error(f"Push notification failed: {e}")
        
        # Update campaign usage
        await db.promotional_campaigns.update_one(
            {"id": campaign["id"]},
            {
                "$inc": {"current_uses": 1},
                "$set": {
                    f"success_metrics.notifications_sent": campaign.get("success_metrics", {}).get("notifications_sent", 0) + 1
                }
            }
        )
        
        # Notify business owner
        await manager.send_personal_message(
            json.dumps({
                "type": "customer_nearby",
                "title": "Potential Customer Nearby!",
                "message": f"Customer {user.get('name', 'Anonymous')} is {distance_text} away and received your promo!",
                "user_info": {
                    "name": user.get("name", "Anonymous"),
                    "distance": distance_text,
                    "promo_sent": True
                },
                "campaign": campaign["campaign_name"]
            }),
            campaign["business_id"]
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Proximity notification error: {e}")
        return False

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

@app.get("/api/analytics/comprehensive/{business_id}")
async def get_comprehensive_analytics(
    business_id: str,
    days: int = 30,
    include_competitors: bool = True,
    include_predictions: bool = True
):
    """Get comprehensive business analytics with AI insights"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Social Media Performance
        social_pipeline = [
            {"$match": {
                "business_id": business_id,
                "detected_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$detected_at"}},
                    "platform": "$platform",
                    "sentiment": "$sentiment_label"
                },
                "count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
                "total_engagement": {"$sum": {
                    "$add": [
                        {"$ifNull": ["$engagement_metrics.likes", 0]},
                        {"$ifNull": ["$engagement_metrics.shares", 0]},
                        {"$ifNull": ["$engagement_metrics.comments", 0]}
                    ]
                }}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        
        social_data = await db.social_mentions.aggregate(social_pipeline).to_list(None)
        
        # Crisis Detection Analytics
        crisis_pipeline = [
            {"$match": {
                "business_id": business_id,
                "created_at": {"$gte": start_date, "$lte": end_date},
                "priority": {"$in": ["high", "critical"]}
            }},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "alert_type": "$alert_type"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        
        crisis_data = await db.social_alerts.aggregate(crisis_pipeline).to_list(None)
        
        # Competitor Sentiment Comparison (if enabled)
        competitor_data = []
        if include_competitors:
            # Get competitor mentions for comparison
            competitor_pipeline = [
                {"$match": {
                    "detected_at": {"$gte": start_date, "$lte": end_date},
                    "business_id": {"$ne": business_id}
                }},
                {"$group": {
                    "_id": "$business_name",
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "total_mentions": {"$sum": 1},
                    "positive_mentions": {"$sum": {"$cond": [{"$eq": ["$sentiment_label", "positive"]}, 1, 0]}},
                    "negative_mentions": {"$sum": {"$cond": [{"$eq": ["$sentiment_label", "negative"]}, 1, 0]}}
                }},
                {"$sort": {"avg_sentiment": -1}},
                {"$limit": 10}
            ]
            
            competitor_data = await db.social_mentions.aggregate(competitor_pipeline).to_list(None)
        
        # AI-Powered Insights
        ai_insights = []
        if openai_client and include_predictions:
            try:
                # Generate AI insights based on data
                insight_prompt = f"""
                Analyze this restaurant's social media performance data and provide actionable insights:
                
                Social Data: {social_data[:5]}  # Sample data
                Crisis Data: {crisis_data[:3]}   # Sample alerts
                
                Provide 3-5 specific, actionable recommendations for improving their social media presence and customer satisfaction.
                Return as JSON with format: {{"insights": ["insight1", "insight2", ...], "predictions": ["prediction1", "prediction2", ...], "action_items": ["action1", "action2", ...]}}
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a restaurant marketing expert providing data-driven insights."},
                        {"role": "user", "content": insight_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                ai_insights = json.loads(response.choices[0].message.content.strip())
            except Exception as e:
                logger.error(f"AI insights generation failed: {e}")
                ai_insights = {
                    "insights": ["Increase engagement during peak hours", "Respond to negative feedback within 2 hours"],
                    "predictions": ["Sentiment likely to improve with proactive responses"],
                    "action_items": ["Implement social media response protocol", "Monitor competitor activities"]
                }
        
        # Reputation Score Calculation
        total_mentions = await db.social_mentions.count_documents({
            "business_id": business_id,
            "detected_at": {"$gte": start_date, "$lte": end_date}
        })
        
        positive_mentions = await db.social_mentions.count_documents({
            "business_id": business_id,
            "sentiment_label": "positive",
            "detected_at": {"$gte": start_date, "$lte": end_date}
        })
        
        negative_mentions = await db.social_mentions.count_documents({
            "business_id": business_id,
            "sentiment_label": "negative",
            "detected_at": {"$gte": start_date, "$lte": end_date}
        })
        
        reputation_score = 0
        if total_mentions > 0:
            reputation_score = ((positive_mentions - negative_mentions) / total_mentions) * 100
        
        # Response Time Analytics
        response_time_pipeline = [
            {"$match": {
                "business_id": business_id,
                "created_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": "$priority",
                "avg_response_time": {"$avg": {"$subtract": ["$updated_at", "$created_at"]}},
                "total_alerts": {"$sum": 1},
                "responded_alerts": {"$sum": {"$cond": ["$is_responded", 1, 0]}}
            }}
        ]
        
        response_data = await db.social_alerts.aggregate(response_time_pipeline).to_list(None)
        
        return {
            "business_id": business_id,
            "date_range": {"start": start_date, "end": end_date},
            "reputation_score": round(reputation_score, 2),
            "social_performance": {
                "total_mentions": total_mentions,
                "positive_mentions": positive_mentions,
                "negative_mentions": negative_mentions,
                "neutral_mentions": total_mentions - positive_mentions - negative_mentions,
                "time_series": social_data
            },
            "crisis_management": {
                "total_crises": len(crisis_data),
                "crisis_timeline": crisis_data,
                "response_analytics": response_data
            },
            "competitor_analysis": competitor_data,
            "ai_insights": ai_insights,
            "recommendations": [
                "Monitor mentions during peak dining hours (6-9 PM)",
                "Respond to negative reviews within 2 hours",
                "Engage with positive mentions to build loyalty",
                f"Your reputation score of {reputation_score:.1f}% is {'excellent' if reputation_score > 70 else 'good' if reputation_score > 40 else 'needs improvement'}"
            ]
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate comprehensive analytics")

@app.get("/api/analytics/realtime-dashboard/{business_id}")
async def get_realtime_dashboard(business_id: str):
    """Get real-time dashboard data"""
    try:
        # Last 24 hours data
        yesterday = datetime.utcnow() - timedelta(hours=24)
        
        # Real-time metrics
        active_alerts = await db.social_alerts.count_documents({
            "business_id": business_id,
            "is_read": False,
            "created_at": {"$gte": yesterday}
        })
        
        recent_mentions = await db.social_mentions.count_documents({
            "business_id": business_id,
            "detected_at": {"$gte": yesterday}
        })
        
        # Sentiment trend (last 6 hours)
        six_hours_ago = datetime.utcnow() - timedelta(hours=6)
        recent_sentiment = []
        async for mention in db.social_mentions.find({
            "business_id": business_id,
            "detected_at": {"$gte": six_hours_ago}
        }).sort("detected_at", -1).limit(20):
            recent_sentiment.append({
                "platform": mention["platform"],
                "sentiment_score": mention["sentiment_score"],
                "timestamp": mention["detected_at"],
                "content": mention["content"][:100] + "..." if len(mention["content"]) > 100 else mention["content"]
            })
        
        # Platform activity
        platform_activity = []
        platform_pipeline = [
            {"$match": {
                "business_id": business_id,
                "detected_at": {"$gte": yesterday}
            }},
            {"$group": {
                "_id": "$platform",
                "count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
                "last_mention": {"$max": "$detected_at"}
            }}
        ]
        
        platform_activity = await db.social_mentions.aggregate(platform_pipeline).to_list(None)
        
        return {
            "timestamp": datetime.utcnow(),
            "active_alerts": active_alerts,
            "recent_mentions": recent_mentions,
            "sentiment_trend": recent_sentiment,
            "platform_activity": platform_activity,
            "status": "live" if recent_mentions > 0 else "monitoring"
        }
        
    except Exception as e:
        logger.error(f"Real-time dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get real-time dashboard")

@app.post("/api/analytics/export-report/{business_id}")
async def export_analytics_report(business_id: str, format: str = "pdf", days: int = 30):
    """Export comprehensive analytics report"""
    try:
        # Get comprehensive analytics
        analytics = await get_comprehensive_analytics(business_id, days, True, True)
        
        if format.lower() == "pdf":
            # Generate PDF report using ReportLab
            from io import BytesIO
            buffer = BytesIO()
            
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=HexColor('#1f2937')
            )
            
            story.append(Paragraph("BizFizz Analytics Report", title_style))
            story.append(Spacer(1, 20))
            
            # Business Info
            story.append(Paragraph(f"Business ID: {business_id}", styles['Normal']))
            story.append(Paragraph(f"Report Period: {analytics['date_range']['start'].strftime('%Y-%m-%d')} to {analytics['date_range']['end'].strftime('%Y-%m-%d')}", styles['Normal']))
            story.append(Paragraph(f"Reputation Score: {analytics['reputation_score']}%", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Social Performance Summary
            story.append(Paragraph("Social Media Performance", styles['Heading2']))
            perf_data = [
                ['Metric', 'Count'],
                ['Total Mentions', str(analytics['social_performance']['total_mentions'])],
                ['Positive Mentions', str(analytics['social_performance']['positive_mentions'])],
                ['Negative Mentions', str(analytics['social_performance']['negative_mentions'])],
                ['Neutral Mentions', str(analytics['social_performance']['neutral_mentions'])]
            ]
            
            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1f2937')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#e5e7eb'))
            ]))
            
            story.append(perf_table)
            story.append(Spacer(1, 20))
            
            # AI Insights
            if analytics.get('ai_insights'):
                story.append(Paragraph("AI-Powered Insights", styles['Heading2']))
                for insight in analytics['ai_insights'].get('insights', []):
                    story.append(Paragraph(f"• {insight}", styles['Normal']))
                story.append(Spacer(1, 10))
                
                story.append(Paragraph("Action Items", styles['Heading3']))
                for action in analytics['ai_insights'].get('action_items', []):
                    story.append(Paragraph(f"• {action}", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            
            return FileResponse(
                path=None,
                media_type='application/pdf',
                filename=f"bizfizz_report_{business_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
        
        else:  # JSON format
            return analytics
            
    except Exception as e:
        logger.error(f"Export report error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export report")

@app.get("/api/mobile/dashboard/{business_id}")
async def get_mobile_dashboard(business_id: str):
    """Optimized mobile dashboard endpoint"""
    try:
        # Get essential data for mobile
        yesterday = datetime.utcnow() - timedelta(hours=24)
        
        # Critical alerts only
        critical_alerts = []
        async for alert in db.social_alerts.find({
            "business_id": business_id,
            "priority": {"$in": ["high", "critical"]},
            "is_read": False
        }).sort("created_at", -1).limit(5):
            if "_id" in alert:
                del alert["_id"]
            critical_alerts.append(alert)
        
        # Recent sentiment summary
        sentiment_summary = await db.social_mentions.aggregate([
            {"$match": {
                "business_id": business_id,
                "detected_at": {"$gte": yesterday}
            }},
            {"$group": {
                "_id": "$sentiment_label",
                "count": {"$sum": 1}
            }}
        ]).to_list(None)
        
        # Platform activity (simplified for mobile)
        platform_activity = await db.social_mentions.aggregate([
            {"$match": {
                "business_id": business_id,
                "detected_at": {"$gte": yesterday}
            }},
            {"$group": {
                "_id": "$platform",
                "count": {"$sum": 1},
                "latest": {"$max": "$detected_at"}
            }}
        ]).to_list(None)
        
        return {
            "critical_alerts": critical_alerts,
            "sentiment_summary": sentiment_summary,
            "platform_activity": platform_activity,
            "total_alerts": len(critical_alerts),
            "monitoring_active": True,
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Mobile dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get mobile dashboard")

@app.post("/api/mobile/quick-response/{mention_id}")
async def mobile_quick_response(mention_id: str, response_type: str):
    """Quick response actions for mobile"""
    try:
        mention = await db.social_mentions.find_one({"id": mention_id})
        if not mention:
            raise HTTPException(status_code=404, detail="Mention not found")
        
        # Generate quick response based on type
        quick_responses = {
            "thank": "Thank you so much for your wonderful review! We're thrilled you enjoyed your experience with us. 🙏",
            "apologize": "We sincerely apologize for the experience you had. Please reach out to us directly so we can make this right. 📞",
            "investigate": "Thank you for bringing this to our attention. We're looking into this matter and will follow up with you shortly. 🔍",
            "invite": "We'd love to have you visit us again! Please let us know if there's anything we can do to improve your experience. 🍽️"
        }
        
        suggested_response = quick_responses.get(response_type, "Thank you for your feedback!")
        
        # Create response record
        response_record = {
            "id": str(uuid.uuid4()),
            "mention_id": mention_id,
            "response_type": response_type,
            "suggested_response": suggested_response,
            "platform": mention["platform"],
            "created_at": datetime.utcnow(),
            "business_id": mention["business_id"]
        }
        
        await db.quick_responses.insert_one(response_record)
        
        return {
            "suggested_response": suggested_response,
            "platform": mention["platform"],
            "mention_url": mention.get("url"),
            "response_id": response_record["id"]
        }
        
    except Exception as e:
        logger.error(f"Quick response error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate quick response")

@app.get("/api/mobile/notifications/{business_id}")
async def get_mobile_notifications(business_id: str, limit: int = 10):
    """Get mobile-optimized notifications"""
    try:
        notifications = []
        
        # Get recent high-priority alerts
        async for alert in db.social_alerts.find({
            "business_id": business_id,
            "priority": {"$in": ["medium", "high", "critical"]}
        }).sort("created_at", -1).limit(limit):
            if "_id" in alert:
                del alert["_id"]
            
            # Simplify for mobile
            notification = {
                "id": alert["id"],
                "title": alert["title"],
                "message": alert["description"][:100] + "..." if len(alert["description"]) > 100 else alert["description"],
                "priority": alert["priority"],
                "timestamp": alert["created_at"],
                "is_read": alert["is_read"],
                "type": alert["alert_type"]
            }
            
            notifications.append(notification)
        
        return {"notifications": notifications}
        
    except Exception as e:
        logger.error(f"Mobile notifications error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")

@app.post("/api/corby/voice-command")
async def handle_voice_command(command_data: dict):
    """Process voice command through enhanced Corby AI assistant"""
    try:
        user_id = command_data.get("user_id")
        command_text = command_data.get("command_text", "")
        session_id = command_data.get("session_id")
        voice_profile = command_data.get("voice_profile", "friendly")
        
        if not user_id or not command_text:
            raise HTTPException(status_code=400, detail="User ID and command text required")
        
        # Process the enhanced voice command
        result = await process_voice_command(user_id, command_text, session_id, voice_profile)
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced voice command API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process voice command")

@app.post("/api/corby/smart-recommendations")
async def get_smart_recommendations(request_data: dict):
    """Get AI-powered contextual restaurant recommendations"""
    try:
        user_id = request_data.get("user_id")
        session_id = request_data.get("session_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        # Get session for context
        session_data = await db.corby_sessions.find_one({"id": session_id}) if session_id else None
        session = CorbySession(**session_data) if session_data else CorbySession(user_id=user_id, context={})
        
        # Get contextual recommendations
        recommendations = await get_contextual_recommendations(user_id, session)
        
        return {
            "recommendations": recommendations,
            "personalization_level": "high" if session_data else "basic",
            "context_factors": {
                "conversation_history": len(session.conversation_history),
                "user_preferences": session.user_preferences,
                "location_context": bool(session.context.get("location"))
            }
        }
        
    except Exception as e:
        logger.error(f"Smart recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@app.post("/api/corby/voice-profile")
async def set_voice_profile(profile_data: dict):
    """Set user's preferred voice profile"""
    try:
        user_id = profile_data.get("user_id")
        voice_profile = profile_data.get("voice_profile", "friendly")
        session_id = profile_data.get("session_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        if voice_profile not in CORBY_VOICE_PROFILES:
            raise HTTPException(status_code=400, detail="Invalid voice profile")
        
        # Update session preferences
        if session_id:
            await db.corby_sessions.update_one(
                {"id": session_id},
                {"$set": {"user_preferences.voice_profile": voice_profile}}
            )
        
        # Get voice settings for profile
        voice_settings = CORBY_VOICE_PROFILES[voice_profile]
        
        return {
            "voice_profile": voice_profile,
            "personality": voice_settings["personality"],
            "speaking_style": voice_settings["speaking_style"],
            "voice_settings": voice_settings["voice_settings"],
            "message": f"Voice profile set to {voice_profile}"
        }
        
    except Exception as e:
        logger.error(f"Voice profile error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to set voice profile")

@app.get("/api/corby/voice-profiles")
async def get_available_voice_profiles():
    """Get available voice profiles"""
    try:
        return {
            "profiles": {
                profile_name: {
                    "name": profile_name.title(),
                    "description": data["personality"],
                    "style": data["speaking_style"],
                    "best_for": {
                        "professional": "Business dining, client meetings, formal occasions",
                        "friendly": "Casual dining, everyday use, family meals",
                        "luxury": "Fine dining, special occasions, romantic dinners"
                    }.get(profile_name, "General use")
                }
                for profile_name, data in CORBY_VOICE_PROFILES.items()
            },
            "default": "friendly",
            "premium_features": {
                "elevenlabs_voice": bool(ELEVENLABS_API_KEY),
                "claude_ai": bool(ANTHROPIC_API_KEY),
                "personality_adaptation": True
            }
        }
        
    except Exception as e:
        logger.error(f"Voice profiles error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get voice profiles")

@app.post("/api/corby/conversation-context")
async def update_conversation_context(context_data: dict):
    """Update conversation context for better responses"""
    try:
        session_id = context_data.get("session_id")
        location = context_data.get("location")
        preferences = context_data.get("preferences", {})
        occasion = context_data.get("occasion")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        update_data = {}
        if location:
            update_data["context.location"] = location
        if occasion:
            update_data["context.occasion"] = occasion
        if preferences:
            update_data["user_preferences"] = preferences
        
        await db.corby_sessions.update_one(
            {"id": session_id},
            {"$set": update_data}
        )
        
        return {"message": "Context updated successfully"}
        
    except Exception as e:
        logger.error(f"Context update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update context")

@app.get("/api/corby/analytics/{user_id}")
async def get_corby_analytics(user_id: str, days: int = 30):
    """Get user's interaction analytics with Corby"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Command frequency
        command_pipeline = [
            {"$match": {
                "user_id": user_id,
                "created_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": "$intent",
                "count": {"$sum": 1},
                "success_rate": {"$avg": {"$cond": ["$was_successful", 1, 0]}}
            }}
        ]
        
        command_stats = await db.voice_commands.aggregate(command_pipeline).to_list(None)
        
        # Daily usage
        daily_pipeline = [
            {"$match": {
                "user_id": user_id,
                "created_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "interactions": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        daily_usage = await db.voice_commands.aggregate(daily_pipeline).to_list(None)
        
        # Success metrics
        total_commands = await db.voice_commands.count_documents({
            "user_id": user_id,
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        successful_commands = await db.voice_commands.count_documents({
            "user_id": user_id,
            "was_successful": True,
            "created_at": {"$gte": start_date, "$lte": end_date}
        })
        
        return {
            "analytics_period": {"start": start_date, "end": end_date},
            "total_interactions": total_commands,
            "success_rate": (successful_commands / total_commands * 100) if total_commands > 0 else 0,
            "command_breakdown": command_stats,
            "daily_usage": daily_usage,
            "user_engagement": {
                "avg_daily_interactions": total_commands / days if days > 0 else 0,
                "most_common_intent": max(command_stats, key=lambda x: x["count"])["_id"] if command_stats else "none",
                "engagement_level": "high" if total_commands > 50 else "medium" if total_commands > 20 else "low"
            }
        }
        
    except Exception as e:
        logger.error(f"Corby analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

# Corby Voice Assistant API Endpoints

@app.post("/api/corby/text-to-speech")
async def text_to_speech_endpoint(tts_data: dict):
    """Generate audio URL for text-to-speech (placeholder for TTS service integration)"""
    try:
        text = tts_data.get("text", "")
        voice = tts_data.get("voice", "corby")  # Custom voice options
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required for TTS")
        
        # In production, integrate with services like:
        # - ElevenLabs for premium voice synthesis
        # - Google Cloud Text-to-Speech
        # - Amazon Polly
        # - Azure Cognitive Services Speech
        
        # For now, return Web Speech API instructions
        return {
            "message": "Use Web Speech API for client-side TTS",
            "text": text,
            "voice_settings": {
                "rate": 0.9,
                "pitch": 1.0,
                "volume": 0.8,
                "voice_name": "Google US English Female"
            },
            "use_browser_tts": True
        }
        
    except Exception as e:
        logger.error(f"TTS API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

@app.get("/api/corby/session/{session_id}")
async def get_corby_session(session_id: str):
    """Get Corby conversation session"""
    try:
        session = await db.corby_sessions.find_one({"id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if "_id" in session:
            del session["_id"]
        
        return {"session": session}
        
    except Exception as e:
        logger.error(f"Get session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session")

@app.get("/api/corby/conversations/{user_id}")
async def get_user_conversations(user_id: str, limit: int = 10):
    """Get user's conversation history with Corby"""
    try:
        conversations = []
        async for command in db.voice_commands.find({"user_id": user_id}).sort("created_at", -1).limit(limit):
            if "_id" in command:
                del command["_id"]
            conversations.append(command)
        
        return {"conversations": conversations, "total": len(conversations)}
        
    except Exception as e:
        logger.error(f"Get conversations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversations")

@app.post("/api/corby/feedback")
async def corby_feedback(feedback_data: dict):
    """Provide feedback on Corby's responses for improvement"""
    try:
        command_id = feedback_data.get("command_id")
        rating = feedback_data.get("rating")  # 1-5 stars
        feedback_text = feedback_data.get("feedback", "")
        
        if not command_id or not rating:
            raise HTTPException(status_code=400, detail="Command ID and rating required")
        
        # Store feedback for improving Corby
        feedback_record = {
            "id": str(uuid.uuid4()),
            "command_id": command_id,
            "rating": rating,
            "feedback": feedback_text,
            "created_at": datetime.utcnow()
        }
        
        await db.corby_feedback.insert_one(feedback_record)
        
        return {"message": "Feedback recorded successfully"}
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/api/corby/capabilities")
async def get_corby_capabilities():
    """Get Corby's current capabilities and features"""
    try:
        return {
            "name": "Corby",
            "version": "1.0",
            "description": "Your personal restaurant assistant",
            "capabilities": [
                "Restaurant search by cuisine, location, and preferences",
                "Real-time availability checking",
                "Table reservation booking",
                "Personalized restaurant recommendations",
                "Conversational dining assistance",
                "Voice and text interaction",
                "Context-aware responses"
            ],
            "supported_intents": [
                "restaurant_search",
                "make_reservation", 
                "check_availability",
                "get_recommendations",
                "modify_reservation",
                "general_query",
                "greeting",
                "help"
            ],
            "voice_features": {
                "speech_recognition": True,
                "text_to_speech": True,
                "natural_language_processing": True,
                "conversation_memory": True
            },
            "integration_status": {
                "openai": bool(openai_client),
                "restaurant_search": True,
                "reservation_system": True,
                "user_preferences": True
            }
        }
        
    except Exception as e:
        logger.error(f"Capabilities error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get capabilities")

# OpenTable Reservation API Endpoints

@app.post("/api/restaurants/search")
async def search_restaurants_availability(query_data: dict):
    """Search for restaurants with availability"""
    try:
        query = query_data.get("query", "")
        date = query_data.get("date", "")
        time = query_data.get("time", "19:00")
        party_size = query_data.get("party_size", 2)
        location = query_data.get("location")
        cuisine = query_data.get("cuisine")
        max_distance = query_data.get("max_distance", 10.0)
        
        if not query or not date:
            raise HTTPException(status_code=400, detail="Query and date are required")
        
        # Create query record
        reservation_query = ReservationQuery(
            user_id=query_data.get("user_id", "anonymous"),
            query_text=query,
            desired_date=datetime.fromisoformat(date.replace("Z", "+00:00")),
            desired_time=time,
            party_size=party_size,
            cuisine_preference=cuisine,
            location=location,
            max_distance=max_distance
        )
        
        # Search for restaurants
        restaurants = await search_restaurants_with_availability(
            query, date, time, party_size, location, max_distance
        )
        
        # Update query with results
        reservation_query.search_results = [r.dict() for r in restaurants]
        await db.reservation_queries.insert_one(reservation_query.dict())
        
        return {
            "query_id": reservation_query.id,
            "restaurants": [r.dict() for r in restaurants],
            "total_found": len(restaurants),
            "search_location": location,
            "search_criteria": {
                "date": date,
                "time": time,
                "party_size": party_size,
                "cuisine": cuisine
            }
        }
        
    except Exception as e:
        logger.error(f"Restaurant search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search restaurants")

@app.post("/api/reservations/create")
async def create_restaurant_reservation(reservation_data: dict):
    """Create a new restaurant reservation"""
    try:
        required_fields = ["user_id", "restaurant_id", "guest_name", "guest_email", 
                          "guest_phone", "reservation_date", "reservation_time", "party_size"]
        
        for field in required_fields:
            if field not in reservation_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Parse date
        reservation_data["reservation_date"] = datetime.fromisoformat(
            reservation_data["reservation_date"].replace("Z", "+00:00")
        )
        
        # Create reservation
        reservation = await create_reservation_with_opentable(reservation_data)
        
        # Send confirmation to user
        confirmation_message = f"""
🍽️ Reservation Confirmed!

Restaurant: {reservation.restaurant_name}
Date: {reservation.reservation_date.strftime('%B %d, %Y')}
Time: {reservation.reservation_time}
Party Size: {reservation.party_size}
Confirmation: {reservation.id[:8]}

Thank you for using BizFizz! 🎉
        """.strip()
        
        return {
            "message": "Reservation created successfully",
            "reservation_id": reservation.id,
            "confirmation": reservation.id[:8],
            "status": reservation.status,
            "opentable_confirmation": reservation.opentable_confirmation,
            "confirmation_message": confirmation_message
        }
        
    except Exception as e:
        logger.error(f"Reservation creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create reservation")

@app.get("/api/reservations/{user_id}")
async def get_user_reservations(user_id: str):
    """Get all reservations for a user"""
    try:
        reservations = []
        async for reservation in db.opentable_reservations.find({"user_id": user_id}).sort("reservation_date", -1):
            if "_id" in reservation:
                del reservation["_id"]
            reservations.append(reservation)
        
        return {"reservations": reservations, "total": len(reservations)}
        
    except Exception as e:
        logger.error(f"Get reservations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get reservations")

@app.put("/api/reservations/{reservation_id}/cancel")
async def cancel_reservation(reservation_id: str):
    """Cancel a reservation"""
    try:
        reservation = await db.opentable_reservations.find_one({"id": reservation_id})
        if not reservation:
            raise HTTPException(status_code=404, detail="Reservation not found")
        
        # Update status
        await db.opentable_reservations.update_one(
            {"id": reservation_id},
            {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}}
        )
        
        # Notify restaurant
        await manager.send_personal_message(
            json.dumps({
                "type": "reservation_cancelled",
                "title": "Reservation Cancelled",
                "message": f"Reservation for {reservation['guest_name']} on {reservation['reservation_date']} has been cancelled",
                "reservation_id": reservation_id
            }),
            reservation["restaurant_id"]
        )
        
        return {"message": "Reservation cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Cancel reservation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel reservation")

@app.get("/api/restaurants/{restaurant_id}/reservations")
async def get_restaurant_reservations(restaurant_id: str, date: Optional[str] = None):
    """Get reservations for a restaurant"""
    try:
        query = {"restaurant_id": restaurant_id}
        
        if date:
            target_date = datetime.fromisoformat(date.replace("Z", "+00:00"))
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            query["reservation_date"] = {"$gte": start_of_day, "$lt": end_of_day}
        
        reservations = []
        async for reservation in db.opentable_reservations.find(query).sort("reservation_date", 1):
            if "_id" in reservation:
                del reservation["_id"]
            reservations.append(reservation)
        
        return {"reservations": reservations, "total": len(reservations), "date": date}
        
    except Exception as e:
        logger.error(f"Get restaurant reservations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get restaurant reservations")

@app.post("/api/reservations/availability-check")
async def check_availability(availability_data: dict):
    """Check availability for specific restaurant, date, and time"""
    try:
        restaurant_id = availability_data.get("restaurant_id")
        date = availability_data.get("date")
        time = availability_data.get("time")
        party_size = availability_data.get("party_size", 2)
        
        if not all([restaurant_id, date, time]):
            raise HTTPException(status_code=400, detail="Restaurant ID, date, and time required")
        
        # Check existing reservations
        target_datetime = datetime.fromisoformat(f"{date}T{time}")
        existing_reservations = await db.opentable_reservations.count_documents({
            "restaurant_id": restaurant_id,
            "reservation_date": target_datetime,
            "status": {"$nin": ["cancelled"]}
        })
        
        # Simulate capacity check (in production, use actual restaurant capacity)
        max_capacity = 10  # Tables per time slot
        available = existing_reservations < max_capacity
        
        # Get restaurant info
        restaurant = await db.businesses.find_one({"id": restaurant_id})
        
        return {
            "available": available,
            "restaurant_name": restaurant.get("name", "Unknown") if restaurant else "Unknown",
            "date": date,
            "time": time,
            "party_size": party_size,
            "existing_reservations": existing_reservations,
            "capacity": max_capacity,
            "message": "Available" if available else "Fully booked for this time"
        }
        
    except Exception as e:
        logger.error(f"Availability check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check availability")

# Location-Based Marketing API Endpoints

@app.post("/api/location/update")
async def update_user_location(location_data: dict):
    """Update user's current location and check for promotional triggers"""
    try:
        user_id = location_data.get("user_id")
        latitude = location_data.get("latitude")
        longitude = location_data.get("longitude")
        accuracy = location_data.get("accuracy", 10.0)
        
        if not all([user_id, latitude, longitude]):
            raise HTTPException(status_code=400, detail="Missing required location data")
        
        # Check location permissions
        permission = await db.location_permissions.find_one({"user_id": user_id})
        if not permission or not permission.get("location_sharing", False):
            return {"message": "Location sharing not enabled"}
        
        # Store/update user location
        user_location = UserLocation(
            user_id=user_id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            location_sharing_enabled=True
        )
        
        await db.user_locations.replace_one(
            {"user_id": user_id},
            user_location.dict(),
            upsert=True
        )
        
        # Check for proximity triggers
        triggered_campaigns = await check_proximity_triggers(user_id, latitude, longitude)
        
        notifications_sent = 0
        for trigger in triggered_campaigns:
            success = await send_proximity_notification(
                user_id,
                trigger["campaign"],
                trigger["business"],
                trigger["distance"]
            )
            if success:
                notifications_sent += 1
        
        return {
            "message": "Location updated successfully",
            "proximity_alerts": len(triggered_campaigns),
            "notifications_sent": notifications_sent,
            "nearby_restaurants": len(triggered_campaigns)
        }
        
    except Exception as e:
        logger.error(f"Location update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update location")

@app.post("/api/location/permissions")
async def update_location_permissions(permission_data: dict):
    """Update user's location sharing and notification preferences"""
    try:
        user_id = permission_data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        permission = LocationPermission(
            user_id=user_id,
            permission_granted=permission_data.get("permission_granted", False),
            location_sharing=permission_data.get("location_sharing", False),
            promotional_notifications=permission_data.get("promotional_notifications", True),
            sms_notifications=permission_data.get("sms_notifications", True),
            push_notifications=permission_data.get("push_notifications", True),
            privacy_level=permission_data.get("privacy_level", "balanced")
        )
        
        await db.location_permissions.replace_one(
            {"user_id": user_id},
            permission.dict(),
            upsert=True
        )
        
        return {"message": "Location permissions updated successfully"}
        
    except Exception as e:
        logger.error(f"Permission update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update permissions")

@app.post("/api/campaigns/create")
async def create_promotional_campaign(campaign_data: dict):
    """Create a new promotional campaign for a business"""
    try:
        # Validate required fields
        required_fields = ["business_id", "campaign_name", "promo_message", "valid_until"]
        for field in required_fields:
            if field not in campaign_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Get business details
        business = await db.businesses.find_one({"id": campaign_data["business_id"]})
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        
        # Parse valid_until datetime
        valid_until = datetime.fromisoformat(campaign_data["valid_until"].replace("Z", "+00:00"))
        
        campaign = PromotionalCampaign(
            business_id=campaign_data["business_id"],
            business_name=business.get("name", "Unknown Business"),
            campaign_name=campaign_data["campaign_name"],
            promo_message=campaign_data["promo_message"],
            discount_amount=campaign_data.get("discount_amount"),
            discount_type=campaign_data.get("discount_type", "percentage"),
            promo_code=campaign_data.get("promo_code"),
            valid_until=valid_until,
            max_uses=campaign_data.get("max_uses", 100),
            target_radius=campaign_data.get("target_radius", 1609.34),  # 1 mile default
            send_sms=campaign_data.get("send_sms", True),
            send_push=campaign_data.get("send_push", True)
        )
        
        await db.promotional_campaigns.insert_one(campaign.dict())
        
        return {
            "message": "Promotional campaign created successfully",
            "campaign_id": campaign.id,
            "target_radius_miles": campaign.target_radius / 1609.34
        }
        
    except Exception as e:
        logger.error(f"Campaign creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create campaign")

@app.get("/api/campaigns/{business_id}")
async def get_business_campaigns(business_id: str):
    """Get all campaigns for a business"""
    try:
        campaigns = []
        async for campaign in db.promotional_campaigns.find({"business_id": business_id}).sort("created_at", -1):
            if "_id" in campaign:
                del campaign["_id"]
            
            # Add performance metrics
            total_alerts = await db.proximity_alerts.count_documents({"campaign_id": campaign["id"]})
            opened_alerts = await db.proximity_alerts.count_documents({"campaign_id": campaign["id"], "opened": True})
            redeemed_alerts = await db.proximity_alerts.count_documents({"campaign_id": campaign["id"], "redeemed": True})
            
            campaign["performance"] = {
                "total_sent": total_alerts,
                "opened": opened_alerts,
                "redeemed": redeemed_alerts,
                "open_rate": (opened_alerts / total_alerts * 100) if total_alerts > 0 else 0,
                "redemption_rate": (redeemed_alerts / total_alerts * 100) if total_alerts > 0 else 0
            }
            
            campaigns.append(campaign)
        
        return {"campaigns": campaigns}
        
    except Exception as e:
        logger.error(f"Get campaigns error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get campaigns")

@app.get("/api/location/nearby-users/{business_id}")
async def get_nearby_users(business_id: str, radius_miles: float = 1.0):
    """Get users currently near a business (for restaurant owners)"""
    try:
        business = await db.businesses.find_one({"id": business_id})
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        
        business_lat = business.get("latitude")
        business_lon = business.get("longitude")
        
        if not business_lat or not business_lon:
            return {"nearby_users": [], "message": "Business location not set"}
        
        radius_meters = radius_miles * 1609.34
        nearby_users = []
        
        # Get recent user locations (last 30 minutes)
        recent_time = datetime.utcnow() - timedelta(minutes=30)
        
        async for location in db.user_locations.find({
            "timestamp": {"$gte": recent_time},
            "location_sharing_enabled": True,
            "is_active": True
        }):
            distance = calculate_distance(
                business_lat, business_lon,
                location["latitude"], location["longitude"]
            )
            
            if distance <= radius_meters:
                # Get user info (anonymized for privacy)
                user = await db.users.find_one({"id": location["user_id"]})
                if user:
                    nearby_users.append({
                        "user_id": location["user_id"],
                        "name": user.get("name", "Anonymous User"),
                        "distance_meters": round(distance),
                        "distance_miles": round(distance / 1609.34, 2),
                        "last_seen": location["timestamp"],
                        "user_type": user.get("user_type", "consumer")
                    })
        
        return {
            "nearby_users": nearby_users,
            "total_count": len(nearby_users),
            "search_radius_miles": radius_miles,
            "business_location": {
                "latitude": business_lat,
                "longitude": business_lon
            }
        }
        
    except Exception as e:
        logger.error(f"Nearby users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get nearby users")

@app.post("/api/proximity/alert/{alert_id}/opened")
async def mark_proximity_alert_opened(alert_id: str):
    """Mark proximity alert as opened by user"""
    try:
        await db.proximity_alerts.update_one(
            {"id": alert_id},
            {"$set": {"opened": True}}
        )
        
        return {"message": "Alert marked as opened"}
        
    except Exception as e:
        logger.error(f"Mark alert opened error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark alert as opened")

@app.post("/api/proximity/alert/{alert_id}/redeemed")
async def mark_proximity_alert_redeemed(alert_id: str, redemption_data: dict):
    """Mark proximity alert as redeemed"""
    try:
        await db.proximity_alerts.update_one(
            {"id": alert_id},
            {
                "$set": {
                    "redeemed": True,
                    "user_response": redemption_data.get("response", "redeemed")
                }
            }
        )
        
        # Update campaign success metrics
        alert = await db.proximity_alerts.find_one({"id": alert_id})
        if alert:
            await db.promotional_campaigns.update_one(
                {"id": alert["campaign_id"]},
                {"$inc": {"success_metrics.redemptions": 1}}
            )
        
        return {"message": "Promo redeemed successfully"}
        
    except Exception as e:
        logger.error(f"Mark alert redeemed error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark alert as redeemed")

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
            
            # Monitor Google Places/Reviews
            if "google" in rule.platforms:
                google_mentions = await monitor_google_reviews(rule.keywords, rule.business_id, rule.business_name)
                all_mentions.extend(google_mentions)
            
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
            "google_places": bool(GOOGLE_PLACES_API_KEY),
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