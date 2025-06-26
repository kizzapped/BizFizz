import requests
import unittest
import json
import sys
import uuid
from datetime import datetime, timedelta

class BizFizzUltimateAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BizFizzUltimateAPITester, self).__init__(*args, **kwargs)
        # Get the backend URL from the frontend .env file
        self.base_url = "https://090f1bbb-1ae7-49b3-b27a-6fd8915a0f58.preview.emergentagent.com"
        self.tests_run = 0
        self.tests_passed = 0
        self.competitor_ids = []
        self.location = "Times Square, New York"
        self.report_id = None
        self.api_integrations = {}
        self.user_id = None
        self.business_id = None
        self.test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        self.test_password = "TestPassword123!"
        self.test_business_name = f"Test Restaurant {uuid.uuid4().hex[:8]}"

    def setUp(self):
        self.tests_run += 1

    def test_01_health_check(self):
        """Test the health check endpoint"""
        print(f"\n🔍 Testing health check endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "BizFizz Ultimate Platform")
            
            # Check API integrations
            self.assertIn("integrations", data)
            self.assertIsInstance(data["integrations"], dict)
            self.assertIn("google_maps", data["integrations"])
            self.assertIn("openai", data["integrations"])
            self.assertIn("yelp", data["integrations"])
            self.assertIn("stripe", data["integrations"])
            
            # Check for features
            self.assertIn("features", data)
            self.assertIn("business_intelligence", data["features"])
            self.assertIn("consumer_marketplace", data["features"])
            self.assertIn("payment_processing", data["features"])
            self.assertIn("real_time_messaging", data["features"])
            self.assertIn("advertising_platform", data["features"])
            
            # Store integration status for later tests
            self.api_integrations = data["integrations"]
            
            print(f"✅ Health check passed - Status: {response.status_code}")
            print(f"✅ API Integrations: Google Maps: {data['integrations']['google_maps']}, OpenAI: {data['integrations']['openai']}, Yelp: {data['integrations']['yelp']}, Stripe: {data['integrations']['stripe']}")
            print(f"✅ Features: {', '.join([k for k, v in data['features'].items() if v])}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Health check failed - Error: {str(e)}")
            raise

    def test_02_user_registration(self):
        """Test user registration for both business and consumer"""
        print(f"\n🔍 Testing user registration...")
        
        try:
            # Test business owner registration
            business_payload = {
                "email": self.test_email,
                "password": self.test_password,
                "user_type": "business",
                "business_name": self.test_business_name
            }
            
            response = requests.post(
                f"{self.base_url}/api/users/register",
                json=business_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("user_id", data)
            
            # Store user ID for later tests
            self.user_id = data["user_id"]
            
            print(f"✅ Business owner registration passed - User ID: {self.user_id}")
            
            # Test consumer registration
            consumer_email = f"consumer_{uuid.uuid4().hex[:8]}@example.com"
            consumer_payload = {
                "email": consumer_email,
                "password": self.test_password,
                "user_type": "consumer",
                "first_name": "Test",
                "last_name": "Consumer"
            }
            
            response = requests.post(
                f"{self.base_url}/api/users/register",
                json=consumer_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("user_id", data)
            
            print(f"✅ Consumer registration passed - User ID: {data['user_id']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ User registration failed - Error: {str(e)}")
            raise

    def test_03_get_user_profile(self):
        """Test getting user profile"""
        print(f"\n🔍 Testing get user profile...")
        
        # Make sure we have a user ID
        if not self.user_id:
            self.test_02_user_registration()
        
        try:
            response = requests.get(f"{self.base_url}/api/users/{self.user_id}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("id", data)
            self.assertIn("email", data)
            self.assertIn("user_type", data)
            self.assertIn("subscription_tier", data)
            self.assertIn("subscription_status", data)
            self.assertIn("credits", data)
            
            # Validate data
            self.assertEqual(data["id"], self.user_id)
            self.assertEqual(data["email"], self.test_email)
            self.assertEqual(data["user_type"], "business")
            self.assertEqual(data["subscription_tier"], "starter")
            self.assertEqual(data["subscription_status"], "active")
            
            print(f"✅ Get user profile passed - Email: {data['email']}, Type: {data['user_type']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get user profile failed - Error: {str(e)}")
            raise

    def test_04_create_business_profile(self):
        """Test creating a business profile"""
        print(f"\n🔍 Testing create business profile...")
        
        # Make sure we have a user ID
        if not self.user_id:
            self.test_02_user_registration()
        
        try:
            business_payload = {
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "business_name": self.test_business_name,
                "business_type": "restaurant",
                "address": "123 Test Street, New York, NY 10001",
                "phone": "555-123-4567",
                "website": "https://example.com",
                "description": "A test restaurant for API testing",
                "hours": {
                    "Monday": "9:00 AM - 10:00 PM",
                    "Tuesday": "9:00 AM - 10:00 PM",
                    "Wednesday": "9:00 AM - 10:00 PM",
                    "Thursday": "9:00 AM - 10:00 PM",
                    "Friday": "9:00 AM - 11:00 PM",
                    "Saturday": "10:00 AM - 11:00 PM",
                    "Sunday": "10:00 AM - 9:00 PM"
                },
                "amenities": ["Outdoor Seating", "Takeout", "Delivery"],
                "photos": ["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"]
            }
            
            response = requests.post(
                f"{self.base_url}/api/businesses",
                json=business_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("business_id", data)
            
            # Store business ID for later tests
            self.business_id = data["business_id"]
            
            print(f"✅ Create business profile passed - Business ID: {self.business_id}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Create business profile failed - Error: {str(e)}")
            raise

    def test_05_get_businesses(self):
        """Test getting businesses for consumer marketplace"""
        print(f"\n🔍 Testing get businesses...")
        
        try:
            response = requests.get(f"{self.base_url}/api/businesses?limit=10")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("businesses", data)
            self.assertIsInstance(data["businesses"], list)
            
            # Check if our business is in the list
            if self.business_id and data["businesses"]:
                business_ids = [business.get("id") for business in data["businesses"] if "id" in business]
                if self.business_id in business_ids:
                    print(f"  Found our business in the list")
            
            print(f"✅ Get businesses passed - Found {len(data['businesses'])} businesses")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get businesses failed - Error: {str(e)}")
            raise

    def test_06_get_business_details(self):
        """Test getting detailed business information"""
        print(f"\n🔍 Testing get business details...")
        
        # Make sure we have a business ID
        if not self.business_id:
            self.test_04_create_business_profile()
        
        try:
            response = requests.get(f"{self.base_url}/api/businesses/{self.business_id}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("id", data)
            self.assertIn("business_name", data)
            self.assertIn("business_type", data)
            self.assertIn("address", data)
            self.assertIn("reviews", data)
            
            # Validate data
            self.assertEqual(data["id"], self.business_id)
            self.assertEqual(data["business_name"], self.test_business_name)
            self.assertEqual(data["business_type"], "restaurant")
            
            print(f"✅ Get business details passed - Business: {data['business_name']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get business details failed - Error: {str(e)}")
            raise

    def test_07_create_review(self):
        """Test creating a consumer review"""
        print(f"\n🔍 Testing create review...")
        
        # Make sure we have a business ID
        if not self.business_id:
            self.test_04_create_business_profile()
        
        try:
            review_payload = {
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "business_id": self.business_id,
                "rating": 4.5,
                "review_text": "This is a test review for the restaurant. The food was great!",
                "photos": [],
                "visit_date": datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/reviews",
                json=review_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("review_id", data)
            
            print(f"✅ Create review passed - Review ID: {data['review_id']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Create review failed - Error: {str(e)}")
            raise

    def test_08_get_business_reviews(self):
        """Test getting reviews for a business"""
        print(f"\n🔍 Testing get business reviews...")
        
        # Make sure we have a business ID
        if not self.business_id:
            self.test_04_create_business_profile()
        
        try:
            response = requests.get(f"{self.base_url}/api/reviews/business/{self.business_id}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("reviews", data)
            self.assertIsInstance(data["reviews"], list)
            
            print(f"✅ Get business reviews passed - Found {len(data['reviews'])} reviews")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get business reviews failed - Error: {str(e)}")
            raise

    def test_09_send_message(self):
        """Test sending a message"""
        print(f"\n🔍 Testing send message...")
        
        # Make sure we have a user ID
        if not self.user_id:
            self.test_02_user_registration()
        
        try:
            # Create a recipient user for testing
            recipient_email = f"recipient_{uuid.uuid4().hex[:8]}@example.com"
            recipient_payload = {
                "email": recipient_email,
                "password": "TestPassword123!",
                "user_type": "consumer",
                "first_name": "Test",
                "last_name": "Recipient"
            }
            
            response = requests.post(
                f"{self.base_url}/api/users/register",
                json=recipient_payload
            )
            
            self.assertEqual(response.status_code, 200)
            recipient_data = response.json()
            recipient_id = recipient_data["user_id"]
            
            # Send a message
            message_payload = {
                "id": str(uuid.uuid4()),
                "sender_id": self.user_id,
                "recipient_id": recipient_id,
                "message_type": "text",
                "content": "This is a test message from the API test suite.",
                "business_id": self.business_id
            }
            
            response = requests.post(
                f"{self.base_url}/api/messages",
                json=message_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("message_id", data)
            
            print(f"✅ Send message passed - Message ID: {data['message_id']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Send message failed - Error: {str(e)}")
            raise

    def test_10_get_messages(self):
        """Test getting messages for a user"""
        print(f"\n🔍 Testing get messages...")
        
        # Make sure we have a user ID
        if not self.user_id:
            self.test_02_user_registration()
        
        try:
            response = requests.get(f"{self.base_url}/api/messages/{self.user_id}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("messages", data)
            self.assertIsInstance(data["messages"], list)
            
            print(f"✅ Get messages passed - Found {len(data['messages'])} messages")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get messages failed - Error: {str(e)}")
            raise

    def test_11_stripe_config(self):
        """Test getting Stripe configuration"""
        print(f"\n🔍 Testing Stripe configuration...")
        
        try:
            response = requests.get(f"{self.base_url}/api/stripe/config")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("publishableKey", data)
            self.assertTrue(data["publishableKey"].startswith("pk_"))
            
            print(f"✅ Stripe configuration passed - Key: {data['publishableKey'][:10]}...")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Stripe configuration failed - Error: {str(e)}")
            raise

    def test_12_subscription_tiers(self):
        """Test getting subscription tiers"""
        print(f"\n🔍 Testing subscription tiers...")
        
        try:
            response = requests.get(f"{self.base_url}/api/subscription-tiers")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("tiers", data)
            self.assertIsInstance(data["tiers"], list)
            
            # Validate each tier
            tier_ids = []
            for tier in data["tiers"]:
                self.assertIn("id", tier)
                self.assertIn("name", tier)
                self.assertIn("price", tier)
                self.assertIn("features", tier)
                self.assertIn("credits", tier)
                tier_ids.append(tier["id"])
            
            # Check for expected tier IDs
            expected_tiers = ["starter", "professional", "enterprise"]
            for tier in expected_tiers:
                self.assertIn(tier, tier_ids, f"Expected tier '{tier}' not found")
            
            # Check advertising packages
            self.assertIn("advertising_packages", data)
            self.assertIsInstance(data["advertising_packages"], list)
            
            for package in data["advertising_packages"]:
                self.assertIn("id", package)
                self.assertIn("name", package)
                self.assertIn("price", package)
                self.assertIn("duration_days", package)
                self.assertIn("features", package)
            
            print(f"✅ Subscription tiers passed - Found {len(data['tiers'])} tiers and {len(data['advertising_packages'])} advertising packages")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Subscription tiers failed - Error: {str(e)}")
            raise

    def test_13_create_checkout_session(self):
        """Test creating a checkout session"""
        print(f"\n🔍 Testing create checkout session...")
        
        # Make sure we have a user ID
        if not self.user_id:
            self.test_02_user_registration()
        
        try:
            # Test with free tier (starter)
            checkout_payload = {
                "package_type": "subscription",
                "package_id": "starter",
                "user_id": self.user_id,
                "origin_url": "https://example.com"
            }
            
            response = requests.post(
                f"{self.base_url}/api/payments/create-checkout-session",
                json=checkout_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # For free tier, we should get a message about activation
            if "message" in data:
                self.assertIn("Free package activated", data["message"])
                print(f"✅ Free tier activation passed - Message: {data['message']}")
            # For paid tiers, we should get a checkout URL
            elif "checkout_url" in data:
                self.assertIn("session_id", data)
                self.assertTrue(data["checkout_url"].startswith("https://"))
                print(f"✅ Checkout session creation passed - Session ID: {data['session_id']}")
            else:
                self.fail("Unexpected response format")
            
            # Test with paid tier (professional)
            checkout_payload = {
                "package_type": "subscription",
                "package_id": "professional",
                "user_id": self.user_id,
                "origin_url": "https://example.com"
            }
            
            response = requests.post(
                f"{self.base_url}/api/payments/create-checkout-session",
                json=checkout_payload
            )
            
            # If Stripe is not configured, this might fail
            if response.status_code == 200:
                data = response.json()
                if "checkout_url" in data:
                    self.assertIn("session_id", data)
                    self.assertTrue(data["checkout_url"].startswith("https://"))
                    print(f"✅ Paid tier checkout passed - Session ID: {data['session_id']}")
            elif response.status_code == 503:
                print(f"⚠️ Stripe payment processing not available - Status: {response.status_code}")
            
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Create checkout session failed - Error: {str(e)}")
            raise

    def test_14_create_advertisement(self):
        """Test creating a business advertisement"""
        print(f"\n🔍 Testing create advertisement...")
        
        # Make sure we have a business ID
        if not self.business_id:
            self.test_04_create_business_profile()
        
        try:
            ad_payload = {
                "id": str(uuid.uuid4()),
                "business_id": self.business_id,
                "user_id": self.user_id,
                "ad_type": "featured",
                "title": "Test Advertisement",
                "description": "This is a test advertisement for the API test suite.",
                "image_url": "https://example.com/ad.jpg",
                "target_demographics": ["foodies", "families"],
                "budget_amount": 99.0,
                "duration_days": 30,
                "expires_at": (datetime.utcnow().replace(microsecond=0) + timedelta(days=30)).isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/advertisements",
                json=ad_payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("ad_id", data)
            
            # Store ad ID for later tests
            self.ad_id = data["ad_id"]
            
            print(f"✅ Create advertisement passed - Ad ID: {self.ad_id}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Create advertisement failed - Error: {str(e)}")
            raise

    def test_15_get_advertisements(self):
        """Test getting active advertisements"""
        print(f"\n🔍 Testing get advertisements...")
        
        try:
            response = requests.get(f"{self.base_url}/api/advertisements")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("advertisements", data)
            self.assertIsInstance(data["advertisements"], list)
            
            print(f"✅ Get advertisements passed - Found {len(data['advertisements'])} advertisements")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get advertisements failed - Error: {str(e)}")
            raise

    def test_16_track_ad_click(self):
        """Test tracking advertisement click"""
        print(f"\n🔍 Testing track ad click...")
        
        # Make sure we have an ad ID
        if not hasattr(self, 'ad_id'):
            self.test_14_create_advertisement()
        
        try:
            response = requests.post(f"{self.base_url}/api/advertisements/{self.ad_id}/click")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertEqual(data["message"], "Click tracked successfully")
            
            print(f"✅ Track ad click passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Track ad click failed - Error: {str(e)}")
            raise

    def test_17_ultimate_competitor_search(self):
        """Test the ultimate competitor search endpoint"""
        print(f"\n🔍 Testing ultimate competitor search...")
        
        try:
            payload = {
                "location": self.location,
                "radius": 5,
                "business_type": "restaurant"
            }
            
            response = requests.post(
                f"{self.base_url}/api/ultimate-competitor-search",
                json=payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("competitors", data)
            self.assertIn("market_intelligence", data)
            self.assertIn("location", data)
            self.assertIn("total_found", data)
            
            # Validate market intelligence
            self.assertIn("total_competitors", data["market_intelligence"])
            self.assertIn("platform_members", data["market_intelligence"])
            self.assertIn("average_rating", data["market_intelligence"])
            
            # Store competitor IDs for later tests
            self.competitor_ids = [comp["id"] for comp in data["competitors"]]
            
            print(f"✅ Ultimate competitor search passed - Found {data['total_found']} competitors")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Ultimate competitor search failed - Error: {str(e)}")
            raise

    def test_18_generate_ultimate_intelligence_report(self):
        """Test generating ultimate intelligence report"""
        print(f"\n🔍 Testing generate ultimate intelligence report...")
        
        # Make sure we have competitor IDs
        if not self.competitor_ids:
            self.test_17_ultimate_competitor_search()
        
        try:
            # Use the first three competitor IDs
            test_ids = self.competitor_ids[:3]
            
            payload = {
                "competitor_ids": test_ids,
                "location": self.location,
                "user_business": {
                    "name": self.test_business_name
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate-ultimate-intelligence-report",
                json=payload
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("id", data)
            self.assertIn("report_type", data)
            self.assertIn("location", data)
            self.assertIn("competitors", data)
            self.assertIn("report_date", data)
            self.assertIn("pricing_analysis", data)
            self.assertIn("sentiment_analysis", data)
            self.assertIn("executive_summary", data)
            self.assertIn("strategic_recommendations", data)
            
            # Store report ID for later tests
            self.report_id = data["id"]
            
            print(f"✅ Generate ultimate intelligence report passed - Report ID: {data['id']}")
            print(f"  Executive Summary: {data['executive_summary'][:100]}...")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Generate ultimate intelligence report failed - Error: {str(e)}")
            raise

    def print_summary(self):
        print(f"\n📊 Tests passed: {self.tests_passed}/{self.tests_run}")
        return self.tests_passed == self.tests_run

def run_tests():
    # Create test suite
    suite = unittest.TestSuite()
    tester = BizFizzUltimateAPITester()
    
    # Add tests in order
    suite.addTest(BizFizzUltimateAPITester('test_01_health_check'))
    suite.addTest(BizFizzUltimateAPITester('test_02_user_registration'))
    suite.addTest(BizFizzUltimateAPITester('test_03_get_user_profile'))
    suite.addTest(BizFizzUltimateAPITester('test_04_create_business_profile'))
    suite.addTest(BizFizzUltimateAPITester('test_05_get_businesses'))
    suite.addTest(BizFizzUltimateAPITester('test_06_get_business_details'))
    suite.addTest(BizFizzUltimateAPITester('test_07_create_review'))
    suite.addTest(BizFizzUltimateAPITester('test_08_get_business_reviews'))
    suite.addTest(BizFizzUltimateAPITester('test_09_send_message'))
    suite.addTest(BizFizzUltimateAPITester('test_10_get_messages'))
    suite.addTest(BizFizzUltimateAPITester('test_11_stripe_config'))
    suite.addTest(BizFizzUltimateAPITester('test_12_subscription_tiers'))
    suite.addTest(BizFizzUltimateAPITester('test_13_create_checkout_session'))
    suite.addTest(BizFizzUltimateAPITester('test_14_create_advertisement'))
    suite.addTest(BizFizzUltimateAPITester('test_15_get_advertisements'))
    suite.addTest(BizFizzUltimateAPITester('test_16_track_ad_click'))
    suite.addTest(BizFizzUltimateAPITester('test_17_ultimate_competitor_search'))
    suite.addTest(BizFizzUltimateAPITester('test_18_generate_ultimate_intelligence_report'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    tester.print_summary()
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())