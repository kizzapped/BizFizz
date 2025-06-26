import requests
import unittest
import json
import sys
import uuid
from datetime import datetime

class BizFizzAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BizFizzAPITester, self).__init__(*args, **kwargs)
        # Use the external URL
        self.base_url = "https://32969b86-898a-4616-9a3f-d03a79d2efff.preview.emergentagent.com"
        self.tests_run = 0
        self.tests_passed = 0
        
        # Social media monitoring test data
        self.test_business_id = str(uuid.uuid4())
        self.test_business_name = "Pasta Paradise Italian Restaurant"
        self.test_alert_id = None

    def setUp(self):
        self.tests_run += 1

    def test_01_health_check(self):
        """Test the health check endpoint"""
        print(f"\n🔍 Testing health check endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            
            # Check API integrations
            self.assertIn("integrations", data)
            self.assertIsInstance(data["integrations"], dict)
            
            # Check for features
            self.assertIn("features", data)
            
            print(f"✅ Health check passed - Status: {response.status_code}")
            print(f"✅ API Integrations: {data['integrations']}")
            print(f"✅ Features: {data['features']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Health check failed - Error: {str(e)}")
            raise

    def test_02_start_social_monitoring(self):
        """Test starting social media monitoring"""
        print(f"\n🔍 Testing social media monitoring start endpoint...")
        
        try:
            # Create a monitoring rule
            monitoring_rule = {
                "id": str(uuid.uuid4()),
                "business_id": self.test_business_id,
                "business_name": self.test_business_name,
                "keywords": ["pasta", "italian food", "pizza", self.test_business_name],
                "mentions": ["@pastaparadise", "#italianfood"],
                "hashtags": ["#pasta", "#italianfood", "#foodie"],
                "platforms": ["twitter", "facebook", "news"],
                "alert_settings": {
                    "negative_sentiment_threshold": -0.3,
                    "high_engagement_threshold": 50,
                    "notify_email": "owner@pastaparadise.com",
                    "notify_sms": "+15551234567"
                },
                "is_active": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/social/monitoring/start",
                json=monitoring_rule,
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("message", data)
            self.assertIn("rule_id", data)
            
            print(f"✅ Social monitoring start passed - Rule ID: {data['rule_id']}")
            print(f"  Response: {data['message']}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Social monitoring start failed - Error: {str(e)}")
            raise

    def test_03_get_social_mentions(self):
        """Test getting social media mentions"""
        print(f"\n🔍 Testing get social mentions endpoint...")
        
        try:
            # Get mentions for our test business
            response = requests.get(
                f"{self.base_url}/api/social/mentions/{self.test_business_id}",
                params={"limit": 10},
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("mentions", data)
            self.assertIn("total", data)
            
            print(f"✅ Get social mentions passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get social mentions failed - Error: {str(e)}")
            raise

    def test_04_get_social_alerts(self):
        """Test getting social media alerts"""
        print(f"\n🔍 Testing get social alerts endpoint...")
        
        try:
            # Get alerts for our test business
            response = requests.get(
                f"{self.base_url}/api/social/alerts/{self.test_business_id}",
                params={"limit": 10},
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("alerts", data)
            self.assertIn("total", data)
            
            # If we have alerts, store an ID for the mark-read test
            if data["alerts"]:
                self.test_alert_id = data["alerts"][0]["id"]
            
            print(f"✅ Get social alerts passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get social alerts failed - Error: {str(e)}")
            raise

    def test_05_mark_alert_read(self):
        """Test marking an alert as read"""
        print(f"\n🔍 Testing mark alert as read endpoint...")
        
        try:
            # Skip if we don't have an alert ID
            if not self.test_alert_id:
                print(f"  No alert ID available, using a mock ID")
                self.test_alert_id = str(uuid.uuid4())
            
            # Mark the alert as read
            response = requests.put(
                f"{self.base_url}/api/social/alerts/{self.test_alert_id}/mark-read",
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            # Accept either 200 (success) or 404 (alert not found)
            self.assertTrue(response.status_code in [200, 404], 
                           f"Expected status code 200 or 404, got {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn("message", data)
            
            print(f"✅ Mark alert as read passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Mark alert as read failed - Error: {str(e)}")
            raise

    def test_06_get_social_analytics(self):
        """Test getting social media analytics"""
        print(f"\n🔍 Testing get social analytics endpoint...")
        
        try:
            # Get analytics for our test business
            response = requests.get(
                f"{self.base_url}/api/social/analytics/{self.test_business_id}",
                params={"days": 7},
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("sentiment_distribution", data)
            self.assertIn("platform_distribution", data)
            self.assertIn("time_series", data)
            self.assertIn("alert_summary", data)
            self.assertIn("date_range", data)
            
            print(f"✅ Get social analytics passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get social analytics failed - Error: {str(e)}")
            raise

    def test_07_get_news_articles(self):
        """Test getting news articles"""
        print(f"\n🔍 Testing get news articles endpoint...")
        
        try:
            # Get news articles
            response = requests.get(
                f"{self.base_url}/api/news/articles",
                params={"keywords": "restaurant,food", "limit": 10},
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validate response structure
            self.assertIn("articles", data)
            self.assertIn("total", data)
            
            print(f"✅ Get news articles passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ Get news articles failed - Error: {str(e)}")
            raise

    def print_summary(self):
        print(f"\n📊 Tests passed: {self.tests_passed}/{self.tests_run}")
        return self.tests_passed == self.tests_run

def run_tests():
    # Create test suite
    suite = unittest.TestSuite()
    tester = BizFizzAPITester()
    
    # Add tests in order
    suite.addTest(BizFizzAPITester('test_01_health_check'))
    suite.addTest(BizFizzAPITester('test_02_start_social_monitoring'))
    suite.addTest(BizFizzAPITester('test_03_get_social_mentions'))
    suite.addTest(BizFizzAPITester('test_04_get_social_alerts'))
    suite.addTest(BizFizzAPITester('test_05_mark_alert_read'))
    suite.addTest(BizFizzAPITester('test_06_get_social_analytics'))
    suite.addTest(BizFizzAPITester('test_07_get_news_articles'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    tester.print_summary()
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())