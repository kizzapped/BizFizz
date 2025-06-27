import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area } from 'recharts';
import { FaSearch, FaChartBar, FaUsers, FaDollarSign, FaStar, FaMapMarkerAlt, FaPhone, FaGlobe, FaArrowLeft, FaDownload, FaPrint, FaEye, FaBuilding, FaChartLine, FaShieldAlt, FaBrain, FaRocket, FaCog, FaLightbulb, FaTarget, FaTrendingUp, FaUserTie, FaIndustry, FaChartPie, FaMagic, FaGem, FaCrown, FaComments, FaStore, FaHeart, FaCreditCard, FaCheck, FaEnvelope, FaImage, FaBell, FaFilter, FaSort, FaNewspaper, FaTwitter } from 'react-icons/fa';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [userType, setUserType] = useState('business'); // business or consumer
  const [currentUser, setCurrentUser] = useState(null);
  const [searchLocation, setSearchLocation] = useState('');
  const [searchRadius, setSearchRadius] = useState(5);
  const [competitors, setCompetitors] = useState([]);
  const [selectedCompetitors, setSelectedCompetitors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [reports, setReports] = useState([]);
  const [subscriptionTiers, setSubscriptionTiers] = useState([]);
  const [marketIntelligence, setMarketIntelligence] = useState(null);
  const [businesses, setBusinesses] = useState([]);
  const [selectedBusiness, setSelectedBusiness] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [websocket, setWebsocket] = useState(null);
  const [advertisements, setAdvertisements] = useState([]);
  const [socialMentions, setSocialMentions] = useState([]);
  const [socialAlerts, setSocialAlerts] = useState([]);
  const [monitoringRules, setMonitoringRules] = useState([]);
  const [socialAnalytics, setSocialAnalytics] = useState(null);
  const [newsArticles, setNewsArticles] = useState([]);
  const [realTimeDashboard, setRealTimeDashboard] = useState(null);
  const [comprehensiveAnalytics, setComprehensiveAnalytics] = useState(null);
  const [mobileNotifications, setMobileNotifications] = useState([]);
  const [isMobileView, setIsMobileView] = useState(false);
  const [newMonitoringRule, setNewMonitoringRule] = useState({
    business_name: '',
    keywords: '',
    mentions: '',
    hashtags: '',
    platforms: ['twitter', 'facebook', 'google', 'news']
  });
  const [userRegistration, setUserRegistration] = useState({
    email: '',
    password: '',
    user_type: 'consumer',
    first_name: '',
    last_name: '',
    business_name: ''
  });

  useEffect(() => {
    fetchSubscriptionTiers();
    fetchBusinesses();
    fetchAdvertisements();
    
    // Mobile detection
    const checkMobile = () => {
      setIsMobileView(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    if (currentUser) {
      initializeWebSocket();
      fetchSocialMentions();
      fetchSocialAlerts();
      fetchNewsArticles();
      fetchMobileNotifications();
      
      if (currentUser.user_type === 'business') {
        fetchSocialAnalytics();
        fetchRealTimeDashboard();
        fetchComprehensiveAnalytics();
        
        // Real-time dashboard updates every 30 seconds
        const interval = setInterval(() => {
          fetchRealTimeDashboard();
          fetchMobileNotifications();
        }, 30000);
        
        return () => {
          clearInterval(interval);
          window.removeEventListener('resize', checkMobile);
        };
      }
    }
    
    return () => {
      if (websocket) {
        websocket.close();
      }
      window.removeEventListener('resize', checkMobile);
    };
  }, [currentUser]);

  const initializeWebSocket = () => {
    if (currentUser && !websocket) {
      const ws_url = API_BASE_URL.replace('http', 'ws');
      const ws = new WebSocket(`${ws_url}/ws/${currentUser.id}`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'new_message') {
          setMessages(prev => [data.message, ...prev]);
        }
      };
      
      setWebsocket(ws);
    }
  };

  const fetchSubscriptionTiers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/subscription-tiers`);
      const data = await response.json();
      setSubscriptionTiers(data.tiers);
    } catch (error) {
      console.error('Error fetching subscription tiers:', error);
    }
  };

  const fetchBusinesses = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/businesses?limit=50`);
      const data = await response.json();
      setBusinesses(data.businesses);
    } catch (error) {
      console.error('Error fetching businesses:', error);
    }
  };

  const fetchAdvertisements = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/advertisements`);
      const data = await response.json();
      setAdvertisements(data.advertisements);
    } catch (error) {
      console.error('Error fetching advertisements:', error);
    }
  };

  const fetchSocialMentions = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/social/mentions/${currentUser.id}`);
        const data = await response.json();
        setSocialMentions(data.mentions || []);
      }
    } catch (error) {
      console.error('Error fetching social mentions:', error);
    }
  };

  const fetchSocialAlerts = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/social/alerts/${currentUser.id}`);
        const data = await response.json();
        setSocialAlerts(data.alerts || []);
      }
    } catch (error) {
      console.error('Error fetching social alerts:', error);
    }
  };

  const fetchSocialAnalytics = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/social/analytics/${currentUser.id}?days=7`);
        const data = await response.json();
        setSocialAnalytics(data);
      }
    } catch (error) {
      console.error('Error fetching social analytics:', error);
    }
  };

  const fetchNewsArticles = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/news/articles?keywords=restaurant,food`);
      const data = await response.json();
      setNewsArticles(data.articles || []);
    } catch (error) {
      console.error('Error fetching news articles:', error);
    }
  };

  const fetchRealTimeDashboard = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/analytics/realtime-dashboard/${currentUser.id}`);
        const data = await response.json();
        setRealTimeDashboard(data);
      }
    } catch (error) {
      console.error('Error fetching real-time dashboard:', error);
    }
  };

  const fetchComprehensiveAnalytics = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/analytics/comprehensive/${currentUser.id}?days=30&include_competitors=true&include_predictions=true`);
        const data = await response.json();
        setComprehensiveAnalytics(data);
      }
    } catch (error) {
      console.error('Error fetching comprehensive analytics:', error);
    }
  };

  const fetchMobileNotifications = async () => {
    try {
      if (currentUser && currentUser.id) {
        const response = await fetch(`${API_BASE_URL}/api/mobile/notifications/${currentUser.id}?limit=20`);
        const data = await response.json();
        setMobileNotifications(data.notifications || []);
      }
    } catch (error) {
      console.error('Error fetching mobile notifications:', error);
    }
  };

  const registerUser = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userRegistration),
      });
      
      if (response.ok) {
        const data = await response.json();
        // Simulate login after registration
        setCurrentUser({
          id: data.user_id,
          email: userRegistration.email,
          user_type: userRegistration.user_type,
          subscription_tier: 'starter'
        });
        setCurrentPage('dashboard');
      }
    } catch (error) {
      console.error('Error registering user:', error);
    }
  };

  const createCheckoutSession = async (packageType, packageId) => {
    try {
      if (!currentUser) {
        alert('Please register/login first');
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/payments/create-checkout-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          package_type: packageType,
          package_id: packageId,
          user_id: currentUser.id,
          origin_url: window.location.origin
        }),
      });
      
      const data = await response.json();
      
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      } else if (data.message) {
        alert(data.message);
        // Refresh user data for free packages
        window.location.reload();
      }
    } catch (error) {
      console.error('Error creating checkout session:', error);
    }
  };

  const sendMessage = async (recipientId, content, messageType = 'text') => {
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
          recipient_id: recipientId,
          content: content,
          type: messageType
        }));
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const searchCompetitors = async () => {
    if (!searchLocation.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/ultimate-competitor-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          location: searchLocation,
          radius: searchRadius,
          business_type: 'restaurant',
          include_demographics: true,
          include_social_media: true
        }),
      });
      
      const data = await response.json();
      setCompetitors(data.competitors);
      setMarketIntelligence(data.market_intelligence);
      setCurrentPage('competitors');
    } catch (error) {
      console.error('Error searching competitors:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateUltimateReport = async () => {
    if (selectedCompetitors.length === 0) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-ultimate-intelligence-report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          competitor_ids: selectedCompetitors,
          location: searchLocation,
          user_business: currentUser?.business_name ? { name: currentUser.business_name } : null,
          analysis_depth: "ultimate",
          include_predictions: true,
          include_recommendations: true
        }),
      });
      
      const data = await response.json();
      setReport(data);
      setCurrentPage('ultimate-report');
    } catch (error) {
      console.error('Error generating ultimate report:', error);
    } finally {
      setLoading(false);
    }
  };

  const startSocialMonitoring = async () => {
    try {
      if (!currentUser || !newMonitoringRule.business_name || !newMonitoringRule.keywords) {
        alert('Please fill in all required fields');
        return;
      }

      const rule = {
        business_id: currentUser.id,
        business_name: newMonitoringRule.business_name,
        keywords: newMonitoringRule.keywords.split(',').map(k => k.trim()),
        mentions: newMonitoringRule.mentions.split(',').map(m => m.trim()).filter(m => m),
        hashtags: newMonitoringRule.hashtags.split(',').map(h => h.trim()).filter(h => h),
        platforms: newMonitoringRule.platforms,
        alert_settings: {
          negative_sentiment_threshold: -0.5,
          high_engagement_threshold: 100,
          email_alerts: true,
          realtime_alerts: true
        },
        is_active: true
      };

      const response = await fetch(`${API_BASE_URL}/api/social/monitoring/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(rule),
      });
      
      if (response.ok) {
        alert('Social media monitoring started successfully!');
        setNewMonitoringRule({
          business_name: '',
          keywords: '',
          mentions: '',
          hashtags: '',
          platforms: ['twitter', 'facebook', 'news']
        });
        fetchSocialMentions();
        fetchSocialAlerts();
      }
    } catch (error) {
      console.error('Error starting social monitoring:', error);
      alert('Error starting social monitoring');
    }
  };

  const markAlertAsRead = async (alertId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/social/alerts/${alertId}/mark-read`, {
        method: 'PUT',
      });
      
      if (response.ok) {
        fetchSocialAlerts();
      }
    } catch (error) {
      console.error('Error marking alert as read:', error);
    }
  };

  const toggleCompetitorSelection = (competitorId) => {
    setSelectedCompetitors(prev => 
      prev.includes(competitorId)
        ? prev.filter(id => id !== competitorId)
        : [...prev, competitorId]
    );
  };

  const HomePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <FaBrain className="text-3xl text-blue-600 mr-3" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">BizFizz</h1>
                <p className="text-sm text-gray-600">Ultimate Business Intelligence & Consumer Marketplace</p>
              </div>
            </div>
            
            {/* User Type Selector */}
            <div className="flex items-center space-x-4">
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setUserType('business')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    userType === 'business' 
                      ? 'bg-blue-600 text-white' 
                      : 'text-gray-600 hover:text-blue-600'
                  }`}
                >
                  <FaBuilding className="inline mr-2" />
                  Business
                </button>
                <button
                  onClick={() => setUserType('consumer')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    userType === 'consumer' 
                      ? 'bg-green-600 text-white' 
                      : 'text-gray-600 hover:text-green-600'
                  }`}
                >
                  <FaUsers className="inline mr-2" />
                  Consumer
                </button>
              </div>
              
              <nav className="flex space-x-6">
                <button 
                  onClick={() => setCurrentPage('home')}
                  className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                >
                  Home
                </button>
                {userType === 'business' && (
                  <>
                    <button 
                      onClick={() => setCurrentPage('dashboard')}
                      className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                    >
                      Dashboard
                    </button>
                    <button 
                      onClick={() => setCurrentPage('social-monitoring')}
                      className="text-gray-700 hover:text-blue-600 font-medium transition-colors flex items-center"
                    >
                      <FaBell className="mr-1" />
                      Live Monitoring
                      {socialAlerts.filter(a => !a.is_read).length > 0 && (
                        <span className="ml-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                          {socialAlerts.filter(a => !a.is_read).length}
                        </span>
                      )}
                    </button>
                  </>
                )}
                {userType === 'consumer' && (
                  <button 
                    onClick={() => setCurrentPage('marketplace')}
                    className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                  >
                    Marketplace
                  </button>
                )}
                <button 
                  onClick={() => setCurrentPage('pricing')}
                  className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
                >
                  Pricing
                </button>
                {!currentUser && (
                  <button 
                    onClick={() => setCurrentPage('register')}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Sign Up
                  </button>
                )}
                {currentUser && (
                  <button 
                    onClick={() => setCurrentPage('messages')}
                    className="text-gray-700 hover:text-blue-600 font-medium transition-colors relative"
                  >
                    <FaComments className="inline mr-1" />
                    Messages
                    {messages.length > 0 && (
                      <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                        {messages.length}
                      </span>
                    )}
                  </button>
                )}
              </nav>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {userType === 'business' ? (
            <>
              <div className="text-center mb-16">
                <h2 className="text-6xl font-bold text-gray-900 mb-6">
                  Dominate Your <span className="text-blue-600">Market</span>
                </h2>
                <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                  Get AI-powered competitive intelligence, connect with customers, and grow your business with the ultimate platform for restaurant owners.
                </p>
              </div>

              {/* Business Search Interface */}
              <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl p-8 mb-16">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">Start Your Competitive Analysis</h3>
                  <p className="text-gray-600">Discover your competition and dominate your local market</p>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4 mb-8">
                  <div className="flex-1 relative">
                    <FaMapMarkerAlt className="absolute left-3 top-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Enter your location (zip code, city, address)"
                      value={searchLocation}
                      onChange={(e) => setSearchLocation(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && searchCompetitors()}
                      className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
                    />
                  </div>
                  <select
                    value={searchRadius}
                    onChange={(e) => setSearchRadius(Number(e.target.value))}
                    className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent min-w-[150px]"
                  >
                    <option value={1}>1 mile radius</option>
                    <option value={3}>3 miles radius</option>
                    <option value={5}>5 miles radius</option>
                    <option value={10}>10 miles radius</option>
                    <option value={15}>15 miles radius</option>
                  </select>
                </div>
                
                <button
                  onClick={searchCompetitors}
                  disabled={loading || !searchLocation.trim()}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg transition-all transform hover:scale-105 flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Analyzing Market...
                    </>
                  ) : (
                    <>
                      <FaSearch className="mr-2" />
                      Discover Competitors
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="text-center mb-16">
                <h2 className="text-6xl font-bold text-gray-900 mb-6">
                  Discover Amazing <span className="text-green-600">Restaurants</span>
                </h2>
                <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                  Find the best restaurants, read authentic reviews, and connect with fellow food lovers in our community marketplace.
                </p>
              </div>

              {/* Consumer Search Interface */}
              <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl p-8 mb-16">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">Find Your Perfect Dining Experience</h3>
                  <p className="text-gray-600">Search restaurants, read reviews, and discover hidden gems</p>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4 mb-6">
                  <input
                    type="text"
                    placeholder="Search restaurants, cuisine, or location..."
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent text-base"
                  />
                  <button
                    onClick={() => setCurrentPage('marketplace')}
                    className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-8 py-3 rounded-lg hover:from-green-700 hover:to-emerald-700 font-semibold flex items-center justify-center"
                  >
                    <FaSearch className="mr-2" />
                    Explore Restaurants
                  </button>
                </div>
                
                <div className="flex flex-wrap gap-2 justify-center">
                  {['Italian', 'Asian', 'Mexican', 'American', 'Fast Food', 'Fine Dining'].map((cuisine) => (
                    <button
                      key={cuisine}
                      className="px-4 py-2 bg-gray-100 text-gray-700 rounded-full hover:bg-green-100 hover:text-green-700 transition-colors text-sm"
                    >
                      {cuisine}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Featured Advertisements */}
          {advertisements.length > 0 && (
            <div className="mb-16">
              <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">Featured Restaurants</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {advertisements.slice(0, 3).map((ad) => (
                  <div key={ad.id} className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
                    {ad.image_url && (
                      <img src={ad.image_url} alt={ad.title} className="w-full h-48 object-cover" />
                    )}
                    <div className="p-6">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-lg font-semibold">{ad.title}</h4>
                        <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">
                          Featured
                        </span>
                      </div>
                      <p className="text-gray-600 text-sm mb-4">{ad.description}</p>
                      <button
                        onClick={() => {
                          // Track click
                          fetch(`${API_BASE_URL}/api/advertisements/${ad.id}/click`, { method: 'POST' });
                        }}
                        className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        Learn More
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {userType === 'business' ? [
              {
                title: "AI-Powered Intelligence",
                description: "10 specialized AI agents analyze your competition 24/7",
                icon: <FaBrain className="text-4xl text-blue-500" />,
                features: ["Market Analysis", "Pricing Intelligence", "Customer Sentiment", "Competitive Monitoring"]
              },
              {
                title: "Consumer Marketplace",
                description: "Reach customers directly through our platform",
                icon: <FaStore className="text-4xl text-green-500" />,
                features: ["Business Listings", "Customer Reviews", "Direct Messaging", "Advertising Tools"]
              },
              {
                title: "Real-time Insights",
                description: "Get instant alerts and actionable recommendations",
                icon: <FaChartLine className="text-4xl text-purple-500" />,
                features: ["Live Dashboard", "Performance Metrics", "Growth Opportunities", "Risk Alerts"]
              }
            ] : [
              {
                title: "Restaurant Discovery",
                description: "Find amazing restaurants with detailed information",
                icon: <FaSearch className="text-4xl text-green-500" />,
                features: ["Advanced Search", "Photo Galleries", "Menu Viewing", "Location Mapping"]
              },
              {
                title: "Community Reviews",
                description: "Read and share authentic dining experiences",
                icon: <FaStar className="text-4xl text-yellow-500" />,
                features: ["Verified Reviews", "Photo Sharing", "Rating System", "Helpful Votes"]
              },
              {
                title: "Social Features",
                description: "Connect with fellow food enthusiasts",
                icon: <FaComments className="text-4xl text-blue-500" />,
                features: ["Direct Messaging", "Recommendations", "Favorite Lists", "Follow Friends"]
              }
            ].map((feature, index) => (
              <div key={index} className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center mb-6">
                  {feature.icon}
                  <h4 className="text-xl font-semibold ml-4">{feature.title}</h4>
                </div>
                <p className="text-gray-600 mb-6">{feature.description}</p>
                <ul className="space-y-2">
                  {feature.features.map((item, idx) => (
                    <li key={idx} className="flex items-center text-sm text-gray-500">
                      <FaCheck className="text-green-500 mr-2" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );

  const RegisterPage = () => (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <div className="flex justify-center">
            <FaBrain className="text-5xl text-blue-600" />
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Join BizFizz Today
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Choose your account type and get started
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-8">
          <div className="space-y-6">
            {/* User Type Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Account Type</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setUserRegistration({...userRegistration, user_type: 'business'})}
                  className={`flex items-center justify-center px-4 py-3 border rounded-md text-sm font-medium ${
                    userRegistration.user_type === 'business'
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <FaBuilding className="mr-2" />
                  Business Owner
                </button>
                <button
                  type="button"
                  onClick={() => setUserRegistration({...userRegistration, user_type: 'consumer'})}
                  className={`flex items-center justify-center px-4 py-3 border rounded-md text-sm font-medium ${
                    userRegistration.user_type === 'consumer'
                      ? 'border-green-500 bg-green-50 text-green-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <FaUsers className="mr-2" />
                  Food Lover
                </button>
              </div>
            </div>

            {/* Form Fields */}
            <div className="grid grid-cols-1 gap-4">
              {userRegistration.user_type === 'consumer' && (
                <>
                  <input
                    type="text"
                    placeholder="First Name"
                    value={userRegistration.first_name}
                    onChange={(e) => setUserRegistration({...userRegistration, first_name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <input
                    type="text"
                    placeholder="Last Name"
                    value={userRegistration.last_name}
                    onChange={(e) => setUserRegistration({...userRegistration, last_name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </>
              )}
              
              {userRegistration.user_type === 'business' && (
                <input
                  type="text"
                  placeholder="Business Name"
                  value={userRegistration.business_name}
                  onChange={(e) => setUserRegistration({...userRegistration, business_name: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
              
              <input
                type="email"
                placeholder="Email Address"
                value={userRegistration.email}
                onChange={(e) => setUserRegistration({...userRegistration, email: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              
              <input
                type="password"
                placeholder="Password"
                value={userRegistration.password}
                onChange={(e) => setUserRegistration({...userRegistration, password: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <button
              onClick={registerUser}
              className={`w-full py-3 px-4 rounded-md text-white font-medium ${
                userRegistration.user_type === 'business'
                  ? 'bg-blue-600 hover:bg-blue-700'
                  : 'bg-green-600 hover:bg-green-700'
              } transition-colors`}
            >
              Create Account
            </button>

            <div className="text-center">
              <button
                onClick={() => setCurrentPage('home')}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                ← Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const MarketplacePage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Restaurant Marketplace</h2>
          <p className="text-gray-600">Discover amazing restaurants and connect with fellow food lovers</p>
        </div>

        {/* Search and Filters */}
        <div className="bg-white p-6 rounded-lg shadow-sm mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search restaurants..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
              />
            </div>
            <select className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500">
              <option value="">All Cuisines</option>
              <option value="italian">Italian</option>
              <option value="asian">Asian</option>
              <option value="mexican">Mexican</option>
              <option value="american">American</option>
            </select>
            <select className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500">
              <option value="">All Prices</option>
              <option value="1">$ Budget</option>
              <option value="2">$$ Moderate</option>
              <option value="3">$$$ Expensive</option>
              <option value="4">$$$$ Very Expensive</option>
            </select>
          </div>
        </div>

        {/* Restaurant Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {businesses.map((business) => (
            <div key={business.id} className="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer">
              {business.photos && business.photos.length > 0 && (
                <img 
                  src={business.photos[0]} 
                  alt={business.business_name}
                  className="w-full h-48 object-cover rounded-t-lg"
                />
              )}
              <div className="p-6">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">{business.business_name}</h3>
                  <div className="flex items-center">
                    <FaStar className="text-yellow-400 mr-1" />
                    <span className="text-sm text-gray-600">{business.avg_rating || 'New'}</span>
                  </div>
                </div>
                
                <p className="text-gray-600 text-sm mb-3">{business.address}</p>
                
                {business.description && (
                  <p className="text-gray-700 text-sm mb-4 line-clamp-2">{business.description}</p>
                )}
                
                <div className="flex justify-between items-center">
                  <div className="flex space-x-2">
                    <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                      {business.business_type}
                    </span>
                    {business.is_verified && (
                      <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                        <FaCheck className="inline mr-1" />
                        Verified
                      </span>
                    )}
                  </div>
                  
                  <button
                    onClick={() => {
                      setSelectedBusiness(business);
                      setCurrentPage('business-details');
                    }}
                    className="text-green-600 hover:text-green-800 text-sm font-medium"
                  >
                    View Details
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const PricingPage = () => (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Choose Your Plan
          </h2>
          <p className="text-xl text-gray-600">
            {userType === 'business' 
              ? 'Get the competitive intelligence you need to dominate your market'
              : 'Enhance your dining experience with premium features'
            }
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          {subscriptionTiers.map((tier, index) => (
            <div
              key={tier.id}
              className={`bg-white rounded-xl shadow-lg p-8 ${
                tier.popular ? 'border-2 border-blue-500 relative transform scale-105' : 'border border-gray-200'
              }`}
            >
              {tier.popular && (
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <span className="bg-blue-500 text-white px-6 py-2 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
              )}
              
              <div className="text-center">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">{tier.name}</h3>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-gray-900">${tier.price}</span>
                  <span className="text-gray-600">/month</span>
                </div>
                
                <ul className="text-left space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start">
                      <FaCheck className="text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <button
                  onClick={() => createCheckoutSession('subscription', tier.id)}
                  className={`w-full py-3 rounded-lg font-semibold transition-all ${
                    tier.popular
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  }`}
                >
                  {tier.price === 0 ? 'Get Started Free' : 'Start Free Trial'}
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Business Advertising Packages */}
        {userType === 'business' && (
          <div>
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Advertising Packages
              </h3>
              <p className="text-lg text-gray-600">
                Promote your business to thousands of food lovers
              </p>
            </div>
            
            {/* Add advertising packages here */}
            <div className="bg-white rounded-lg p-8 text-center">
              <FaRocket className="text-4xl text-orange-500 mx-auto mb-4" />
              <h4 className="text-xl font-semibold mb-2">Coming Soon: Advanced Advertising</h4>
              <p className="text-gray-600">Premium advertising features to reach more customers</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const MessagesPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow-sm h-96">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold">Messages</h3>
          </div>
          <div className="p-6">
            <div className="text-center text-gray-500">
              <FaComments className="text-4xl mx-auto mb-4" />
              <p>Real-time messaging coming soon!</p>
              <p className="text-sm">Connect with other food lovers and businesses</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const AdvancedAnalyticsDashboard = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">📊 Advanced Analytics Dashboard</h2>
          <p className="text-gray-600">AI-powered insights and comprehensive business intelligence</p>
        </div>

        {/* Real-time Status Cards */}
        {realTimeDashboard && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-sm p-6 border-l-4 border-red-500">
              <div className="flex items-center">
                <FaBell className="text-red-500 text-2xl mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Critical Alerts</p>
                  <p className="text-2xl font-bold text-gray-900">{realTimeDashboard.active_alerts}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-sm p-6 border-l-4 border-blue-500">
              <div className="flex items-center">
                <FaComments className="text-blue-500 text-2xl mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Recent Mentions</p>
                  <p className="text-2xl font-bold text-gray-900">{realTimeDashboard.recent_mentions}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-sm p-6 border-l-4 border-green-500">
              <div className="flex items-center">
                <FaChartLine className="text-green-500 text-2xl mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Status</p>
                  <p className="text-lg font-bold text-gray-900 capitalize">{realTimeDashboard.status}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-sm p-6 border-l-4 border-purple-500">
              <div className="flex items-center">
                <FaClock className="text-purple-500 text-2xl mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Last Updated</p>
                  <p className="text-sm text-gray-900">{new Date(realTimeDashboard.last_updated).toLocaleTimeString()}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Comprehensive Analytics */}
        {comprehensiveAnalytics && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Reputation Score */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <FaStar className="mr-2 text-yellow-500" />
                Reputation Score
              </h3>
              <div className="text-center">
                <div className={`text-6xl font-bold mb-2 ${
                  comprehensiveAnalytics.reputation_score > 70 ? 'text-green-500' :
                  comprehensiveAnalytics.reputation_score > 40 ? 'text-yellow-500' : 'text-red-500'
                }`}>
                  {comprehensiveAnalytics.reputation_score}%
                </div>
                <p className="text-gray-600">
                  {comprehensiveAnalytics.reputation_score > 70 ? 'Excellent' :
                   comprehensiveAnalytics.reputation_score > 40 ? 'Good' : 'Needs Improvement'}
                </p>
              </div>
              
              <div className="mt-6 grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {comprehensiveAnalytics.social_performance.positive_mentions}
                  </div>
                  <div className="text-xs text-gray-500">Positive</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-gray-600">
                    {comprehensiveAnalytics.social_performance.neutral_mentions}
                  </div>
                  <div className="text-xs text-gray-500">Neutral</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-600">
                    {comprehensiveAnalytics.social_performance.negative_mentions}
                  </div>
                  <div className="text-xs text-gray-500">Negative</div>
                </div>
              </div>
            </div>

            {/* AI Insights */}
            {comprehensiveAnalytics.ai_insights && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <FaBrain className="mr-2 text-purple-500" />
                  AI-Powered Insights
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Key Insights</h4>
                    <ul className="space-y-2">
                      {comprehensiveAnalytics.ai_insights.insights?.map((insight, index) => (
                        <li key={index} className="flex items-start">
                          <FaLightbulb className="text-yellow-500 mr-2 mt-1 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Action Items</h4>
                    <ul className="space-y-2">
                      {comprehensiveAnalytics.ai_insights.action_items?.map((action, index) => (
                        <li key={index} className="flex items-start">
                          <FaTarget className="text-blue-500 mr-2 mt-1 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Platform Activity */}
        {realTimeDashboard?.platform_activity && (
          <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
            <h3 className="text-xl font-semibold mb-4">📱 Platform Activity (Last 24 Hours)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {realTimeDashboard.platform_activity.map((platform) => (
                <div key={platform._id} className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className={`inline-block w-4 h-4 rounded-full mb-2 ${
                    platform._id === 'twitter' ? 'bg-blue-500' :
                    platform._id === 'facebook' ? 'bg-blue-700' :
                    platform._id === 'google' ? 'bg-red-500' :
                    platform._id === 'news' ? 'bg-gray-600' : 'bg-green-500'
                  }`}></div>
                  <div className="text-lg font-bold">{platform.count}</div>
                  <div className="text-sm text-gray-600 capitalize">{platform._id}</div>
                  <div className="text-xs text-gray-500">
                    {new Date(platform.latest).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Sentiment Timeline */}
        {realTimeDashboard?.sentiment_trend && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h3 className="text-xl font-semibold mb-4">💭 Live Sentiment Feed</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {realTimeDashboard.sentiment_trend.slice(0, 10).map((item, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className={`w-3 h-3 rounded-full mt-2 ${
                    item.sentiment_score > 0.1 ? 'bg-green-500' :
                    item.sentiment_score < -0.1 ? 'bg-red-500' : 'bg-gray-500'
                  }`}></div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium capitalize">{item.platform}</span>
                      <span className="text-xs text-gray-500">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{item.content}</p>
                    <div className="text-xs text-gray-500 mt-1">
                      Sentiment: {item.sentiment_score.toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const SocialMonitoringPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">🚨 Live Social Media Monitoring</h2>
          <p className="text-gray-600">Monitor what people are saying about your business across all social platforms in real-time</p>
        </div>

        {/* Monitoring Setup */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h3 className="text-xl font-semibold mb-4">Setup Monitoring</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <input
              type="text"
              placeholder="Business Name"
              value={newMonitoringRule.business_name}
              onChange={(e) => setNewMonitoringRule({...newMonitoringRule, business_name: e.target.value})}
              className="border rounded px-3 py-2"
            />
            <input
              type="text"
              placeholder="Keywords (comma-separated)"
              value={newMonitoringRule.keywords}
              onChange={(e) => setNewMonitoringRule({...newMonitoringRule, keywords: e.target.value})}
              className="border rounded px-3 py-2"
            />
            <input
              type="text"
              placeholder="@Mentions (comma-separated)"
              value={newMonitoringRule.mentions}
              onChange={(e) => setNewMonitoringRule({...newMonitoringRule, mentions: e.target.value})}
              className="border rounded px-3 py-2"
            />
            <input
              type="text"
              placeholder="#Hashtags (comma-separated)"
              value={newMonitoringRule.hashtags}
              onChange={(e) => setNewMonitoringRule({...newMonitoringRule, hashtags: e.target.value})}
              className="border rounded px-3 py-2"
            />
          </div>
          
          <div className="mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Platforms to Monitor:</p>
            <div className="flex flex-wrap gap-2">
              {['twitter', 'facebook', 'instagram', 'google', 'news'].map((platform) => (
                <label key={platform} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={newMonitoringRule.platforms.includes(platform)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setNewMonitoringRule({
                          ...newMonitoringRule,
                          platforms: [...newMonitoringRule.platforms, platform]
                        });
                      } else {
                        setNewMonitoringRule({
                          ...newMonitoringRule,
                          platforms: newMonitoringRule.platforms.filter(p => p !== platform)
                        });
                      }
                    }}
                    className="mr-2"
                  />
                  <span className="text-sm capitalize">{platform}</span>
                </label>
              ))}
            </div>
          </div>
          
          <button
            onClick={startSocialMonitoring}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            🚀 Start Live Monitoring
          </button>
        </div>

        {/* Alerts Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <FaBell className="mr-2 text-red-500" />
              Live Alerts ({socialAlerts.filter(a => !a.is_read).length})
            </h3>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {socialAlerts.slice(0, 10).map((alert) => (
                <div 
                  key={alert.id} 
                  className={`border-l-4 p-4 rounded ${
                    alert.priority === 'critical' ? 'border-red-500 bg-red-50' :
                    alert.priority === 'high' ? 'border-orange-500 bg-orange-50' :
                    alert.priority === 'medium' ? 'border-yellow-500 bg-yellow-50' :
                    'border-blue-500 bg-blue-50'
                  } ${!alert.is_read ? 'ring-2 ring-opacity-50' : 'opacity-75'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-sm">{alert.title}</h4>
                    <span className={`text-xs px-2 py-1 rounded ${
                      alert.priority === 'critical' ? 'bg-red-200 text-red-800' :
                      alert.priority === 'high' ? 'bg-orange-200 text-orange-800' :
                      alert.priority === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                      'bg-blue-200 text-blue-800'
                    }`}>
                      {alert.priority.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">{alert.description}</p>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>{new Date(alert.created_at).toLocaleString()}</span>
                    {!alert.is_read && (
                      <button
                        onClick={() => markAlertAsRead(alert.id)}
                        className="text-blue-600 hover:text-blue-800"
                      >
                        Mark as Read
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {socialAlerts.length === 0 && (
                <p className="text-gray-500 text-center">No alerts yet. Start monitoring to see live updates!</p>
              )}
            </div>
          </div>

          {/* Analytics Overview */}
          {socialAnalytics && (
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-xl font-semibold mb-4">📊 Sentiment Analytics</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                {socialAnalytics.sentiment_distribution?.map((item) => (
                  <div key={item._id} className="text-center">
                    <div className={`text-2xl font-bold ${
                      item._id === 'positive' ? 'text-green-600' :
                      item._id === 'negative' ? 'text-red-600' :
                      'text-gray-600'
                    }`}>
                      {item.count}
                    </div>
                    <div className="text-sm text-gray-600 capitalize">{item._id}</div>
                  </div>
                ))}
              </div>
              
              <div className="mt-4">
                <h4 className="font-medium mb-2">Platform Distribution</h4>
                {socialAnalytics.platform_distribution?.map((platform) => (
                  <div key={platform._id} className="flex justify-between items-center mb-1">
                    <span className="text-sm capitalize">{platform._id}</span>
                    <span className="text-sm font-medium">{platform.count} mentions</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Live Mentions Feed */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <FaTwitter className="mr-2 text-blue-500" />
            Live Mentions Feed
          </h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {socialMentions.map((mention) => (
              <div key={mention.id} className="border-b pb-4">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center">
                    <span className={`inline-block w-3 h-3 rounded-full mr-2 ${
                      mention.platform === 'twitter' ? 'bg-blue-500' :
                      mention.platform === 'facebook' ? 'bg-blue-700' :
                      mention.platform === 'instagram' ? 'bg-pink-500' :
                      mention.platform === 'news' ? 'bg-gray-600' :
                      'bg-green-500'
                    }`}></span>
                    <span className="font-semibold text-sm capitalize">{mention.platform}</span>
                    {mention.author_username && (
                      <span className="text-gray-600 text-sm ml-2">@{mention.author_username}</span>
                    )}
                  </div>
                  <div className="flex items-center">
                    <span className={`text-xs px-2 py-1 rounded ${
                      mention.sentiment_label === 'positive' ? 'bg-green-100 text-green-800' :
                      mention.sentiment_label === 'negative' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {mention.sentiment_label} ({mention.sentiment_score.toFixed(2)})
                    </span>
                  </div>
                </div>
                <p className="text-gray-800 mb-2">{mention.content}</p>
                <div className="flex justify-between items-center text-xs text-gray-500">
                  <span>{new Date(mention.detected_at).toLocaleString()}</span>
                  {mention.url && (
                    <a 
                      href={mention.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800"
                    >
                      View Original
                    </a>
                  )}
                </div>
              </div>
            ))}
            {socialMentions.length === 0 && (
              <p className="text-gray-500 text-center">No mentions detected yet. Start monitoring to see live updates!</p>
            )}
          </div>
        </div>

        {/* News Articles */}
        {newsArticles.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-6 mt-8">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <FaNewspaper className="mr-2 text-gray-600" />
              Industry News & Insights
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {newsArticles.slice(0, 6).map((article) => (
                <div key={article.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                  <h4 className="font-semibold text-sm mb-2 line-clamp-2">{article.title}</h4>
                  <p className="text-gray-600 text-xs mb-2 line-clamp-3">{article.content}</p>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>{article.source}</span>
                    <a 
                      href={article.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800"
                    >
                      Read More
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Other page components would go here (CompetitorsPage, ReportPage, etc.)
  // For brevity, I'm showing the key new pages

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'register':
        return <RegisterPage />;
      case 'marketplace':
        return <MarketplacePage />;
      case 'pricing':
        return <PricingPage />;
      case 'messages':
        return <MessagesPage />;
      case 'social-monitoring':
        return <SocialMonitoringPage />;
      case 'competitors':
        return <div className="p-8 text-center">Competitors page (existing implementation)</div>;
      case 'ultimate-report':
        return <div className="p-8 text-center">Report page (existing implementation)</div>;
      case 'dashboard':
        return <div className="p-8 text-center">Dashboard page (existing implementation)</div>;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="App">
      {renderCurrentPage()}
    </div>
  );
}

export default App;