import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { FaSearch, FaChartBar, FaUsers, FaDollarSign, FaStar, FaMapMarkerAlt, FaPhone, FaGlobe, FaArrowLeft, FaDownload, FaPrint, FaEye, FaBuilding, FaChartLine, FaShieldAlt, FaBrain } from 'react-icons/fa';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [searchLocation, setSearchLocation] = useState('');
  const [searchRadius, setSearchRadius] = useState(5);
  const [competitors, setCompetitors] = useState([]);
  const [selectedCompetitors, setSelectedCompetitors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [reports, setReports] = useState([]);
  const [subscriptionTiers, setSubscriptionTiers] = useState([]);
  const [marketInsights, setMarketInsights] = useState(null);

  useEffect(() => {
    fetchSubscriptionTiers();
    fetchReports();
  }, []);

  const fetchSubscriptionTiers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/subscription-tiers`);
      const data = await response.json();
      setSubscriptionTiers(data.tiers);
    } catch (error) {
      console.error('Error fetching subscription tiers:', error);
    }
  };

  const fetchReports = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reports`);
      const data = await response.json();
      setReports(data.reports);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const searchCompetitors = async () => {
    if (!searchLocation.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/search-competitors`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          location: searchLocation,
          radius: searchRadius,
          business_type: 'restaurant'
        }),
      });
      
      const data = await response.json();
      setCompetitors(data.competitors);
      setMarketInsights(data.market_insights);
      setCurrentPage('competitors');
    } catch (error) {
      console.error('Error searching competitors:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateComprehensiveReport = async () => {
    if (selectedCompetitors.length === 0) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-comprehensive-report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          competitor_ids: selectedCompetitors,
          location: searchLocation
        }),
      });
      
      const data = await response.json();
      setReport(data);
      setCurrentPage('comprehensive-report');
      fetchReports();
    } catch (error) {
      console.error('Error generating comprehensive report:', error);
    } finally {
      setLoading(false);
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
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <FaBrain className="text-3xl text-blue-600 mr-3" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">BizFizz</h1>
                <p className="text-sm text-gray-600">Advanced Competitive Intelligence Platform</p>
              </div>
            </div>
            <nav className="flex space-x-8">
              <button 
                onClick={() => setCurrentPage('home')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                Home
              </button>
              <button 
                onClick={() => setCurrentPage('dashboard')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                Dashboard
              </button>
              <button 
                onClick={() => setCurrentPage('pricing')}
                className="text-gray-700 hover:text-blue-600 font-medium transition-colors"
              >
                Pricing
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-6xl font-bold text-gray-900 mb-6">
              Know Your <span className="text-blue-600">Competition</span>
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Unlock the power of AI-driven competitive intelligence. Get real-time insights on competitor pricing, customer sentiment, foot traffic, and market positioning to dominate your local market.
            </p>
          </div>

          {/* Search Interface */}
          <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl p-8 mb-16">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-semibold text-gray-900 mb-2">Start Your Market Analysis</h3>
              <p className="text-gray-600">Enter your location to discover competitors and generate comprehensive intelligence reports</p>
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
                  autoComplete="address-level2"
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

          {/* AI Agents Showcase */}
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              5 AI Agents Working for You
            </h3>
            <p className="text-lg text-gray-600">
              Our specialized AI agents continuously analyze your competition across all key business dimensions
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "PriceWatch Agent",
                description: "Analyzes competitor pricing strategies, revenue estimates, and identifies pricing opportunities in your market",
                icon: <FaDollarSign className="text-3xl text-green-500" />,
                features: ["Dynamic pricing analysis", "Revenue benchmarking", "Price gap identification"]
              },
              {
                title: "Sentiment Agent",
                description: "Deep-dives into customer reviews and sentiment across all platforms to uncover satisfaction drivers",
                icon: <FaStar className="text-3xl text-yellow-500" />,
                features: ["Review sentiment analysis", "Customer pain points", "Satisfaction benchmarks"]
              },
              {
                title: "CrowdAnalyst Agent",
                description: "Estimates foot traffic patterns, customer behavior, and market capacity for strategic planning",
                icon: <FaUsers className="text-3xl text-purple-500" />,
                features: ["Traffic pattern analysis", "Customer behavior insights", "Market capacity assessment"]
              },
              {
                title: "Sentinel Agent",
                description: "Monitors competitive activities, marketing strategies, and emerging threats in real-time",
                icon: <FaShieldAlt className="text-3xl text-red-500" />,
                features: ["Competitive monitoring", "Threat assessment", "Marketing strategy analysis"]
              },
              {
                title: "Intelligence Synthesizer",
                description: "Combines all agent insights into actionable strategic recommendations and market entry strategies",
                icon: <FaBrain className="text-3xl text-blue-500" />,
                features: ["Strategic synthesis", "Market entry strategy", "Executive recommendations"]
              }
            ].map((agent, index) => (
              <div key={index} className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center mb-4">
                  {agent.icon}
                  <h4 className="text-xl font-semibold ml-3">{agent.title}</h4>
                </div>
                <p className="text-gray-600 mb-4">{agent.description}</p>
                <ul className="space-y-2">
                  {agent.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center text-sm text-gray-500">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              Comprehensive Business Intelligence
            </h3>
            <p className="text-lg text-gray-600">
              Everything you need to understand and outcompete your local market
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <FaChartBar className="text-4xl text-blue-500 mx-auto mb-4" />
              <h4 className="text-xl font-semibold mb-2">Visual Analytics</h4>
              <p className="text-gray-600">Interactive charts and graphs showing market trends, competitor performance, and revenue analysis</p>
            </div>
            <div className="text-center">
              <FaTrendingUp className="text-4xl text-green-500 mx-auto mb-4" />
              <h4 className="text-xl font-semibold mb-2">Real-time Monitoring</h4>
              <p className="text-gray-600">Continuous monitoring of competitor activities, pricing changes, and market shifts</p>
            </div>
            <div className="text-center">
              <FaBuilding className="text-4xl text-purple-500 mx-auto mb-4" />
              <h4 className="text-xl font-semibold mb-2">Market Intelligence</h4>
              <p className="text-gray-600">Deep market analysis with strategic recommendations for business growth and positioning</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );

  const CompetitorsPage = () => {
    const competitorChartData = competitors.map(comp => ({
      name: comp.name.length > 15 ? comp.name.substring(0, 15) + '...' : comp.name,
      rating: comp.rating || 0,
      reviews: comp.review_count || 0,
      price: comp.price_level || 1,
      traffic: comp.business_metrics?.estimated_daily_traffic || 50
    }));

    const priceDistribution = competitors.reduce((acc, comp) => {
      const price = '$'.repeat(comp.price_level || 1);
      acc[price] = (acc[price] || 0) + 1;
      return acc;
    }, {});

    const priceChartData = Object.entries(priceDistribution).map(([price, count]) => ({
      price,
      count
    }));

    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-8">
            <button
              onClick={() => setCurrentPage('home')}
              className="flex items-center text-blue-600 hover:text-blue-800 mb-4 transition-colors"
            >
              <FaArrowLeft className="mr-2" />
              Back to Search
            </button>
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                  Competitors in {searchLocation}
                </h2>
                <p className="text-gray-600">
                  Found {competitors.length} restaurants within {searchRadius} miles
                </p>
              </div>
              {marketInsights && (
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <h3 className="font-semibold text-gray-900 mb-2">Market Overview</h3>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>Avg Rating: {marketInsights.average_rating?.toFixed(1)}★</p>
                    <p>Price Range: {marketInsights.price_range}</p>
                    <p>Total Reviews: {marketInsights.total_reviews?.toLocaleString()}</p>
                    <p>Market: {marketInsights.market_saturation}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Market Analytics Dashboard */}
          {competitors.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Competitor Ratings vs Reviews</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={competitorChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="rating" fill="#8884d8" name="Rating" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Price Level Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={priceChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({price, count}) => `${price} (${count})`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {priceChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Competitor Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {competitors.map((competitor) => (
              <div
                key={competitor.id}
                className={`bg-white p-6 rounded-lg shadow-sm border-2 cursor-pointer transition-all hover:shadow-md ${
                  selectedCompetitors.includes(competitor.id)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => toggleCompetitorSelection(competitor.id)}
              >
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 line-clamp-2">
                    {competitor.name}
                  </h3>
                  <div className="flex items-center ml-2">
                    <FaStar className="text-yellow-500 mr-1" />
                    <span className="text-sm font-medium text-gray-700">{competitor.rating || 'N/A'}</span>
                  </div>
                </div>
                
                <div className="space-y-3 mb-4">
                  <div className="flex items-start">
                    <FaMapMarkerAlt className="text-gray-500 text-sm mr-2 mt-0.5" />
                    <p className="text-sm text-gray-600 line-clamp-2">{competitor.address}</p>
                  </div>
                  
                  {competitor.phone && (
                    <div className="flex items-center">
                      <FaPhone className="text-gray-500 text-sm mr-2" />
                      <p className="text-sm text-gray-600">{competitor.phone}</p>
                    </div>
                  )}
                  
                  {competitor.categories && competitor.categories.length > 0 && (
                    <div className="flex items-start">
                      <span className="text-gray-500 text-sm mr-2">🍽️</span>
                      <div className="flex flex-wrap gap-1">
                        {competitor.categories.slice(0, 3).map((category, index) => (
                          <span 
                            key={index}
                            className="bg-gray-100 text-gray-700 px-2 py-1 rounded-full text-xs"
                          >
                            {category}
                          </span>
                        ))}
                        {competitor.categories.length > 3 && (
                          <span className="text-gray-500 text-xs">+{competitor.categories.length - 3} more</span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Business Metrics */}
                  {competitor.business_metrics && (
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h4 className="text-xs font-semibold text-gray-700 mb-2">Business Insights</h4>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500">Daily Traffic:</span>
                          <span className="font-medium ml-1">{competitor.business_metrics.estimated_daily_traffic}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Avg Check:</span>
                          <span className="font-medium ml-1">${competitor.business_metrics.avg_check_estimate}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Position:</span>
                          <span className="font-medium ml-1">{competitor.business_metrics.market_position}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Strength:</span>
                          <span className="font-medium ml-1">{competitor.business_metrics.competitive_strength}/100</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex justify-between items-center text-sm border-t pt-3">
                  <div className="flex items-center space-x-4">
                    <span className="text-gray-500 flex items-center">
                      <FaUsers className="mr-1" />
                      {competitor.review_count || 0} reviews
                    </span>
                    <span className="text-gray-500 flex items-center">
                      <FaDollarSign className="mr-1" />
                      {'$'.repeat(competitor.price_level || 1)}
                    </span>
                  </div>
                  
                  {selectedCompetitors.includes(competitor.id) && (
                    <span className="text-blue-600 font-medium">✓ Selected</span>
                  )}
                </div>
                
                {competitor.website && (
                  <div className="mt-3 pt-3 border-t">
                    <a 
                      href={competitor.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 text-xs flex items-center"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <FaGlobe className="mr-1" />
                      View on Yelp
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Generate Report Section */}
          {selectedCompetitors.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold mb-2">
                    Generate Comprehensive Intelligence Report
                  </h3>
                  <p className="text-gray-600">
                    Selected {selectedCompetitors.length} competitors for analysis with all 5 AI agents
                  </p>
                </div>
                <button
                  onClick={generateComprehensiveReport}
                  disabled={loading}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3 rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 font-semibold flex items-center"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <FaChartBar className="mr-2" />
                      Generate Report
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const ComprehensiveReportPage = () => {
    if (!report) return null;

    // Prepare chart data
    const competitorMetrics = report.competitors?.map(comp => ({
      name: comp.name.length > 10 ? comp.name.substring(0, 10) + '...' : comp.name,
      rating: comp.rating || 0,
      strength: comp.business_metrics?.competitive_strength || 50,
      revenue: comp.business_metrics?.estimated_monthly_revenue || 50000,
      traffic: comp.business_metrics?.estimated_daily_traffic || 100
    })) || [];

    const radarData = report.competitors?.map(comp => ({
      competitor: comp.name.substring(0, 10),
      rating: (comp.rating || 0) * 20,
      reviews: Math.min((comp.review_count || 0) / 100, 10) * 10,
      price: (comp.price_level || 1) * 25,
      traffic: Math.min((comp.business_metrics?.estimated_daily_traffic || 0) / 5, 100)
    })) || [];

    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-8">
            <button
              onClick={() => setCurrentPage('competitors')}
              className="flex items-center text-blue-600 hover:text-blue-800 mb-4 transition-colors"
            >
              <FaArrowLeft className="mr-2" />
              Back to Competitors
            </button>
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                  Comprehensive Intelligence Report
                </h2>
                <p className="text-gray-600">
                  AI-Powered Market Analysis for {report.location} • Generated {new Date(report.report_date).toLocaleDateString()}
                </p>
              </div>
              <div className="flex space-x-3">
                <button className="flex items-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                  <FaDownload className="mr-2" />
                  Download PDF
                </button>
                <button className="flex items-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                  <FaPrint className="mr-2" />
                  Print Report
                </button>
              </div>
            </div>
          </div>

          {/* Executive Summary */}
          <div className="bg-white p-6 rounded-lg shadow-sm mb-8">
            <h3 className="text-xl font-semibold mb-4 text-gray-900">Executive Summary</h3>
            <p className="text-gray-700 leading-relaxed">{report.executive_summary}</p>
            
            {report.market_metrics && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900">Total Market Revenue</h4>
                  <p className="text-2xl font-bold text-blue-600">${(report.market_metrics.total_market_revenue / 1000000).toFixed(1)}M</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900">Avg Competitor Strength</h4>
                  <p className="text-2xl font-bold text-green-600">{report.market_metrics.average_competitor_strength?.toFixed(0)}/100</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900">Growth Potential</h4>
                  <p className="text-2xl font-bold text-purple-600">{report.market_metrics.market_growth_potential}</p>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900">Competitors Analyzed</h4>
                  <p className="text-2xl font-bold text-orange-600">{report.total_competitors}</p>
                </div>
              </div>
            )}
          </div>

          {/* Visual Analytics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Competitor Strength Comparison */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-lg font-semibold mb-4">Competitive Strength Analysis</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={competitorMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip formatter={(value, name) => [value, name === 'strength' ? 'Competitive Strength' : name]} />
                  <Bar dataKey="strength" fill="#8884d8" name="Strength Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Revenue vs Traffic */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-lg font-semibold mb-4">Revenue vs Daily Traffic</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={competitorMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line yAxisId="left" type="monotone" dataKey="revenue" stroke="#8884d8" name="Monthly Revenue ($)" />
                  <Line yAxisId="right" type="monotone" dataKey="traffic" stroke="#82ca9d" name="Daily Traffic" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Agent Reports */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* PriceWatch Analysis */}
            {report.pricewatch_analysis && (
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <FaDollarSign className="text-green-500 text-xl mr-2" />
                  <h3 className="text-lg font-semibold">PriceWatch Agent Report</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Market Overview</h4>
                    <p className="text-sm text-gray-600">{report.pricewatch_analysis.market_overview}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Pricing Opportunities</h4>
                    <ul className="space-y-1">
                      {report.pricewatch_analysis.pricing_opportunities?.map((opp, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="text-green-500 mr-2">•</span>
                          {opp}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Sentiment Analysis */}
            {report.sentiment_analysis && (
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <FaStar className="text-yellow-500 text-xl mr-2" />
                  <h3 className="text-lg font-semibold">Sentiment Agent Report</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Customer Pain Points</h4>
                    <ul className="space-y-1">
                      {report.sentiment_analysis.customer_pain_points?.map((point, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="text-red-500 mr-2">•</span>
                          {point}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Satisfaction Drivers</h4>
                    <ul className="space-y-1">
                      {report.sentiment_analysis.satisfaction_drivers?.map((driver, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="text-green-500 mr-2">•</span>
                          {driver}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* CrowdAnalyst Report */}
            {report.crowd_analysis && (
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <FaUsers className="text-purple-500 text-xl mr-2" />
                  <h3 className="text-lg font-semibold">CrowdAnalyst Agent Report</h3>
                </div>
                <div className="space-y-4">
                  {report.crowd_analysis.market_capacity && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Market Capacity</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Daily Customers:</span>
                          <span className="font-medium ml-2">{report.crowd_analysis.market_capacity.total_daily_customers}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Market Saturation:</span>
                          <span className="font-medium ml-2">{report.crowd_analysis.market_capacity.market_saturation}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Avg per Restaurant:</span>
                          <span className="font-medium ml-2">{report.crowd_analysis.market_capacity.average_per_restaurant}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Revenue/Customer:</span>
                          <span className="font-medium ml-2">{report.crowd_analysis.market_capacity.revenue_per_customer}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Sentinel Report */}
            {report.sentinel_analysis && (
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex items-center mb-4">
                  <FaShieldAlt className="text-red-500 text-xl mr-2" />
                  <h3 className="text-lg font-semibold">Sentinel Agent Report</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Threat Assessment</h4>
                    <p className="text-sm text-gray-600">{report.sentinel_analysis.threat_assessment}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Emerging Trends</h4>
                    <ul className="space-y-1">
                      {report.sentinel_analysis.emerging_trends?.map((trend, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="text-blue-500 mr-2">•</span>
                          {trend}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Strategic Recommendations */}
          {report.market_synthesis && (
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center mb-4">
                <FaBrain className="text-blue-500 text-xl mr-2" />
                <h3 className="text-lg font-semibold">Intelligence Synthesizer Report</h3>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Strategic Opportunities</h4>
                  <ul className="space-y-2">
                    {report.market_synthesis.strategic_opportunities?.map((opp, index) => (
                      <li key={index} className="text-sm text-gray-700 flex items-start">
                        <span className="text-green-500 mr-2 mt-1">✓</span>
                        {opp}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Success Metrics</h4>
                  <ul className="space-y-2">
                    {report.market_synthesis.success_metrics?.map((metric, index) => (
                      <li key={index} className="text-sm text-gray-700 flex items-start">
                        <span className="text-blue-500 mr-2 mt-1">📊</span>
                        {metric}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              {report.market_synthesis.market_entry_strategy && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Recommended Market Entry Strategy</h4>
                  <p className="text-sm text-blue-800">{report.market_synthesis.market_entry_strategy}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const DashboardPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Intelligence Dashboard</h2>
          <p className="text-gray-600">Your competitive intelligence reports and analytics</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <FaChartBar className="text-blue-500 text-2xl mr-3" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Total Reports</h3>
                <p className="text-3xl font-bold text-blue-600">{reports.length}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <FaBuilding className="text-green-500 text-2xl mr-3" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Competitors Analyzed</h3>
                <p className="text-3xl font-bold text-green-600">
                  {reports.reduce((sum, report) => sum + (report.total_competitors || 0), 0)}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <FaTrendingUp className="text-purple-500 text-2xl mr-3" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">This Month</h3>
                <p className="text-3xl font-bold text-purple-600">{reports.length}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center">
              <FaUsers className="text-orange-500 text-2xl mr-3" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Plan</h3>
                <p className="text-lg font-medium text-gray-700">Free</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-900">Recent Intelligence Reports</h3>
          </div>
          <div className="divide-y divide-gray-200">
            {reports.map((report) => (
              <div key={report.id} className="p-6 hover:bg-gray-50 transition-colors">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-gray-900 flex items-center">
                      {report.location}
                      {report.report_type === 'comprehensive_intelligence' && (
                        <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                          Comprehensive
                        </span>
                      )}
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">
                      {report.total_competitors} competitors analyzed
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(report.report_date).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setReport(report);
                      setCurrentPage(report.report_type === 'comprehensive_intelligence' ? 'comprehensive-report' : 'report');
                    }}
                    className="flex items-center text-blue-600 hover:text-blue-800 text-sm font-medium transition-colors"
                  >
                    <FaEye className="mr-1" />
                    View Report
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const PricingPage = () => (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Choose Your Intelligence Level
          </h2>
          <p className="text-xl text-gray-600">
            Get the competitive intelligence you need to dominate your market
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {subscriptionTiers.map((tier, index) => (
            <div
              key={tier.name}
              className={`bg-white rounded-xl shadow-lg p-8 ${
                index === 1 ? 'border-2 border-blue-500 relative transform scale-105' : 'border border-gray-200'
              }`}
            >
              {index === 1 && (
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
                
                {/* AI Agents */}
                <div className="mb-6">
                  <h4 className="font-semibold text-gray-900 mb-2">AI Agents Included</h4>
                  <div className="flex flex-wrap gap-1 justify-center">
                    {tier.ai_agents?.map((agent, agentIndex) => (
                      <span key={agentIndex} className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                        {agent}
                      </span>
                    ))}
                  </div>
                </div>

                <ul className="text-left space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start">
                      <span className="text-green-500 mr-2 mt-1">✓</span>
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <button
                  className={`w-full py-3 rounded-lg font-semibold transition-all ${
                    index === 1
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

        {/* Feature Comparison */}
        <div className="mt-16">
          <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">Feature Comparison</h3>
          <div className="bg-white rounded-lg shadow-sm overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature</th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Free</th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Professional</th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Enterprise</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">AI-Powered Analysis</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">Basic</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓ Advanced</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓ Premium</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Real-time Data</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Visual Reports</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">Basic</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓ Advanced</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓ Premium</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">API Access</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">-</td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600">✓</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'competitors':
        return <CompetitorsPage />;
      case 'comprehensive-report':
        return <ComprehensiveReportPage />;
      case 'dashboard':
        return <DashboardPage />;
      case 'pricing':
        return <PricingPage />;
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