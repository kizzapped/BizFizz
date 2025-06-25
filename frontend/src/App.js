import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

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
      setCurrentPage('competitors');
    } catch (error) {
      console.error('Error searching competitors:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    if (selectedCompetitors.length === 0) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-report`, {
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
      setCurrentPage('report');
      fetchReports(); // Refresh reports list
    } catch (error) {
      console.error('Error generating report:', error);
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">BizFizz</h1>
              <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">BETA</span>
            </div>
            <nav className="flex space-x-8">
              <button 
                onClick={() => setCurrentPage('home')}
                className="text-gray-700 hover:text-blue-600 font-medium"
              >
                Home
              </button>
              <button 
                onClick={() => setCurrentPage('dashboard')}
                className="text-gray-700 hover:text-blue-600 font-medium"
              >
                Dashboard
              </button>
              <button 
                onClick={() => setCurrentPage('pricing')}
                className="text-gray-700 hover:text-blue-600 font-medium"
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-5xl font-bold text-gray-900 mb-6">
                Know Your <span className="text-blue-600">Competition</span>
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Get real-time competitive intelligence for your restaurant. Track competitor pricing, reviews, and customer sentiment in your area.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 mb-8">
                <input
                  type="text"
                  placeholder="Enter your location (zip code, city, address)"
                  value={searchLocation}
                  onChange={(e) => setSearchLocation(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchCompetitors()}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base"
                  autoComplete="address-level2"
                />
                <select
                  value={searchRadius}
                  onChange={(e) => setSearchRadius(Number(e.target.value))}
                  className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent min-w-[120px]"
                >
                  <option value={1}>1 mile</option>
                  <option value={3}>3 miles</option>
                  <option value={5}>5 miles</option>
                  <option value={10}>10 miles</option>
                  <option value={15}>15 miles</option>
                </select>
              </div>
              <button
                onClick={searchCompetitors}
                disabled={loading || !searchLocation.trim()}
                className="w-full sm:w-auto bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
              >
                {loading ? 'Searching...' : 'Find Competitors'}
              </button>
            </div>
            <div className="lg:justify-self-end">
              <img
                src="https://images.unsplash.com/photo-1551288049-bebda4e38f71"
                alt="Business Analytics Dashboard"
                className="rounded-lg shadow-xl w-full max-w-md"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              Powerful AI Agents Working for You
            </h3>
            <p className="text-lg text-gray-600">
              Our specialized AI agents continuously monitor your competition
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "ReviewMiner Agent",
                description: "Analyzes competitor reviews and sentiment across all platforms",
                icon: "📊"
              },
              {
                title: "PriceWatch Agent",
                description: "Monitors menu changes and pricing strategies",
                icon: "💰"
              },
              {
                title: "Sentinel Agent",
                description: "Tracks competitor ads and promotional activities",
                icon: "🎯"
              },
              {
                title: "CrowdAnalyst Agent",
                description: "Estimates foot traffic and customer behavior patterns",
                icon: "👥"
              },
              {
                title: "GeoScout Agent",
                description: "Detects physical changes and new openings",
                icon: "🗺️"
              },
              {
                title: "InsightSynthesizer",
                description: "Generates actionable reports and recommendations",
                icon: "🧠"
              }
            ].map((feature, index) => (
              <div key={index} className="bg-gray-50 p-6 rounded-lg">
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h4 className="text-xl font-semibold mb-2">{feature.title}</h4>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );

  const CompetitorsPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => setCurrentPage('home')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Search
          </button>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Competitors in {searchLocation}
          </h2>
          <p className="text-gray-600">
            Found {competitors.length} restaurants within {searchRadius} miles
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {competitors.map((competitor) => (
            <div
              key={competitor.id}
              className={`bg-white p-6 rounded-lg shadow-sm border-2 cursor-pointer transition-all ${
                selectedCompetitors.includes(competitor.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => toggleCompetitorSelection(competitor.id)}
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  {competitor.name}
                </h3>
                <div className="flex items-center">
                  <span className="text-yellow-500 mr-1">★</span>
                  <span className="text-sm text-gray-600">{competitor.rating}</span>
                </div>
              </div>
              <p className="text-sm text-gray-600 mb-2">{competitor.address}</p>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">
                  {competitor.review_count} reviews
                </span>
                <span className="text-gray-500">
                  {'$'.repeat(competitor.price_level)}
                </span>
              </div>
              {competitor.website && (
                <p className="text-xs text-blue-600 mt-2">{competitor.website}</p>
              )}
            </div>
          ))}
        </div>

        {selectedCompetitors.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold mb-4">
              Selected Competitors ({selectedCompetitors.length})
            </h3>
            <button
              onClick={generateReport}
              disabled={loading}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Generating Report...' : 'Generate Intelligence Report'}
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const ReportPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <button
            onClick={() => setCurrentPage('competitors')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Competitors
          </button>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Competitive Intelligence Report
          </h2>
          <p className="text-gray-600">
            Generated on {new Date(report?.report_date).toLocaleDateString()}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Report */}
          <div className="lg:col-span-2 space-y-6">
            {/* Insights Section */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">
                Key Insights
              </h3>
              <ul className="space-y-3">
                {report?.insights?.map((insight, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    <span className="text-gray-700">{insight}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendations Section */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">
                Recommendations
              </h3>
              <ul className="space-y-3">
                {report?.recommendations?.map((rec, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-green-500 mr-2">✓</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Competitors Overview */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">
                Analyzed Competitors
              </h3>
              <div className="space-y-4">
                {report?.competitors?.map((competitor) => (
                  <div key={competitor.id} className="border-l-4 border-blue-500 pl-4">
                    <h4 className="font-medium text-gray-900">{competitor.name}</h4>
                    <p className="text-sm text-gray-600">{competitor.address}</p>
                    <div className="flex items-center mt-1 text-sm">
                      <span className="text-yellow-500 mr-1">★</span>
                      <span className="text-gray-600 mr-4">{competitor.rating}</span>
                      <span className="text-gray-600">{competitor.review_count} reviews</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h3 className="text-lg font-semibold mb-4 text-gray-900">
                Report Summary
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Location:</span>
                  <span className="font-medium">{report?.search_location}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Competitors:</span>
                  <span className="font-medium">{report?.total_competitors}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Generated:</span>
                  <span className="font-medium">
                    {new Date(report?.report_date).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-2 text-blue-900">
                Want More Insights?
              </h3>
              <p className="text-blue-800 text-sm mb-4">
                Upgrade to Premium for real-time monitoring and deeper analytics.
              </p>
              <button
                onClick={() => setCurrentPage('pricing')}
                className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
              >
                View Pricing
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const DashboardPage = () => (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h2>
          <p className="text-gray-600">Your recent competitive intelligence reports</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Total Reports</h3>
            <p className="text-3xl font-bold text-blue-600">{reports.length}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Competitors Analyzed</h3>
            <p className="text-3xl font-bold text-green-600">
              {reports.reduce((sum, report) => sum + (report.total_competitors || 0), 0)}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">This Month</h3>
            <p className="text-3xl font-bold text-purple-600">{reports.length}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Plan</h3>
            <p className="text-lg font-medium text-gray-700">Free</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-900">Recent Reports</h3>
          </div>
          <div className="divide-y divide-gray-200">
            {reports.map((report) => (
              <div key={report.id} className="p-6 hover:bg-gray-50">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-gray-900">{report.search_location}</h4>
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
                      setCurrentPage('report');
                    }}
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  >
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
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Choose Your Plan
          </h2>
          <p className="text-lg text-gray-600">
            Get the competitive intelligence you need to stay ahead
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {subscriptionTiers.map((tier, index) => (
            <div
              key={tier.name}
              className={`bg-white rounded-lg shadow-sm p-8 ${
                index === 1 ? 'border-2 border-blue-500 relative' : 'border border-gray-200'
              }`}
            >
              {index === 1 && (
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <span className="bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
              )}
              <div className="text-center">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">{tier.name}</h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold text-gray-900">${tier.price}</span>
                  <span className="text-gray-600">/month</span>
                </div>
                <ul className="text-left space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                <button
                  className={`w-full py-3 rounded-lg font-semibold ${
                    index === 1
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  }`}
                >
                  {tier.price === 0 ? 'Get Started' : 'Start Free Trial'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'competitors':
        return <CompetitorsPage />;
      case 'report':
        return <ReportPage />;
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