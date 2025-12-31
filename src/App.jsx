import { useState } from 'react'

function App() {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analyze', { method: 'POST' });
      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error("Error analyzing:", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial', backgroundColor: '#1a1a1a', color: 'white', minHeight: '100vh' }}>
      <h1>🚧 AI Hazard Detection System</h1>
      
      <div style={{ display: 'flex', gap: '20px' }}>
        {/* Left: Video Feed */}
        <div style={{ border: '2px solid #333', borderRadius: '10px', overflow: 'hidden' }}>
          <img 
            src="http://localhost:8000/video_feed" 
            alt="Live Feed" 
            style={{ width: '640px', height: '480px', display: 'block' }} 
          />
        </div>

        {/* Right: Dashboard Panel */}
        <div style={{ flex: 1, padding: '20px', backgroundColor: '#2a2a2a', borderRadius: '10px' }}>
          <h2>Live Analysis</h2>
          
          <button 
            onClick={handleAnalyze} 
            disabled={loading}
            style={{
              padding: '15px 30px', fontSize: '18px', cursor: 'pointer',
              backgroundColor: loading ? '#555' : '#007bff', color: 'white', border: 'none', borderRadius: '5px'
            }}
          >
            {loading ? "Analyzing..." : "🔍 SCAN HAZARD"}
          </button>

          {analysis && (
            <div style={{ marginTop: '30px' }}>
              {/* Risk Meter */}
              <h3>Risk Assessment</h3>
              <div style={{ height: '30px', width: '100%', backgroundColor: '#444', borderRadius: '15px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${analysis.risk_score * 100}%`,
                  backgroundColor: analysis.risk_score > 0.5 ? '#ff4444' : '#00C851',
                  transition: 'width 0.5s'
                }}></div>
              </div>
              <p style={{ fontSize: '24px', fontWeight: 'bold', marginTop: '10px' }}>
                {(analysis.risk_score * 100).toFixed(1)}% Probability of Severity
              </p>

              {/* Details Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '20px' }}>
                <div style={cardStyle}>
                  <small>Weather</small>
                  <div>{analysis.scene_details.Weather || "Unknown"}</div>
                </div>
                <div style={cardStyle}>
                  <small>Road Type</small>
                  <div>{analysis.scene_details.RoadType || "Unknown"}</div>
                </div>
                <div style={cardStyle}>
                  <small>Lighting</small>
                  <div>{analysis.scene_details.Lighting || "Unknown"}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const cardStyle = {
  backgroundColor: '#333', padding: '15px', borderRadius: '8px'
}

export default App