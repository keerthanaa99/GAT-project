import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { TrafficAnalysis } from './components/TrafficAnalysis';
import type { HealthResponse, ModelInfoResponse } from './types/api';
import { fetchHealth, fetchModelInfo } from './services/api';
import { Info } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('analysis');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadInitialData() {
      setHealthLoading(true);
      try {
        const [hData, mData] = await Promise.all([
          fetchHealth().catch(err => {
            console.error('Health fetch error:', err);
            return null;
          }),
          fetchModelInfo().catch(err => {
            console.error('Model info fetch error:', err);
            return null;
          }),
        ]);
        setHealth(hData);
        setModelInfo(mData);
      } finally {
        setHealthLoading(false);
      }
    }

    loadInitialData();
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-primary)' }}>
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        health={health}
        healthLoading={healthLoading}
      />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header modelInfo={modelInfo} />

        <main style={{ padding: '28px 32px', flex: 1, maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
          {currentTab === 'analysis' && <TrafficAnalysis />}

          {currentTab !== 'analysis' && (
            <div className="card" style={{ textAlign: 'center', padding: '60px 20px' }}>
              <div style={{
                background: 'rgba(20, 184, 166, 0.1)',
                padding: '16px',
                borderRadius: '50%',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px'
              }}>
                <Info size={32} color="var(--accent-teal)" />
              </div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: '#f8fafc', marginBottom: '8px' }}>
                Stage 1 Initial Foundation
              </h3>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto' }}>
                The <strong>{currentTab.toUpperCase()}</strong> view is scheduled for implementation in subsequent stages.
                Please switch to <strong>Traffic Analysis</strong> to run CSV flow validation and GAT inference.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
