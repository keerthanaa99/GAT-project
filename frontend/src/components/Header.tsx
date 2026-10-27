import React from 'react';
import type { ModelInfoResponse } from '../types/api';
import { ShieldCheck, Info } from 'lucide-react';

interface HeaderProps {
  modelInfo: ModelInfoResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ modelInfo }) => {
  return (
    <header style={{
      padding: '20px 32px',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      backgroundColor: 'rgba(13, 21, 39, 0.7)',
      backdropFilter: 'blur(8px)',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.02em' }}>
            Traffic Analysis & Intrusion Detection
          </h2>
          <span className="badge" style={{ backgroundColor: 'rgba(20, 184, 166, 0.15)', color: '#2dd4bf', border: '1px solid rgba(20, 184, 166, 0.3)' }}>
            Stage 1 Foundation
          </span>
        </div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Statistical Flow-Similarity Graph Inference using PyTorch Geometric
        </p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {modelInfo ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            backgroundColor: 'var(--bg-card)',
            padding: '8px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.8rem'
          }}>
            <ShieldCheck size={16} color="var(--accent-teal)" />
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Architecture: </span>
              <strong style={{ color: '#f8fafc' }}>{modelInfo.architecture}</strong>
              <span style={{ margin: '0 8px', color: 'var(--border-subtle)' }}>|</span>
              <span style={{ color: 'var(--text-muted)' }}>Features: </span>
              <strong style={{ color: 'var(--accent-teal)' }}>{modelInfo.expected_feature_count}</strong>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <Info size={14} />
            <span>Connecting to FastAPI Backend...</span>
          </div>
        )}
      </div>
    </header>
  );
};
