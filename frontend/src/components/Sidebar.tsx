import React from 'react';
import { 
  ShieldAlert, 
  Activity, 
  Share2, 
  AlertTriangle, 
  BarChart3, 
  CheckCircle2, 
  XCircle, 
  Cpu
} from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface SidebarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onTabChange,
  health,
  healthLoading
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Activity, placeholder: true },
    { id: 'analysis', label: 'Traffic Analysis', icon: ShieldAlert, placeholder: false },
    { id: 'graph', label: 'Network Graph', icon: Share2, placeholder: true },
    { id: 'alerts', label: 'Explainable Alerts', icon: AlertTriangle, placeholder: true },
    { id: 'evaluation', label: 'Model Evaluation', icon: BarChart3, placeholder: true },
  ];

  const isHealthy = health?.status === 'ok';

  return (
    <aside style={{
      width: '260px',
      backgroundColor: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'sticky',
      top: 0,
      padding: '24px 16px',
      zIndex: 20
    }}>
      {/* Brand Header */}
      <div style={{ paddingBottom: '24px', borderBottom: '1px solid var(--border-subtle)', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%)',
            borderRadius: '10px',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(20, 184, 166, 0.3)'
          }}>
            <Cpu size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#f8fafc' }}>
              GAT-XIDS
            </h1>
            <span style={{ fontSize: '0.72rem', color: 'var(--accent-teal)', fontWeight: 600, textTransform: 'uppercase' }}>
              IoMT Intrusion Detection
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
        <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, paddingLeft: '8px', marginBottom: '4px' }}>
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                padding: '10px 12px',
                borderRadius: '8px',
                border: 'none',
                background: isActive ? 'rgba(20, 184, 166, 0.15)' : 'transparent',
                color: isActive ? 'var(--accent-teal)' : 'var(--text-secondary)',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.88rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                textAlign: 'left'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Icon size={18} color={isActive ? 'var(--accent-teal)' : 'var(--text-muted)'} />
                <span>{item.label}</span>
              </div>
              {item.placeholder && (
                <span style={{
                  fontSize: '0.65rem',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: 'rgba(100, 116, 139, 0.2)',
                  color: 'var(--text-muted)'
                }}>
                  Stage 2
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Live Health Status Widget */}
      <div style={{
        backgroundColor: 'rgba(15, 23, 42, 0.8)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '10px',
        padding: '14px',
        marginTop: 'auto'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            System Status
          </span>
          {healthLoading ? (
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Checking...</span>
          ) : isHealthy ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-emerald)', fontSize: '0.72rem', fontWeight: 600 }}>
              <CheckCircle2 size={13} />
              <span>Ready</span>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-red)', fontSize: '0.72rem', fontWeight: 600 }}>
              <XCircle size={13} />
              <span>Degraded</span>
            </div>
          )}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <div>Model Checkpoint: <strong style={{ color: health?.model_ready ? '#34d399' : '#f87171' }}>{health?.model_ready ? 'Loaded' : 'Missing'}</strong></div>
          <div>Preprocessing: <strong style={{ color: health?.preprocessing_ready ? '#34d399' : '#f87171' }}>{health?.preprocessing_ready ? 'Available' : 'Missing'}</strong></div>
        </div>
      </div>
    </aside>
  );
};
