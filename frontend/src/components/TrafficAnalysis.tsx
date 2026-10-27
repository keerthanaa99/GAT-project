import React, { useState, useRef } from 'react';
import { 
  Upload, 
  FileText, 
  AlertCircle, 
  Loader2, 
  Network, 
  Activity, 
  Filter, 
  ChevronLeft, 
  ChevronRight,
  Info,
  Download
} from 'lucide-react';
import type { AnalyzeCSVResponse, FlowPredictionItem } from '../types/api';
import { analyzeTrafficCSV } from '../services/api';

const CATEGORY_BADGES: Record<string, string> = {
  'Benign': 'badge-benign',
  'ARP Spoofing': 'badge-arp',
  'MQTT attacks': 'badge-mqtt',
  'Reconnaissance': 'badge-recon',
  'TCP/IP DoS/DDoS': 'badge-dos',
};

export const TrafficAnalysis: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalyzeCSVResponse | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 15;

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError(`Selected file "${file.name}" is not a CSV. Only .csv files are supported.`);
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError(`Selected file "${file.name}" is not a CSV. Only .csv files are supported.`);
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a compatible CICIoMT2024 network flow CSV file.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await analyzeTrafficCSV(selectedFile, 2000);
      setResults(response);
      setCurrentPage(1);
    } catch (err: any) {
      setError(err.message || 'An error occurred during traffic analysis.');
      setResults(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadPredictionsCSV = () => {
    if (!results) return;

    const headers = [
      'Flow_Index',
      'Predicted_Category',
      'Predicted_Class_ID',
      'Confidence_Percent',
      'Prob_Benign',
      'Prob_ARP_Spoofing',
      'Prob_MQTT_attacks',
      'Prob_Reconnaissance',
      'Prob_TCP_IP_DoS_DDoS'
    ];

    const rows = results.predictions.map(p => [
      p.flow_index,
      `"${p.predicted_category}"`,
      p.predicted_class_id,
      (p.confidence * 100).toFixed(2),
      (p.probabilities['Benign'] ?? 0).toFixed(4),
      (p.probabilities['ARP Spoofing'] ?? 0).toFixed(4),
      (p.probabilities['MQTT attacks'] ?? 0).toFixed(4),
      (p.probabilities['Reconnaissance'] ?? 0).toFixed(4),
      (p.probabilities['TCP/IP DoS/DDoS'] ?? 0).toFixed(4),
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const cleanName = results.filename.replace(/\.csv$/i, '');
    link.setAttribute('download', `${cleanName}_gat_predictions.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Filtered & Paginated records
  const filteredPredictions = results
    ? results.predictions.filter((p: FlowPredictionItem) => 
        categoryFilter === 'all' || p.predicted_category === categoryFilter
      )
    : [];

  const totalPages = Math.ceil(filteredPredictions.length / itemsPerPage);
  const paginatedPredictions = filteredPredictions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Methodological Disclaimer */}
      <div style={{
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        borderLeft: '4px solid #f59e0b',
        borderRadius: '8px',
        padding: '14px 18px',
        fontSize: '0.86rem',
        color: '#fef3c7',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px'
      }}>
        <Info size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>RESEARCH & METHODOLOGY NOTICE:</strong> Graph edges represent <em>statistical behavioral similarity</em> between network flow feature vectors (via k-NN on 45 preprocessed features). They do not represent verified physical device endpoints, MAC/IP wiring, or actual packet routing.
        </div>
      </div>

      {/* CSV File Upload Section */}
      <div className="card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '6px', color: '#f8fafc' }}>
          Ingest & Analyze Network Flow CSV
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
          Upload a CICIoMT2024 standardized network flow dataset containing 45 numerical traffic features.
        </p>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: '2px dashed var(--border-accent)',
            borderRadius: '10px',
            backgroundColor: 'rgba(13, 21, 39, 0.5)',
            padding: '30px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'border-color 0.2s ease',
            marginBottom: '16px'
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv"
            style={{ display: 'none' }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <div style={{
              background: 'rgba(20, 184, 166, 0.1)',
              padding: '12px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Upload size={24} color="var(--accent-teal)" />
            </div>
            <div>
              <span style={{ fontWeight: 600, color: '#f8fafc' }}>
                {selectedFile ? selectedFile.name : 'Click to select CSV file or drag and drop here'}
              </span>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Supported schema: 45 numerical features (Duration, Rate, Packet Size, Flag Counts, etc.)
              </p>
            </div>
          </div>
        </div>

        {/* Selected File Details Banner */}
        {selectedFile && (
          <div style={{
            backgroundColor: 'rgba(15, 23, 42, 0.7)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '10px',
            fontSize: '0.82rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <FileText size={18} color="var(--accent-teal)" />
              <div>
                <strong style={{ color: '#f8fafc' }}>{selectedFile.name}</strong>
                <span style={{ color: 'var(--text-muted)', marginLeft: '10px' }}>
                  ({(selectedFile.size / 1024).toFixed(1)} KB)
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="badge" style={{ backgroundColor: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
                Ready to Process
              </span>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setSelectedFile(null); setResults(null); }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '0.78rem',
                  cursor: 'pointer',
                  textDecoration: 'underline'
                }}
              >
                Clear
              </button>
            </div>
          </div>
        )}

        {/* Action Buttons & Quick Sample Loader */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <button
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Building Flow Graph & Running GAT Inference...</span>
                </>
              ) : (
                <>
                  <Activity size={16} />
                  <span>Analyse Traffic</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={async () => {
                try {
                  const res = await fetch('/sample_input.csv');
                  const blob = await res.blob();
                  const file = new File([blob], 'sample_input.csv', { type: 'text/csv' });
                  setSelectedFile(file);
                  setError(null);
                } catch (e: any) {
                  setError('Failed to load sample_input.csv: ' + e.message);
                }
              }}
              style={{
                background: 'rgba(30, 41, 59, 0.8)',
                color: '#cbd5e1',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '10px 16px',
                fontSize: '0.85rem',
                fontWeight: 500,
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <FileText size={15} color="var(--accent-teal)" />
              <span>Load sample_input.csv (50 flows)</span>
            </button>
          </div>

          <a
            href="/sample_input.csv"
            download="sample_input.csv"
            style={{
              color: 'var(--accent-teal)',
              fontSize: '0.82rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none'
            }}
          >
            <Download size={14} />
            <span>Download template sample_input.csv</span>
          </a>
        </div>

        {/* Error Alert */}
        {error && (
          <div style={{
            marginTop: '16px',
            backgroundColor: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            padding: '12px 16px',
            color: '#fca5a5',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '10px'
          }}>
            <AlertCircle size={18} color="#ef4444" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>{error}</div>
          </div>
        )}
      </div>

      {/* Results Section */}
      {results ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Top Summary Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '14px'
          }}>
            <div className="card" style={{ padding: '16px' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>TOTAL FLOWS</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f8fafc', margin: '4px 0' }}>
                {results.total_records.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--accent-teal)' }}>Analyzed Flows</div>
            </div>

            <div className="card" style={{ padding: '16px' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>GRAPH TOPOLOGY</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--accent-cyan)', margin: '4px 0' }}>
                {results.graph_summary.num_edges.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {results.graph_summary.num_nodes} Nodes (k={results.graph_summary.k_neighbors})
              </div>
            </div>

            {Object.entries(results.category_counts).map(([cat, count]) => (
              <div key={cat} className="card" style={{ padding: '16px' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
                  {cat}
                </div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f8fafc', margin: '4px 0' }}>
                  {count.toLocaleString()}
                </div>
                <span className={`badge ${CATEGORY_BADGES[cat] || 'badge-benign'}`}>
                  {((count / results.total_records) * 100).toFixed(1)}% of total
                </span>
              </div>
            ))}
          </div>

          {/* Detailed Predictions Table */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#f8fafc' }}>
                  Individual Flow Classification Results
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  GAT predictions with confidence scores and class posterior distributions.
                </p>
              </div>

              {/* Filter and Download */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Filter size={15} color="var(--text-muted)" />
                  <select
                    value={categoryFilter}
                    onChange={(e) => { setCategoryFilter(e.target.value); setCurrentPage(1); }}
                    style={{
                      backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '6px',
                      padding: '6px 12px',
                      color: '#f8fafc',
                      fontSize: '0.82rem',
                      outline: 'none'
                    }}
                  >
                    <option value="all">All Categories ({results.total_records})</option>
                    {Object.entries(results.category_counts).map(([cat, count]) => (
                      <option key={cat} value={cat}>{cat} ({count})</option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={handleDownloadPredictionsCSV}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    backgroundColor: 'rgba(20, 184, 166, 0.15)',
                    border: '1px solid var(--border-accent)',
                    borderRadius: '6px',
                    padding: '6px 14px',
                    color: 'var(--accent-teal)',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  <Download size={14} />
                  <span>Download Predictions (CSV)</span>
                </button>
              </div>
            </div>

            {/* Table */}
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Flow Index</th>
                    <th>Predicted Category</th>
                    <th>Confidence</th>
                    <th>Posterior Distribution</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedPredictions.map((pred) => {
                    const badgeClass = CATEGORY_BADGES[pred.predicted_category] || 'badge-benign';
                    const confPercent = (pred.confidence * 100).toFixed(1);

                    return (
                      <tr key={pred.flow_index}>
                        <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.84rem' }}>
                          #{pred.flow_index}
                        </td>
                        <td>
                          <span className={`badge ${badgeClass}`}>
                            {pred.predicted_category}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <strong style={{ fontSize: '0.85rem' }}>{confPercent}%</strong>
                            <div style={{ width: '60px', height: '6px', backgroundColor: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{
                                width: `${confPercent}%`,
                                height: '100%',
                                backgroundColor: pred.predicted_category === 'Benign' ? 'var(--accent-emerald)' : 'var(--accent-red)'
                              }} />
                            </div>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', fontSize: '0.72rem' }}>
                            {Object.entries(pred.probabilities).map(([cat, prob]) => (
                              <span key={cat} style={{
                                padding: '2px 6px',
                                borderRadius: '4px',
                                backgroundColor: 'rgba(15, 23, 42, 0.6)',
                                border: '1px solid var(--border-subtle)',
                                color: cat === pred.predicted_category ? 'var(--accent-teal)' : 'var(--text-muted)'
                              }}>
                                {cat.split(' ')[0]}: {(prob * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '16px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <div>
                  Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredPredictions.length)} of {filteredPredictions.length} records
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    style={{
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '6px',
                      padding: '6px 10px',
                      color: currentPage === 1 ? 'var(--text-muted)' : '#f8fafc',
                      cursor: currentPage === 1 ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <span>Page {currentPage} of {totalPages}</span>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    style={{
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '6px',
                      padding: '6px 10px',
                      color: currentPage === totalPages ? 'var(--text-muted)' : '#f8fafc',
                      cursor: currentPage === totalPages ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Neutral Placeholder before analysis */
        <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
          <Network size={40} color="var(--text-muted)" style={{ margin: '0 auto 14px auto', opacity: 0.6 }} />
          <h4 style={{ fontSize: '1.05rem', color: '#f8fafc', fontWeight: 600 }}>
            Awaiting Traffic Analysis
          </h4>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '440px', margin: '6px auto 0 auto' }}>
            Select a compatible CICIoMT2024 network flow CSV file and click <strong>Analyse Traffic</strong> to construct the similarity graph and compute intrusion predictions.
          </p>
        </div>
      )}
    </div>
  );
};
