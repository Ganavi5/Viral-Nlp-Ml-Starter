// src/pages/Results.js - Viral Genome Prediction Results
import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Download, 
  Eye, 
  BarChart3,
  Dna,
  Target,
  TrendingUp,
  FileText,
  Copy,
  RefreshCw
} from 'lucide-react';

const ResultsPage = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [summary, setSummary] = useState(null);
  const [notification, setNotification] = useState(null);

  // Show notification
  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  // Mock results data (replace with actual API call)
  useEffect(() => {
    // Simulate API call to get results
    setTimeout(() => {
      const mockResults = [
        {
          id: 1,
          sequenceId: "seq1806848_2014_P_Lagheden_Matti-HPV-vacc-C1N31",
          sequence: "TCTGGTAATTGACAAAATGTTAATCAGAAATACTAAATATTTGGCCAAATTTACCTTAATACCCGCTTAGCCTTGCCAACGGCTTTTAAATAGGTAAAATATCACTAGCCTAGACGAAGGTTTGGTATGTGAAAGCTGTACT",
          prediction: "Non-Viral",
          confidence: 98.0,
          viralProbability: 2.0,
          modelUsed: "Random Forest",
          timestamp: "2024-01-15 14:30:22",
          features: {
            length: 150,
            gcContent: 42.7,
            atContent: 57.3,
            kmerComplexity: 0.85
          }
        },
        {
          id: 2,
          sequenceId: "seq368982_2014_F1_PARAFFIN8LANKBLOCKS",
          sequence: "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",
          prediction: "Non-Viral",
          confidence: 98.0,
          viralProbability: 2.0,
          modelUsed: "Random Forest",
          timestamp: "2024-01-15 14:30:22",
          features: {
            length: 128,
            gcContent: 50.0,
            atContent: 50.0,
            kmerComplexity: 0.12
          }
        },
        {
          id: 3,
          sequenceId: "seq1135576_2011_N19_VIRASKINPAPMISEQ",
          sequence: "TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCT",
          prediction: "Non-Viral",
          confidence: 96.0,
          viralProbability: 4.0,
          modelUsed: "Random Forest",
          timestamp: "2024-01-15 14:30:22",
          features: {
            length: 120,
            gcContent: 50.0,
            atContent: 50.0,
            kmerComplexity: 0.25
          }
        },
        {
          id: 4,
          sequenceId: "seq277_2014_G6_Multip_MSS",
          sequence: "CCGGGGGAAATTTTCCCGGGAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAA",
          prediction: "Non-Viral",
          confidence: 93.0,
          viralProbability: 7.0,
          modelUsed: "Random Forest",
          timestamp: "2024-01-15 14:30:22",
          features: {
            length: 116,
            gcContent: 58.6,
            atContent: 41.4,
            kmerComplexity: 0.35
          }
        }
      ];

      setResults(mockResults);
      
      // Calculate summary
      const viralCount = mockResults.filter(r => r.prediction === 'Viral').length;
      const nonViralCount = mockResults.filter(r => r.prediction === 'Non-Viral').length;
      const avgConfidence = mockResults.reduce((sum, r) => sum + r.confidence, 0) / mockResults.length;
      
      setSummary({
        total: mockResults.length,
        viral: viralCount,
        nonViral: nonViralCount,
        avgConfidence: avgConfidence.toFixed(1)
      });
      
      setLoading(false);
    }, 1500);
  }, []);

  // Copy sequence to clipboard
  const copySequence = (sequence) => {
    navigator.clipboard.writeText(sequence);
    showNotification('Sequence copied to clipboard!', 'success');
  };

  // Download results as CSV
  const downloadResults = () => {
    const csvContent = [
      ['Sequence ID', 'Prediction', 'Confidence (%)', 'Viral Probability (%)', 'Sequence Length', 'GC Content (%)', 'Model Used'],
      ...results.map(r => [
        r.sequenceId,
        r.prediction,
        r.confidence,
        r.viralProbability,
        r.features.length,
        r.features.gcContent,
        r.modelUsed
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'viral_genome_predictions.csv';
    a.click();
    window.URL.revokeObjectURL(url);
    
    showNotification('Results downloaded successfully!', 'success');
  };

  // Get prediction color
  const getPredictionColor = (prediction) => {
    return prediction === 'Viral' ? 'text-red-400' : 'text-green-400';
  };

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-green-400';
    if (confidence >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="animate-spin text-cyan-400 mx-auto mb-4" size={48} />
          <p className="text-white text-xl">Processing predictions...</p>
          <p className="text-gray-400">Analyzing genome sequences with Random Forest model</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-lg text-white transition-all duration-300 ${
          notification.type === 'success' ? 'bg-green-500' :
          notification.type === 'error' ? 'bg-red-500' : 'bg-blue-500'
        }`}>
          <div className="flex items-center space-x-2">
            {notification.type === 'success' && <CheckCircle size={20} />}
            {notification.type === 'error' && <AlertTriangle size={20} />}
            <span>{notification.message}</span>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-cyan-400 mb-2">Prediction Results</h1>
            <p className="text-gray-400">Random Forest model predictions for viral genome sequences</p>
          </div>
          <button
            onClick={downloadResults}
            className="bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors"
          >
            <Download size={20} />
            <span>Download Results</span>
          </button>
        </div>

        {/* Summary Cards */}
        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="bg-blue-500/20 p-3 rounded-lg">
                  <FileText className="text-blue-400" size={24} />
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Total Sequences</p>
                  <p className="text-white text-2xl font-bold">{summary.total}</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="bg-red-500/20 p-3 rounded-lg">
                  <AlertTriangle className="text-red-400" size={24} />
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Viral</p>
                  <p className="text-white text-2xl font-bold">{summary.viral}</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="bg-green-500/20 p-3 rounded-lg">
                  <CheckCircle className="text-green-400" size={24} />
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Non-Viral</p>
                  <p className="text-white text-2xl font-bold">{summary.nonViral}</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="bg-cyan-500/20 p-3 rounded-lg">
                  <Target className="text-cyan-400" size={24} />
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Avg Confidence</p>
                  <p className="text-white text-2xl font-bold">{summary.avgConfidence}%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Table */}
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="p-6 border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <BarChart3 className="text-cyan-400" size={24} />
              <h2 className="text-xl font-semibold text-white">Detailed Results</h2>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">Sequence ID</th>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">Prediction</th>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">Confidence</th>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">Length</th>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">GC Content</th>
                  <th className="px-6 py-4 text-left text-gray-300 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={result.id} className={index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'}>
                    <td className="px-6 py-4 text-white font-mono text-sm">
                      {result.sequenceId.length > 30 
                        ? `${result.sequenceId.substring(0, 30)}...` 
                        : result.sequenceId}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        {result.prediction === 'Viral' ? (
                          <AlertTriangle className="text-red-400" size={16} />
                        ) : (
                          <CheckCircle className="text-green-400" size={16} />
                        )}
                        <span className={`font-medium ${getPredictionColor(result.prediction)}`}>
                          {result.prediction}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <span className={`font-medium ${getConfidenceColor(result.confidence)}`}>
                          {result.confidence}%
                        </span>
                        <div className="w-16 bg-gray-600 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              result.confidence >= 90 ? 'bg-green-400' :
                              result.confidence >= 70 ? 'bg-yellow-400' : 'bg-red-400'
                            }`}
                            style={{ width: `${result.confidence}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-300">{result.features.length} bp</td>
                    <td className="px-6 py-4 text-gray-300">{result.features.gcContent}%</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => {
                            setSelectedResult(result);
                            setShowDetails(true);
                          }}
                          className="text-cyan-400 hover:text-cyan-300 p-2 rounded-lg hover:bg-gray-700 transition-colors"
                          title="View Details"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          onClick={() => copySequence(result.sequence)}
                          className="text-gray-400 hover:text-gray-300 p-2 rounded-lg hover:bg-gray-700 transition-colors"
                          title="Copy Sequence"
                        >
                          <Copy size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Model Info */}
        <div className="mt-8 bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Activity className="text-cyan-400" size={24} />
            <h3 className="text-lg font-semibold text-white">Model Information</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-gray-300">
            <div>
              <h4 className="font-medium text-cyan-400 mb-2">Model Details</h4>
              <ul className="space-y-1 text-sm">
                <li>• Algorithm: Random Forest</li>
                <li>• Training Data: 200,000+ sequences</li>
                <li>• Features: K-mer analysis (1800+ features)</li>
                <li>• Cross-validation: 5-fold</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-cyan-400 mb-2">Performance Metrics</h4>
              <ul className="space-y-1 text-sm">
                <li>• Training Accuracy: 100%</li>
                <li>• Test Accuracy: 97.5%</li>
                <li>• Precision: 97.8%</li>
                <li>• Recall: 97.2%</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-cyan-400 mb-2">Processing Info</h4>
              <ul className="space-y-1 text-sm">
                <li>• Feature Extraction: K-mer tokenization</li>
                <li>• Sequence Length: Variable (50-5000 bp)</li>
                <li>• Processing Time: ~0.1s per sequence</li>
                <li>• Batch Processing: Supported</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Detail Modal */}
      {showDetails && selectedResult && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto border border-gray-700">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Sequence Details</h3>
              <button
                onClick={() => setShowDetails(false)}
                className="text-gray-400 hover:text-white p-2 rounded-lg hover:bg-gray-700 transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-medium text-cyan-400 mb-3">Prediction Results</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Prediction:</span>
                      <span className={`font-medium ${getPredictionColor(selectedResult.prediction)}`}>
                        {selectedResult.prediction}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Confidence:</span>
                      <span className={getConfidenceColor(selectedResult.confidence)}>
                        {selectedResult.confidence}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Viral Probability:</span>
                      <span className="text-white">{selectedResult.viralProbability}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Model Used:</span>
                      <span className="text-white">{selectedResult.modelUsed}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-medium text-cyan-400 mb-3">Sequence Features</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Length:</span>
                      <span className="text-white">{selectedResult.features.length} bp</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">GC Content:</span>
                      <span className="text-white">{selectedResult.features.gcContent}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">AT Content:</span>
                      <span className="text-white">{selectedResult.features.atContent}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">K-mer Complexity:</span>
                      <span className="text-white">{selectedResult.features.kmerComplexity}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sequence Display */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-cyan-400">DNA Sequence</h4>
                  <button
                    onClick={() => copySequence(selectedResult.sequence)}
                    className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 text-sm"
                  >
                    <Copy size={14} />
                    <span>Copy</span>
                  </button>
                </div>
                <div className="bg-gray-800 rounded p-4 font-mono text-sm text-green-400 break-all">
                  {selectedResult.sequence}
                </div>
              </div>

              {/* Sequence ID */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-cyan-400 mb-2">Sequence Identifier</h4>
                <p className="text-white font-mono text-sm">{selectedResult.sequenceId}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsPage;
