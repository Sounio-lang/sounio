/**
 * Sounio AI Workbench - Epistemic Octonion Transformer (EOT) Interface
 * 
 * Interactive research sandbox for the EOT PoC, featuring:
 * - Prompt input with generation controls
 * - Response display with Truth-Score visualization
 * - Attention weight visualization
 * - Provenance trace explorer
 * - Benchmark comparison dashboard
 */

import { useState, useCallback, type FC } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Slider } from '@/components/ui/Slider';
import { Badge } from '@/components/ui/Badge';

// Types matching the Sounio backend
interface TruthScoreToken {
  token_id: number;
  token_text: string;
  probability: number;
  truth_score: number;
  uncertainty: number;
  is_low_confidence: boolean;
}

interface GenerationResult {
  tokens: TruthScoreToken[];
  mean_truth_score: number;
  min_truth_score: number;
  overall_confidence: number;
  halted_reason: string;
}

interface AttentionWeight {
  layer: number;
  head: number;
  weights: number[];
}

interface BenchmarkResult {
  task_name: string;
  eot_accuracy: number;
  baseline_accuracy: number;
  hallucination_detection_rate: number;
  silent_hallucination_rate: number;
  ece: number;
}

// Truth Score Color Mapping
const getTruthScoreColor = (score: number): string => {
  if (score >= 0.9) return '#22c55e'; // Green - high confidence
  if (score >= 0.7) return '#eab308'; // Yellow - moderate
  return '#ef4444'; // Red - low confidence
};

// Token Display Component
const TokenDisplay: FC<{ token: TruthScoreToken }> = ({ token }) => {
  const color = getTruthScoreColor(token.truth_score);
  const isLow = token.is_low_confidence;
  
  return (
    <span 
      className={`inline-token ${isLow ? 'low-confidence' : ''}`}
      style={{ 
        backgroundColor: `${color}20`,
        borderBottom: `2px solid ${color}`,
        padding: '0 2px',
        margin: '0 1px',
        borderRadius: '2px',
        cursor: 'pointer'
      }}
      title={`Score: ${(token.truth_score * 100).toFixed(1)}% | Uncertainty: ${(token.uncertainty * 100).toFixed(1)}%`}
    >
      {token.token_text}
    </span>
  );
};

// Truth Score Legend Component
const TruthScoreLegend: FC = () => (
  <div className="flex items-center gap-4 text-sm text-gray-600">
    <div className="flex items-center gap-2">
      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#22c55e' }}></span>
      <span>High (≥90%)</span>
    </div>
    <div className="flex items-center gap-2">
      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#eab308' }}></span>
      <span>Moderate (70-89%)</span>
    </div>
    <div className="flex items-center gap-2">
      <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }}></span>
      <span>Low (&lt;70%)</span>
    </div>
  </div>
);

// Attention Heatmap Component
const AttentionHeatmap: FC<{ weights: AttentionWeight }> = ({ weights }) => {
  const { layer, head, weights: w } = weights;
  const size = Math.sqrt(w.length);
  
  return (
    <div className="attention-heatmap">
      <div className="text-sm font-medium mb-2">
        Layer {layer + 1}, Head {head + 1}
      </div>
      <div 
        className="grid gap-px"
        style={{ 
          gridTemplateColumns: `repeat(${size}, 1fr)`,
          width: '200px',
          height: '200px'
        }}
      >
        {w.map((weight, idx) => (
          <div
            key={idx}
            style={{
              backgroundColor: `rgba(59, 130, 246, ${weight})`,
              minWidth: '4px',
              minHeight: '4px'
            }}
            title={`${weight.toFixed(3)}`}
          />
        ))}
      </div>
    </div>
  );
};

// Confidence Bar Component
const ConfidenceBar: FC<{ score: number; label: string }> = ({ score, label }) => {
  const color = getTruthScoreColor(score);
  
  return (
    <div className="confidence-bar">
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span style={{ color }}>{(score * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-300"
          style={{ 
            width: `${score * 100}%`,
            backgroundColor: color
          }}
        />
      </div>
    </div>
  );
};

// Main Workbench Component
export const EOTWorkbench: FC = () => {
  // State
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  
  // Controls
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [strictMode, setStrictMode] = useState(false);
  const [modelSize, setModelSize] = useState<'tiny' | 'small' | 'medium'>('small');
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  
  // Generate handler
  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    
    // Simulate generation (would call Sounio backend in production)
    setTimeout(() => {
      const mockTokens: TruthScoreToken[] = prompt.split(' ').map((word, idx) => ({
        token_id: idx,
        token_text: word,
        probability: 0.8 + Math.random() * 0.2,
        truth_score: 0.7 + Math.random() * 0.3,
        uncertainty: 0.1 + Math.random() * 0.2,
        is_low_confidence: Math.random() < 0.1
      }));
      
      const mockResult: GenerationResult = {
        tokens: mockTokens,
        mean_truth_score: 0.85,
        min_truth_score: 0.65,
        overall_confidence: 0.82,
        halted_reason: 'MAX_LENGTH'
      };
      
      setResult(mockResult);
      setIsGenerating(false);
    }, 1500);
  }, [prompt]);
  
  // Run benchmark handler
  const handleRunBenchmark = useCallback(async () => {
    setIsGenerating(true);
    
    // Simulate benchmark (would call Sounio backend in production)
    setTimeout(() => {
      const mockResults: BenchmarkResult[] = [
        {
          task_name: 'Fact Recall',
          eot_accuracy: 0.94,
          baseline_accuracy: 0.92,
          hallucination_detection_rate: 0.85,
          silent_hallucination_rate: 0.03,
          ece: 0.08
        },
        {
          task_name: 'Parameter Efficiency',
          eot_accuracy: 0.95,
          baseline_accuracy: 0.95,
          hallucination_detection_rate: 0.0,
          silent_hallucination_rate: 0.0,
          ece: 0.05
        }
      ];
      
      setBenchmarkResults(mockResults);
      setIsGenerating(false);
    }, 2000);
  }, []);
  
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Sounio AI Workbench</h1>
          <p className="text-gray-600 mt-2">
            Epistemic Octonion Transformer (EOT) Research Sandbox
          </p>
        </div>
        
        <div className="grid grid-cols-12 gap-6">
          {/* Left Panel - Controls */}
          <div className="col-span-3 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Confidence Threshold */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Confidence Threshold: {confidenceThreshold.toFixed(1)}
                  </label>
                  <Slider
                    value={confidenceThreshold}
                    onChange={setConfidenceThreshold}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </div>
                
                {/* Strict Mode */}
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="strictMode"
                    checked={strictMode}
                    onChange={(e) => setStrictMode(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="strictMode" className="text-sm">
                    Strict Mode (pause on low confidence)
                  </label>
                </div>
                
                {/* Model Size */}
                <div>
                  <label className="block text-sm font-medium mb-2">Model Size</label>
                  <div className="space-y-2">
                    {(['tiny', 'small', 'medium'] as const).map((size) => (
                      <div key={size} className="flex items-center gap-2">
                        <input
                          type="radio"
                          id={`size-${size}`}
                          name="modelSize"
                          checked={modelSize === size}
                          onChange={() => setModelSize(size)}
                          className="rounded"
                        />
                        <label htmlFor={`size-${size}`} className="text-sm capitalize">
                          {size} ({size === 'tiny' ? '2L' : size === 'small' ? '8L' : '16L'})
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Actions */}
                <Button 
                  onClick={handleGenerate}
                  disabled={isGenerating || !prompt.trim()}
                  className="w-full"
                >
                  {isGenerating ? 'Generating...' : 'Generate'}
                </Button>
                
                <Button 
                  onClick={handleRunBenchmark}
                  disabled={isGenerating}
                  variant="secondary"
                  className="w-full"
                >
                  Run Benchmark
                </Button>
              </CardContent>
            </Card>
            
            {/* Legend */}
            <Card>
              <CardHeader>
                <CardTitle>Truth-Score Legend</CardTitle>
              </CardHeader>
              <CardContent>
                <TruthScoreLegend />
              </CardContent>
            </Card>
          </div>
          
          {/* Center Panel - Response */}
          <div className="col-span-6 space-y-4">
            {/* Prompt Input */}
            <Card>
              <CardHeader>
                <CardTitle>Prompt Input</CardTitle>
              </CardHeader>
              <CardContent>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here..."
                  className="w-full h-32 p-3 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500"
                />
              </CardContent>
            </Card>
            
            {/* Response Display */}
            {result && (
              <Card>
                <CardHeader>
                  <CardTitle>Response</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-lg leading-relaxed mb-4">
                    {result.tokens.map((token, idx) => (
                      <TokenDisplay key={idx} token={token} />
                    ))}
                  </div>
                  
                  <div className="border-t pt-4">
                    <ConfidenceBar 
                      score={result.mean_truth_score} 
                      label="Mean Confidence" 
                    />
                    <div className="mt-2">
                      <ConfidenceBar 
                        score={result.min_truth_score} 
                        label="Min Confidence" 
                      />
                    </div>
                  </div>
                  
                  {result.min_truth_score < confidenceThreshold && (
                    <Badge variant="warning" className="mt-4">
                      Low confidence tokens detected
                    </Badge>
                  )}
                </CardContent>
              </Card>
            )}
            
            {/* Benchmark Results */}
            {benchmarkResults.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Benchmark Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2">Task</th>
                        <th className="text-right py-2">EOT Acc.</th>
                        <th className="text-right py-2">Baseline</th>
                        <th className="text-right py-2">Halluc. Detect</th>
                        <th className="text-right py-2">ECE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkResults.map((r, idx) => (
                        <tr key={idx} className="border-b">
                          <td className="py-2">{r.task_name}</td>
                          <td className="text-right">{(r.eot_accuracy * 100).toFixed(1)}%</td>
                          <td className="text-right">{(r.baseline_accuracy * 100).toFixed(1)}%</td>
                          <td className="text-right">{(r.hallucination_detection_rate * 100).toFixed(1)}%</td>
                          <td className="text-right">{r.ece.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            )}
          </div>
          
          {/* Right Panel - Visualization */}
          <div className="col-span-3 space-y-4">
            {/* Attention Visualizer */}
            <Card>
              <CardHeader>
                <CardTitle>Attention Visualizer</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <select 
                    value={selectedLayer}
                    onChange={(e) => setSelectedLayer(Number(e.target.value))}
                    className="flex-1 p-2 border rounded"
                  >
                    {[0, 1, 2, 3, 4, 5, 6, 7].map((l) => (
                      <option key={l} value={l}>Layer {l + 1}</option>
                    ))}
                  </select>
                  <select 
                    value={selectedHead}
                    onChange={(e) => setSelectedHead(Number(e.target.value))}
                    className="flex-1 p-2 border rounded"
                  >
                    {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
                      <option key={h} value={h}>Head {h + 1}</option>
                    ))}
                  </select>
                </div>
                
                <AttentionHeatmap 
                  weights={{
                    layer: selectedLayer,
                    head: selectedHead,
                    weights: Array(64).fill(0).map(() => Math.random())
                  }}
                />
              </CardContent>
            </Card>
            
            {/* Provenance Trace */}
            {result && (
              <Card>
                <CardHeader>
                  <CardTitle>Provenance Trace</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-2">
                    Click any token to see its provenance
                  </p>
                  <div className="text-xs space-y-1">
                    <p className="font-medium">Sample: {result.tokens[0]?.token_text || 'N/A'}</p>
                    <p>Confidence: {((result.tokens[0]?.truth_score || 0) * 100).toFixed(1)}%</p>
                    <p>Sources: 3 attention heads</p>
                    <p>Weight confidence: 0.98</p>
                  </div>
                </CardContent>
              </Card>
            )}
            
            {/* Export */}
            <Card>
              <CardHeader>
                <CardTitle>Export</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full">
                  Export Session (JSON)
                </Button>
                <Button variant="outline" className="w-full">
                  Export for Paper (ZIP)
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EOTWorkbench;
