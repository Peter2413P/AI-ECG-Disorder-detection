import React, { useState, useRef } from 'react';
import axios from 'axios';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend 
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { UploadCloud, Activity, BrainCircuit, HeartPulse, AlertCircle, CheckCircle, Download } from 'lucide-react';
import { motion } from 'framer-motion';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import ECGViewer from './components/ECGViewer';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const TARGET_CONDITIONS = [
  "Normal_Sinus_Rhythm", "Sinus_Tachycardia", "Sinus_Arrhythmia", "PAC", 
  "RBBB", "LBBB", "IVCD", "Delta_Wave", "Persistent_ST_Elevation", 
  "Left_Atrial_Enlargement", "Ventricular_Fibrillation_Flutter", "Pacemaker_Rhythm"
];

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const reportRef = useRef<HTMLDivElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(Array.from(e.dataTransfer.files));
    }
  };

  const processECG = async () => {
    if (files.length === 0) return;
    setError(null);
    setIsUploading(true);
    
    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file);
      }
      
      const uploadRes = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setIsUploading(false);
      setIsPredicting(true);
      
      const predictRes = await axios.post('/predict', { filepath: uploadRes.data.path });
      setResults(predictRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "An error occurred during processing.");
    } finally {
      setIsUploading(false);
      setIsPredicting(false);
    }
  };

  const [showSettings, setShowSettings] = useState(false);

  const generatePDF = () => {
    // html2canvas often fails on complex Plotly SVGs/WebGL. 
    // Using the native browser print API guarantees high quality, selectable text, and proper chart rendering.
    window.print();
  };

  const renderProbBars = () => {
    if (!results) return null;
    return TARGET_CONDITIONS.map(cond => {
      const prob = results.predictions[cond] || 0;
      const thresh = results.thresholds?.[cond] || 0.5;
      const isPositive = prob >= thresh;
      
      return (
        <motion.div 
          key={cond} 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className={`flex justify-between items-center p-3 rounded-lg border-l-4 ${isPositive ? 'border-red-500 bg-red-500/10' : 'border-green-400 bg-white/5'}`}
        >
          <div className="w-full">
            <div className="flex justify-between text-sm mb-1 font-medium">
              <span>{cond.replace(/_/g, ' ')}</span>
              <span>{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${isPositive ? 'bg-red-500' : 'bg-green-400'}`}
                style={{ width: `${Math.min(prob * 100, 100)}%` }} 
              />
            </div>
          </div>
        </motion.div>
      );
    });
  };

  const renderShapChart = () => {
    if (!results || !results.shap_importance) return null;
    const detectedConditions = Object.keys(results.shap_importance);
    if (detectedConditions.length === 0) return <p className="text-slate-400">No abnormalities detected.</p>;
    
    const primaryCond = detectedConditions[0];
    const shapData = results.shap_importance[primaryCond];
    
    const data = {
      labels: Object.keys(shapData).map(k => k.replace(/_/g, ' ')),
      datasets: [{
        label: `Impact on ${primaryCond.replace(/_/g, ' ')}`,
        data: Object.values(shapData),
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderRadius: 4,
      }],
    };

    return (
      <div className="h-64 mt-4">
        <Bar data={data} options={{
          responsive: true, maintainAspectRatio: false,
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#f8fafc' } } }
        }} />
      </div>
    );
  };
  
  const renderLeadHeatmap = () => {
    if (!results || !results.lead_importance) return null;
    const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
    
    return (
      <div className="grid grid-cols-4 gap-2 mt-4">
        {leads.map(lead => {
          const imp = results.lead_importance[lead] || 0;
          // Calculate opacity based on importance (max is likely around 30-50%)
          const opacity = Math.min(imp / 30, 1) * 0.8 + 0.1;
          return (
            <div key={lead} className="p-2 rounded flex flex-col items-center justify-center border border-white/5" style={{ backgroundColor: `rgba(239, 68, 68, ${opacity})` }}>
              <span className="font-bold">{lead}</span>
              <span className="text-xs">{imp}%</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="max-w-[1600px] mx-auto p-4 md:p-8 flex flex-col gap-8">
      <header className="flex justify-between items-center pb-6 border-b border-white/10">
        <div className="text-3xl font-display font-semibold bg-gradient-to-br from-white to-indigo-300 bg-clip-text text-transparent flex items-center gap-3">
          <img src="/logo.png" alt="KORAK Logo" className="h-10" />
          CardioVision <span className="font-light text-slate-400">Clinical AI</span>
        </div>
        <div className="flex gap-4">
          {results && (
            <button onClick={generatePDF} className="glass-button primary !bg-indigo-500 hover:!bg-indigo-600 !shadow-[0_0_15px_rgba(99,102,241,0.4)]">
              <Download size={18} /> Export PDF
            </button>
          )}
          <button className="glass-button" onClick={() => setShowSettings(true)}>Settings</button>
        </div>
      </header>

      {!results && (
        <motion.div className="glass-panel p-8 max-w-3xl mx-auto w-full" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h2 className="text-2xl mb-2">Upload ECG Record</h2>
          <p className="text-slate-400 mb-8">Upload WFDB (.mat, .dat) or extracted CSV features for clinical AI analysis.</p>
          
          <div 
            className={`border-2 border-dashed rounded-3xl p-12 flex flex-col items-center justify-center cursor-pointer transition-all ${files.length > 0 ? 'border-green-400 bg-green-400/5' : 'border-white/20 bg-black/20 hover:border-green-400/50 hover:bg-white/5'}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileUpload')?.click()}
          >
            <input id="fileUpload" type="file" multiple className="hidden" onChange={(e) => setFiles(e.target.files ? Array.from(e.target.files) : [])} accept=".mat,.dat,.hea,.csv" />
            <UploadCloud size={64} className={`mb-4 ${files.length > 0 ? "text-green-400" : "text-slate-500"}`} />
            {files.length > 0 ? (
              <h3 className="text-xl text-green-400 font-medium">{files.length} file(s) selected</h3>
            ) : (
              <>
                <h3 className="text-xl font-medium">Drag & Drop or Click to Browse</h3>
                <p className="text-slate-500 mt-2">Supports PhysioNet WFDB format (.mat + .hea)</p>
              </>
            )}
          </div>

          <div className="mt-8 flex justify-center">
            <button 
              className="glass-button primary px-12 py-4 text-lg"
              onClick={processECG}
              disabled={files.length === 0 || isUploading || isPredicting}
            >
              {isUploading ? "Uploading..." : isPredicting ? "Running Neural Engines..." : "Analyze ECG"}
              {(isUploading || isPredicting) && <div className="w-5 h-5 ml-3 border-2 border-white/20 border-t-black rounded-full animate-spin" />}
            </button>
          </div>
          
          {error && (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 flex items-center gap-2">
              <AlertCircle size={20} /> {error}
            </div>
          )}
        </motion.div>
      )}

      {results && (
        <div className="flex flex-col gap-8 animate-fade-in" ref={reportRef}>
          {/* Top: Plotly Viewer */}
          <div className="w-full">
            <ECGViewer waveforms={results.waveforms} delineation={results.delineation} fs={results.fs} />
          </div>
          
          {/* FINAL PREDICTION BANNER */}
          {(() => {
            const positiveConditions: {name: string, prob: number}[] = [];
            
            if (results.predictions) {
              for (const cond of Object.keys(results.predictions)) {
                const prob = results.predictions[cond];
                const thresh = results.thresholds?.[cond] || 0.5;
                if (prob >= thresh) {
                  positiveConditions.push({ name: cond, prob });
                }
              }
            }
            
            positiveConditions.sort((a, b) => b.prob - a.prob);
            const anomalies = positiveConditions.filter(c => c.name !== "Normal_Sinus_Rhythm");
            
            const isAnomaly = anomalies.length > 0;
            const primary = isAnomaly ? anomalies[0] : positiveConditions.find(c => c.name === "Normal_Sinus_Rhythm");
            
            const diagnosisName = primary ? primary.name.replace(/_/g, ' ') : "Normal Sinus Rhythm";
            const primaryProb = primary ? primary.prob : 0;
            const secondaryFindings = isAnomaly ? anomalies.slice(1) : [];
            
            return (
              <div className={`w-full p-6 rounded-2xl border-2 flex flex-col md:flex-row items-center justify-between shadow-lg backdrop-blur-sm transition-all gap-6 ${isAnomaly ? 'bg-red-500/10 border-red-500/50 shadow-red-500/10' : 'bg-green-500/10 border-green-500/50 shadow-green-500/10'}`}>
                <div className="flex items-center gap-4 w-full md:w-auto">
                  {isAnomaly ? <AlertCircle size={48} className="text-red-500 shrink-0" /> : <CheckCircle size={48} className="text-green-500 shrink-0" />}
                  <div>
                    <h2 className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">Primary AI Diagnosis</h2>
                    <div className={`text-3xl font-display font-bold ${isAnomaly ? 'text-red-400' : 'text-green-400'}`}>
                      {diagnosisName}
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-row items-center gap-8 w-full md:w-auto justify-between md:justify-end">
                  {secondaryFindings.length > 0 && (
                    <div className="text-left md:text-right border-l-2 border-white/10 pl-6 hidden md:block">
                      <div className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">Secondary Findings</div>
                      <div className="text-md font-display font-medium text-red-300">
                        {secondaryFindings.map(f => `${f.name.replace(/_/g, ' ')} (${(f.prob * 100).toFixed(1)}%)`).join(', ')}
                      </div>
                    </div>
                  )}
                  
                  {(isAnomaly || primaryProb > 0) && (
                    <div className="text-right shrink-0">
                      <div className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">Confidence</div>
                      <div className={`text-3xl font-display font-bold ${isAnomaly ? 'text-red-400' : 'text-green-400'}`}>
                        {(primaryProb * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Mobile Secondary Findings */}
                {secondaryFindings.length > 0 && (
                  <div className="w-full text-left border-t border-white/10 pt-4 block md:hidden">
                    <div className="text-sm text-slate-400 uppercase tracking-wider font-semibold mb-1">Secondary Findings</div>
                    <div className="text-md font-display font-medium text-red-300">
                      {secondaryFindings.map(f => `${f.name.replace(/_/g, ' ')} (${(f.prob * 100).toFixed(1)}%)`).join(', ')}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column */}
            <div className="flex flex-col gap-8">
              <div className="glass-panel p-6">
                <h3 className="flex items-center gap-2 text-xl mb-6 font-semibold"><HeartPulse className="text-red-500" /> Detected Anomalies</h3>
                {Object.keys(results.predictions).length === 0 ? (
                  <div className="text-center py-8 text-green-400">
                    <CheckCircle size={48} className="mx-auto mb-4" />
                    <h4 className="text-xl">Normal Sinus Rhythm</h4>
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">{renderProbBars()}</div>
                )}
              </div>

              <div className="glass-panel p-6">
                <h3 className="text-xl mb-4 font-semibold">Lead Importance</h3>
                <p className="text-sm text-slate-400">Heatmap of localized pathology.</p>
                {renderLeadHeatmap()}
              </div>
              
              <div className="glass-panel p-6">
                <h3 className="text-xl mb-4 font-semibold">Vital Metrics</h3>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(results.key_features).map(([k, v]: [string, any]) => {
                    // Simple color coding logic for demo
                    let color = 'text-white';
                    if (k === 'Heart_Rate') color = (v < 60 || v > 100) ? 'text-red-400' : 'text-green-400';
                    if (k === 'PR_Interval') color = (v > 200) ? 'text-red-400' : 'text-white';
                    
                    return (
                      <div key={k} className="bg-white/5 p-3 rounded-lg border border-white/5">
                        <div className="text-xs text-slate-400 mb-1">{k.replace(/_/g, ' ')}</div>
                        <div className={`text-xl font-semibold ${color}`}>{typeof v === 'number' ? v.toFixed(1) : v}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Right Column (Span 2) */}
            <div className="lg:col-span-2 flex flex-col gap-8">
              <div className="glass-panel p-8">
                <h3 className="flex items-center gap-2 text-2xl mb-6 font-semibold"><BrainCircuit className="text-blue-500" /> AI Clinical Report</h3>
                <div className="prose prose-invert max-w-none">
                  {/* We render markdown via simple replacement for now or just pre-wrap */}
                  <div className="whitespace-pre-wrap font-sans leading-relaxed text-slate-300" dangerouslySetInnerHTML={{
                    __html: results.explanation
                      .replace(/## (.*)/g, '<h2 class="text-xl font-bold text-white mt-6 mb-3 border-b border-white/10 pb-2">$1</h2>')
                      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
                      .replace(/- (.*)/g, '<li class="ml-4 mb-1">$1</li>')
                  }} />
                </div>
              </div>

              <div className="glass-panel p-8">
                <h3 className="text-xl mb-2 font-semibold">Global Feature Importance (SHAP)</h3>
                <p className="text-sm text-slate-400">Quantitative explanation of ML decision drivers.</p>
                {renderShapChart()}
              </div>
            </div>
          </div>
          
          <div className="flex justify-end pb-12">
            <button className="glass-button !px-8" onClick={() => { setResults(null); setFiles([]); }}>
              Analyze New Patient
            </button>
          </div>
        </div>
      )}
      {showSettings && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in">
          <div className="glass-panel p-8 w-full max-w-md relative">
            <h2 className="text-2xl font-display font-semibold mb-6">System Settings</h2>
            
            <div className="flex flex-col gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-1 block">RAG Engine Model</label>
                <select className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-white outline-none focus:border-green-400/50">
                  <option value="phi3">Phi-3 Mini (Local)</option>
                  <option value="llama3">Llama 3 8B (Local)</option>
                </select>
              </div>
              
              <div>
                <label className="text-sm text-slate-400 mb-1 block">UI Theme</label>
                <select className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-white outline-none focus:border-green-400/50">
                  <option value="dark">Dark Mode (Clinical)</option>
                  <option value="light" disabled>Light Mode (Coming Soon)</option>
                </select>
              </div>
            </div>

            <div className="mt-8 flex justify-end gap-3">
              <button className="glass-button !px-6" onClick={() => setShowSettings(false)}>Close</button>
              <button className="glass-button primary !px-6" onClick={() => setShowSettings(false)}>Save Changes</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
