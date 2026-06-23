import React, { useMemo } from 'react';
import ReactPlotly from 'react-plotly.js';

// Vite CommonJS default export fix
const Plot = (ReactPlotly as any).default || ReactPlotly;

interface ECGViewerProps {
  waveforms: Record<string, number[]>;
  delineation: {
    P_Waves: number[][];
    QRS_Complexes: number[][];
    T_Waves: number[][];
  };
  fs: number;
}

const ECGViewer: React.FC<ECGViewerProps> = ({ waveforms, delineation, fs }) => {
  // Define standard 12-lead order for Plotly subplots
  const leadOrder = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

  const plotData = useMemo(() => {
    const data: any[] = [];
    if (!waveforms || Object.keys(waveforms).length === 0) return data;

    // To prevent browser freezing, we decimate if the signal is > 5000 samples (10s @ 500Hz)
    // For 12 leads, 5000 is perfectly fine.
    
    leadOrder.forEach((lead, index) => {
      if (!waveforms[lead]) return;
      
      const sig = waveforms[lead];
      const time = sig.map((_, i) => i / fs);
      
      // Calculate subplot row (12 leads -> 12 rows or a 6x2 grid. Let's do 12 rows for standard continuous scrolling view)
      const yaxis = `y${index + 1 === 1 ? '' : index + 1}`;
      const xaxis = `x${index + 1 === 1 ? '' : index + 1}`;
      
      data.push({
        x: time,
        y: sig,
        type: 'scatter',
        mode: 'lines',
        name: `Lead ${lead}`,
        line: { color: '#4ade80', width: 1.5 },
        xaxis: xaxis,
        yaxis: yaxis,
        hoverinfo: 'none'
      });
    });
    
    return data;
  }, [waveforms, fs]);

  const layout = useMemo(() => {
    const baseLayout: any = {
      title: { text: '12-Lead ECG Waveform Viewer', font: { color: '#f8fafc', family: 'Outfit' } },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#94a3b8' },
      showlegend: false,
      height: 1200, // 100px per lead
      margin: { l: 50, r: 20, t: 60, b: 40 },
      grid: { rows: 12, columns: 1, pattern: 'independent' },
      hovermode: 'closest'
    };

    // Configure axes for 12 subplots
    leadOrder.forEach((lead, index) => {
      const yaxis = `yaxis${index + 1 === 1 ? '' : index + 1}`;
      const xaxis = `xaxis${index + 1 === 1 ? '' : index + 1}`;
      
      baseLayout[yaxis] = {
        title: lead,
        fixedrange: false,
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)'
      };
      
      baseLayout[xaxis] = {
        showticklabels: index === 11, // Only show x-axis time on the bottom plot
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)'
      };
    });

    // Add shapes for delineation highlighting (P, QRS, T)
    // We will overlay these on Lead II (which is index 1 -> yaxis2/xaxis2)
    const shapes: any[] = [];
    
    if (delineation) {
      const addShapes = (regions: number[][], color: string, name: string) => {
        regions.forEach(([start, end]) => {
          shapes.push({
            type: 'rect',
            xref: 'x2', // Attach to Lead II
            yref: 'y2 domain',
            x0: start / fs,
            x1: end / fs,
            y0: 0,
            y1: 1,
            fillcolor: color,
            opacity: 0.2,
            line: { width: 0 }
          });
        });
      };

      addShapes(delineation.P_Waves || [], '#4ade80', 'P Wave');      // Green
      addShapes(delineation.QRS_Complexes || [], '#3b82f6', 'QRS');   // Blue
      addShapes(delineation.T_Waves || [], '#f97316', 'T Wave');      // Orange
    }

    baseLayout.shapes = shapes;

    // Add explicit text annotations for each lead graph
    const annotations: any[] = [];
    leadOrder.forEach((lead, index) => {
      const yref = `y${index + 1 === 1 ? '' : index + 1} domain`;
      annotations.push({
        xref: 'paper',
        yref: yref,
        x: 0.01,
        y: 0.95,
        text: `<b>Lead ${lead}</b>`,
        showarrow: false,
        font: { color: '#ffffff', size: 13, family: 'Outfit' },
        xanchor: 'left',
        yanchor: 'top',
        bgcolor: 'rgba(15, 17, 26, 0.7)',
        borderpad: 4,
        bordercolor: 'rgba(255,255,255,0.1)',
        borderwidth: 1,
        bordercolor: 'rgba(255, 255, 255, 0.2)'
      });
    });
    
    baseLayout.annotations = annotations;

    return baseLayout;
  }, [delineation, fs]);

  const hasData = waveforms && Object.values(waveforms).some(arr => arr.length > 0);

  if (!hasData) {
    return (
      <div className="glass-panel p-8 text-center text-slate-400 flex flex-col items-center justify-center min-h-[300px]">
        <h3 className="text-xl text-white mb-2">No Waveform Data</h3>
        <p>This record only contains extracted features, or the WFDB header (.hea) was missing.</p>
        <p className="text-sm mt-2 opacity-70">Please upload both .mat and .hea files together for graph visualization.</p>
      </div>
    );
  }

  return (
    <div className="w-full overflow-hidden rounded-xl border border-white/10 bg-black/20">
      <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5">
        <h3 className="font-display font-medium text-white">Interactive ECG Viewer</h3>
        <div className="flex gap-4 text-xs font-mono">
          <span className="flex items-center gap-1"><span className="w-3 h-3 bg-green-400/50 rounded-sm"></span> P-Wave</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-500/50 rounded-sm"></span> QRS</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 bg-orange-500/50 rounded-sm"></span> T-Wave</span>
        </div>
      </div>
      <Plot
        data={plotData}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: true, scrollZoom: true }}
      />
    </div>
  );
};

export default ECGViewer;
