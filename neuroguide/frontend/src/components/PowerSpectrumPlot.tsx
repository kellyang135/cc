import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface PowerSpectrumData {
  frequencies: number[]
  power: number[]
  channel: string
}

interface PowerSpectrumPlotProps {
  data: PowerSpectrumData | null
}

export function PowerSpectrumPlot({ data }: PowerSpectrumPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No spectrum data to display
      </div>
    )
  }

  // Convert power to dB scale for better visualization
  const powerDb = data.power.map((p) => 10 * Math.log10(p + 1e-10))

  const traces: Data[] = [
    {
      x: data.frequencies,
      y: powerDb,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#667eea', width: 2 },
      fillcolor: 'rgba(102, 126, 234, 0.2)',
    },
  ]

  // Add frequency band annotations
  const bandAnnotations = [
    { range: [1, 4], label: 'Delta', color: 'rgba(108, 117, 125, 0.1)' },
    { range: [4, 8], label: 'Theta', color: 'rgba(40, 167, 69, 0.1)' },
    { range: [8, 12], label: 'Alpha', color: 'rgba(255, 193, 7, 0.15)' },
    { range: [12, 30], label: 'Beta', color: 'rgba(0, 123, 255, 0.1)' },
    { range: [30, 50], label: 'Gamma', color: 'rgba(220, 53, 69, 0.1)' },
  ]

  const shapes = bandAnnotations.map((band) => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: band.range[0],
    x1: band.range[1],
    y0: 0,
    y1: 1,
    fillcolor: band.color,
    line: { width: 0 },
  }))

  const annotations = bandAnnotations.map((band) => ({
    x: (band.range[0] + band.range[1]) / 2,
    y: 1.05,
    xref: 'x' as const,
    yref: 'paper' as const,
    text: band.label,
    showarrow: false,
    font: { size: 10, color: '#6c757d' },
  }))

  const layout: Partial<Layout> = {
    title: { text: `Power Spectrum - ${data.channel}` },
    xaxis: {
      title: { text: 'Frequency (Hz)' },
      showgrid: true,
      gridcolor: '#e9ecef',
      range: [0, 50],
    },
    yaxis: {
      title: { text: 'Power (dB)' },
      showgrid: true,
      gridcolor: '#e9ecef',
    },
    showlegend: false,
    margin: { t: 60, r: 20, b: 60, l: 60 },
    shapes,
    annotations,
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      style={{ width: '100%', height: '100%' }}
      config={{ responsive: true }}
    />
  )
}
