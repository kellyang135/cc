import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface TimeSeriesData {
  data: number[][]  // channels x samples
  times: number[]
  channels: string[]
  sample_rate: number
}

interface Highlight {
  start_time: number
  end_time: number
  label: string
}

interface TimeSeriesPlotProps {
  data: TimeSeriesData | null
  highlights?: Highlight[]
}

export function TimeSeriesPlot({ data, highlights = [] }: TimeSeriesPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No signal data to display
      </div>
    )
  }

  // Create traces for each channel with vertical offset
  const traces: Data[] = data.channels.map((channel, i) => {
    // Normalize and offset each channel for display
    const channelData = data.data[i]
    const mean = channelData.reduce((a, b) => a + b, 0) / channelData.length
    const std = Math.sqrt(
      channelData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / channelData.length
    )
    const normalized = channelData.map((v) => (v - mean) / (std || 1))
    const offset = (data.channels.length - 1 - i) * 3  // Offset each channel

    return {
      x: data.times,
      y: normalized.map((v) => v + offset),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: channel,
      line: { width: 1 },
    }
  })

  // Add highlight shapes
  const shapes = highlights.map((h) => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: h.start_time,
    x1: h.end_time,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(255, 193, 7, 0.3)',
    line: { width: 0 },
  }))

  // Add highlight annotations
  const annotations = highlights.map((h) => ({
    x: (h.start_time + h.end_time) / 2,
    y: 1,
    xref: 'x' as const,
    yref: 'paper' as const,
    text: h.label,
    showarrow: false,
    font: { size: 12, color: '#856404' },
    bgcolor: 'rgba(255, 243, 205, 0.9)',
    borderpad: 4,
  }))

  const layout: Partial<Layout> = {
    title: { text: 'EEG Signals' },
    xaxis: {
      title: { text: 'Time (s)' },
      showgrid: true,
      gridcolor: '#e9ecef',
    },
    yaxis: {
      title: { text: '' },
      showticklabels: false,
      showgrid: false,
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.15,
    },
    margin: { t: 50, r: 20, b: 80, l: 50 },
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
