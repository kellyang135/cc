import Plot from 'react-plotly.js'
import { Data, Layout } from 'plotly.js'

interface SpectrogramData {
  times: number[]
  frequencies: number[]
  power: number[][]  // frequencies x times
  channel: string
}

interface SpectrogramPlotProps {
  data: SpectrogramData | null
}

export function SpectrogramPlot({ data }: SpectrogramPlotProps) {
  if (!data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#6c757d'
      }}>
        No spectrogram data to display
      </div>
    )
  }

  // Convert power to dB scale
  const powerDb = data.power.map((row) =>
    row.map((p) => 10 * Math.log10(p + 1e-10))
  )

  const traces: Data[] = [
    {
      x: data.times,
      y: data.frequencies,
      z: powerDb,
      type: 'heatmap',
      colorscale: 'Viridis',
      colorbar: {
        title: { text: 'Power (dB)', side: 'right' },
      },
    },
  ]

  const layout: Partial<Layout> = {
    title: { text: `Spectrogram - ${data.channel}` },
    xaxis: {
      title: { text: 'Time (s)' },
      showgrid: false,
    },
    yaxis: {
      title: { text: 'Frequency (Hz)' },
      showgrid: false,
    },
    margin: { t: 50, r: 80, b: 60, l: 60 },
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
