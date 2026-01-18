import { useState } from 'react'
import { Layout } from './components/Layout'
import { ChatPanel } from './components/ChatPanel'
import { TimeSeriesPlot } from './components/TimeSeriesPlot'
import { PowerSpectrumPlot } from './components/PowerSpectrumPlot'
import { SpectrogramPlot } from './components/SpectrogramPlot'
import './index.css'

type VisualizationType = 'none' | 'time_series' | 'power_spectrum' | 'spectrogram'

interface Visualization {
  type: VisualizationType
  data: unknown
}

interface Highlight {
  start_time: number
  end_time: number
  label: string
}

function App() {
  const [visualization, setVisualization] = useState<Visualization>({ type: 'none', data: null })
  const [highlights, setHighlights] = useState<Highlight[]>([])

  const handleVisualization = (viz: { type: string; data: unknown }) => {
    if (viz.type === 'highlight') {
      // Add highlight to existing visualization
      setHighlights((prev) => [...prev, viz.data as Highlight])
    } else {
      // Update main visualization and clear highlights
      setVisualization({ type: viz.type as VisualizationType, data: viz.data })
      setHighlights([])
    }
  }

  const renderVisualization = () => {
    switch (visualization.type) {
      case 'time_series':
        return (
          <TimeSeriesPlot
            data={visualization.data as Parameters<typeof TimeSeriesPlot>[0]['data']}
            highlights={highlights}
          />
        )
      case 'power_spectrum':
        return (
          <PowerSpectrumPlot
            data={visualization.data as Parameters<typeof PowerSpectrumPlot>[0]['data']}
          />
        )
      case 'spectrogram':
        return (
          <SpectrogramPlot
            data={visualization.data as Parameters<typeof SpectrogramPlot>[0]['data']}
          />
        )
      default:
        return (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#6c757d',
            textAlign: 'center',
            padding: '2rem',
          }}>
            <h2 style={{ marginBottom: '1rem', color: '#495057' }}>
              Welcome to NeuroGuide
            </h2>
            <p style={{ maxWidth: '400px', lineHeight: 1.6 }}>
              Start a conversation in the chat panel to explore EEG data.
              I'll guide you through understanding brain signals step by step.
            </p>
          </div>
        )
    }
  }

  return (
    <Layout
      visualization={renderVisualization()}
      chat={<ChatPanel onVisualization={handleVisualization} />}
    />
  )
}

export default App
