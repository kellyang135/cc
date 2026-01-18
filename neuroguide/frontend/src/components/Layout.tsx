import './Layout.css'

interface LayoutProps {
  visualization: React.ReactNode
  chat: React.ReactNode
}

export function Layout({ visualization, chat }: LayoutProps) {
  return (
    <div className="layout">
      <header className="header">
        <h1>NeuroGuide</h1>
        <p>EEG Exploration Assistant</p>
      </header>
      <main className="main">
        <div className="viz-panel">
          {visualization}
        </div>
        <div className="chat-panel">
          {chat}
        </div>
      </main>
    </div>
  )
}
