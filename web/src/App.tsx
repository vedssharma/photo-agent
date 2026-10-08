import { useEffect, useState } from 'react'

import { api } from './api/client'

type Health =
  | { state: 'loading' }
  | { state: 'ok'; version: string; anthropicConfigured: boolean }
  | { state: 'error'; message: string }

function App() {
  const [health, setHealth] = useState<Health>({ state: 'loading' })

  useEffect(() => {
    api
      .GET('/api/health')
      .then(({ data, error, response }) => {
        if (error || !data) throw new Error(`HTTP ${response.status}`)
        setHealth({
          state: 'ok',
          version: data.version,
          anthropicConfigured: data.anthropic_configured,
        })
      })
      .catch((err: unknown) => {
        setHealth({ state: 'error', message: String(err) })
      })
  }, [])

  return (
    <main>
      <h1>photo-agent</h1>
      <p>Describe the edit you want, and the agent does the rest.</p>
      <p className="status">
        {health.state === 'loading' && 'Connecting to the backend…'}
        {health.state === 'ok' && `Backend is up (v${health.version}).`}
        {health.state === 'error' && `Backend unreachable: ${health.message}`}
      </p>
      {health.state === 'ok' && !health.anthropicConfigured && (
        <p className="status">
          No Anthropic API key found. Set <code>ANTHROPIC_API_KEY</code> in{' '}
          <code>.env</code> at the repo root.
        </p>
      )}
    </main>
  )
}

export default App
