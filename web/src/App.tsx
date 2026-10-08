import { useEffect, useState } from 'react'

type Health =
  | { state: 'loading' }
  | { state: 'ok'; version: string }
  | { state: 'error'; message: string }

function App() {
  const [health, setHealth] = useState<Health>({ state: 'loading' })

  useEffect(() => {
    fetch('/api/health')
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const body = (await res.json()) as { version: string }
        setHealth({ state: 'ok', version: body.version })
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
        {health.state === 'error' &&
          `Backend unreachable: ${health.message}`}
      </p>
    </main>
  )
}

export default App
