import { useEffect, useState } from 'react'

import { type JobStatus, fetchJobs } from '../api/documents'

const POLL_MS = 500

/**
 * While `active` (the agent or a manual change is working), polls the document's AI model
 * jobs and returns the one in progress, so the canvas can say what is taking a while
 * ("Finding the sky… 40%"). Returns null when nothing is running.
 */
export function useJobs(docId: string, active: boolean): JobStatus | null {
  const [job, setJob] = useState<JobStatus | null>(null)

  useEffect(() => {
    if (!active) return
    let live = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      try {
        const jobs = await fetchJobs(docId)
        if (!live) return
        setJob(
          jobs.find((j) => j.state === 'running' || j.state === 'queued') ??
            null,
        )
      } catch {
        // Progress is a nicety; the change itself reports its own errors.
      }
      if (live) timer = setTimeout(() => void poll(), POLL_MS)
    }
    timer = setTimeout(() => void poll(), POLL_MS)
    return () => {
      live = false
      clearTimeout(timer)
      setJob(null)
    }
  }, [docId, active])

  return active ? job : null
}

/** "Finding the sky… 40%" */
export function jobText(job: JobStatus): string {
  const percent =
    job.fraction !== null && job.fraction !== undefined
      ? ` ${Math.round(job.fraction * 100)}%`
      : ''
  return `${job.label}…${percent}`
}
