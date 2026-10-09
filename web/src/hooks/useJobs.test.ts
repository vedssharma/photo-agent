import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { JobStatus } from '../api/documents'
import { stubApi } from '../test/fixtures'
import { jobText, useJobs } from './useJobs'

const job: JobStatus = {
  id: 'job1',
  doc_id: 'abc123abc123',
  task: 'segment_sky',
  label: 'Finding the sky',
  state: 'running',
  fraction: 0.4,
  message: '',
  backend: 'classical',
}

describe('useJobs', () => {
  it('polls while active and returns the running job', async () => {
    const fetchMock = stubApi({
      'GET /api/documents/abc123abc123/jobs': () => Response.json([job]),
    })
    const { result, rerender } = renderHook(
      ({ active }) => useJobs('abc123abc123', active),
      { initialProps: { active: true } },
    )
    await waitFor(() => expect(result.current).toEqual(job))
    rerender({ active: false })
    expect(result.current).toBeNull()
    const calls = fetchMock.mock.calls.length
    await new Promise((r) => setTimeout(r, 700))
    expect(fetchMock.mock.calls.length).toBe(calls)
  })

  it('does not poll while idle', async () => {
    const fetchMock = stubApi({})
    renderHook(() => useJobs('abc123abc123', false))
    await new Promise((r) => setTimeout(r, 700))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('describes a job with its progress', () => {
    expect(jobText(job)).toBe('Finding the sky… 40%')
    expect(jobText({ ...job, fraction: null })).toBe('Finding the sky…')
  })
})
