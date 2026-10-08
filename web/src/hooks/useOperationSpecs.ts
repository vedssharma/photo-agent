import { useEffect, useState } from 'react'

import { type OperationSpec, fetchOperationSpecs } from '../api/operations'

/** Operation specs keyed by op name; empty until they load. */
export function useOperationSpecs(): Map<string, OperationSpec> {
  const [specs, setSpecs] = useState<Map<string, OperationSpec>>(new Map())
  useEffect(() => {
    let live = true
    fetchOperationSpecs()
      .then((list) => {
        if (live) setSpecs(new Map(list.map((s) => [s.op, s])))
      })
      .catch(() => {
        // Without specs the panels fall back to read-only summaries.
      })
    return () => {
      live = false
    }
  }, [])
  return specs
}
