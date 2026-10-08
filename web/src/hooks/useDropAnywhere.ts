import { useEffect, useRef } from 'react'

/** Accept files dropped anywhere in the window, and report when a drag is over it. */
export function useDropAnywhere(
  onDragging: (dragging: boolean) => void,
  onDrop: (files: FileList) => void,
) {
  const dropRef = useRef(onDrop)
  const dragRef = useRef(onDragging)
  useEffect(() => {
    dropRef.current = onDrop
    dragRef.current = onDragging
  })

  useEffect(() => {
    let depth = 0
    const hasFiles = (e: DragEvent) =>
      e.dataTransfer?.types.includes('Files') ?? false
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth += 1
      dragRef.current(true)
    }
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth = Math.max(0, depth - 1)
      if (depth === 0) dragRef.current(false)
    }
    const over = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault()
    }
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth = 0
      dragRef.current(false)
      if (e.dataTransfer?.files.length) dropRef.current(e.dataTransfer.files)
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
    }
  }, [])
}
