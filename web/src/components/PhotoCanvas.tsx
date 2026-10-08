import { useEffect, useRef, useState } from 'react'
import type {
  PointerEvent as ReactPointerEvent,
  ReactNode,
  WheelEvent as ReactWheelEvent,
} from 'react'

interface Props {
  /** The edited photo. */
  src: string
  /** The unedited photo, for before/after comparison. */
  beforeSrc: string
  alt: string
  /** Show a "working" veil over the photo, e.g. while the agent edits. */
  busy?: boolean
  /**
   * Drawn exactly over the photo and zoomed with it, e.g. mask handles. Gets the photo's
   * on-screen size before zooming.
   */
  overlay?: (frame: { width: number; height: number }) => ReactNode
}

interface Size {
  width: number
  height: number
}

/** Where an image of `natural` size sits inside `box` with object-fit: contain. */
function containedRect(box: Size, natural: Size) {
  const scale = Math.min(box.width / natural.width, box.height / natural.height)
  const width = natural.width * scale
  const height = natural.height * scale
  return {
    left: (box.width - width) / 2,
    top: (box.height - height) / 2,
    width,
    height,
  }
}

interface View {
  zoom: number
  x: number
  y: number
}

const FIT: View = { zoom: 1, x: 0, y: 0 }
const MIN_ZOOM = 1
const MAX_ZOOM = 8
const COMPARE_KEY = '\\'

function clampZoom(z: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z))
}

function isTyping(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
  )
}

/**
 * Shows the photo fitted to the available space, with zoom (wheel, buttons, double-click),
 * panning by dragging, and two ways to compare with the original: hold the Compare button
 * (or the backslash key), or turn on a split view with a slider.
 */
export function PhotoCanvas({
  src,
  beforeSrc,
  alt,
  busy = false,
  overlay,
}: Props) {
  const [view, setView] = useState<View>(FIT)
  const [holding, setHolding] = useState(false)
  const [split, setSplit] = useState<number | null>(null)
  const [box, setBox] = useState<Size | null>(null)
  const [natural, setNatural] = useState<Size | null>(null)
  const viewport = useRef<HTMLDivElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; view: View } | null>(null)

  useEffect(() => {
    const el = stage.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() =>
      setBox({ width: el.clientWidth, height: el.clientHeight }),
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === COMPARE_KEY && !isTyping(e.target)) setHolding(true)
    }
    const up = (e: KeyboardEvent) => {
      if (e.key === COMPARE_KEY) setHolding(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  /** Zoom by `factor`, keeping the point under (cx, cy) (relative to the center) still. */
  function zoomBy(factor: number, cx = 0, cy = 0) {
    setView((v) => {
      const zoom = clampZoom(v.zoom * factor)
      if (zoom === MIN_ZOOM) return FIT
      const k = zoom / v.zoom
      return { zoom, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k }
    })
  }

  function onWheel(e: ReactWheelEvent) {
    const rect = viewport.current?.getBoundingClientRect()
    const cx = rect ? e.clientX - rect.left - rect.width / 2 : 0
    const cy = rect ? e.clientY - rect.top - rect.height / 2 : 0
    zoomBy(Math.exp(-e.deltaY * 0.002), cx, cy)
  }

  function onPointerDown(e: ReactPointerEvent) {
    if (view.zoom === 1 || e.button !== 0) return
    drag.current = { x: e.clientX, y: e.clientY, view }
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }

  function onPointerMove(e: ReactPointerEvent) {
    const start = drag.current
    if (!start) return
    setView({
      ...start.view,
      x: start.view.x + e.clientX - start.x,
      y: start.view.y + e.clientY - start.y,
    })
  }

  function endDrag() {
    drag.current = null
  }

  const showBefore = holding
  const frame =
    box && natural && natural.width > 0 ? containedRect(box, natural) : null
  return (
    <div className="canvas">
      <div
        ref={viewport}
        className={`viewport${view.zoom > 1 ? ' zoomed' : ''}`}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => (view.zoom > 1 ? setView(FIT) : zoomBy(2))}
      >
        <div
          ref={stage}
          className="stage"
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          }}
        >
          <img
            className="photo"
            src={showBefore ? beforeSrc : src}
            alt={showBefore ? `${alt} (original)` : alt}
            draggable={false}
            onLoad={(e) =>
              setNatural({
                width: e.currentTarget.naturalWidth,
                height: e.currentTarget.naturalHeight,
              })
            }
          />
          {split !== null && !showBefore && (
            <>
              <img
                className="photo before"
                src={beforeSrc}
                alt={`${alt} (original)`}
                draggable={false}
                style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
              />
              <div className="split-line" style={{ left: `${split}%` }} />
            </>
          )}
          {overlay && frame && !showBefore && (
            <div className="photo-frame" style={frame}>
              {overlay({ width: frame.width, height: frame.height })}
            </div>
          )}
        </div>
        {showBefore && <span className="badge">Original</span>}
        {busy && <div className="veil" aria-hidden="true" />}
      </div>

      <div className="canvas-tools">
        <button
          type="button"
          onClick={() => zoomBy(1 / 1.5)}
          disabled={view.zoom <= MIN_ZOOM}
        >
          −
        </button>
        <button
          type="button"
          onClick={() => setView(FIT)}
          aria-label="Fit to window"
        >
          {Math.round(view.zoom * 100)}%
        </button>
        <button
          type="button"
          onClick={() => zoomBy(1.5)}
          disabled={view.zoom >= MAX_ZOOM}
        >
          +
        </button>
        <span className="spacer" />
        {split !== null && (
          <input
            type="range"
            min={0}
            max={100}
            value={split}
            aria-label="Before and after split"
            onChange={(e) => setSplit(Number(e.target.value))}
          />
        )}
        <button
          type="button"
          aria-pressed={split !== null}
          onClick={() => setSplit((s) => (s === null ? 50 : null))}
        >
          Split view
        </button>
        <button
          type="button"
          title="Hold to see the original (or hold the \ key)"
          onPointerDown={() => setHolding(true)}
          onPointerUp={() => setHolding(false)}
          onPointerLeave={() => setHolding(false)}
        >
          Hold to compare
        </button>
      </div>
    </div>
  )
}
