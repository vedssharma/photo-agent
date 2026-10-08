import { useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'

import type { Mask } from '../api/documents'
import type { MaskTool } from '../lib/masks'

type Point = [number, number]
type BrushMask = Extract<Mask, { kind: 'brush' }>
type LinearMask = Extract<Mask, { kind: 'linear' }>
type RadialMask = Extract<Mask, { kind: 'radial' }>

interface Props {
  mask: Mask
  /** Size of the photo on screen before zooming, in CSS pixels. */
  width: number
  height: number
  tool: MaskTool
  /** Grayscale image of where the layer applies, tinted over the photo when shown. */
  maskSrc?: string
  /** Commit the edited mask. */
  onChange: (mask: Mask, label: string) => void
}

const HANDLE_PX = 14
const MIN_STEP = 0.003
const MAX_POINTS = 4000

type Drag =
  | { kind: 'brush'; points: Point[] }
  | { kind: 'line'; handle: 'start' | 'end' }
  | { kind: 'ellipse-new'; center: Point }
  | { kind: 'ellipse-move'; offset: Point }

function toUnit(e: ReactPointerEvent<Element>): Point {
  const rect = e.currentTarget.getBoundingClientRect()
  return [
    (e.clientX - rect.left) / Math.max(rect.width, 1),
    (e.clientY - rect.top) / Math.max(rect.height, 1),
  ]
}

/**
 * Draws the selected layer's mask over the photo and lets the user shape it by dragging:
 * paint brush strokes, place a linear gradient, or draw and move a radial one.
 * Coordinates are fractions of the framed photo, like the server's.
 */
export function MaskOverlay({
  mask,
  width,
  height,
  tool,
  maskSrc,
  onChange,
}: Props) {
  // A local copy while dragging; the committed mask comes back from the server as a prop.
  const [draft, setDraft] = useState<Mask | null>(null)
  const drag = useRef<Drag | null>(null)
  const shown = draft ?? mask
  const px = (p: readonly number[]): Point => [p[0] * width, p[1] * height]
  const near = (a: readonly number[], b: readonly number[]) => {
    const [ax, ay] = px(a)
    const [bx, by] = px(b)
    return Math.hypot(ax - bx, ay - by) <= HANDLE_PX
  }

  function onPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    e.stopPropagation()
    e.currentTarget.setPointerCapture?.(e.pointerId)
    const p = toUnit(e)
    if (mask.kind === 'brush') {
      drag.current = { kind: 'brush', points: [p] }
      setDraft(withStroke(mask, [p], tool))
    } else if (mask.kind === 'linear') {
      const current = (draft ?? mask) as LinearMask
      if (near(p, current.end)) drag.current = { kind: 'line', handle: 'end' }
      else if (near(p, current.start))
        drag.current = { kind: 'line', handle: 'start' }
      else {
        drag.current = { kind: 'line', handle: 'end' }
        setDraft({ ...current, start: p, end: p })
      }
    } else if (mask.kind === 'radial') {
      const current = (draft ?? mask) as RadialMask
      if (near(p, current.center))
        drag.current = {
          kind: 'ellipse-move',
          offset: [p[0] - current.center[0], p[1] - current.center[1]],
        }
      else drag.current = { kind: 'ellipse-new', center: p }
    }
  }

  function onPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    const d = drag.current
    if (!d) return
    e.stopPropagation()
    const p = toUnit(e)
    if (d.kind === 'brush' && mask.kind === 'brush') {
      const last = d.points[d.points.length - 1]
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) < MIN_STEP) return
      if (d.points.length >= MAX_POINTS) return
      d.points.push(p)
      setDraft(withStroke(mask, d.points, tool))
    } else if (d.kind === 'line') {
      setDraft((m) => ({ ...((m ?? mask) as LinearMask), [d.handle]: p }))
    } else if (d.kind === 'ellipse-new') {
      const rx = Math.max(0.01, Math.abs(p[0] - d.center[0]))
      const ry = e.shiftKey
        ? (rx * width) / Math.max(height, 1)
        : Math.max(0.01, Math.abs(p[1] - d.center[1]))
      setDraft({
        ...((draft ?? mask) as RadialMask),
        center: d.center,
        radius_x: Math.min(rx, 2),
        radius_y: Math.min(ry, 2),
      })
    } else if (d.kind === 'ellipse-move') {
      setDraft({
        ...((draft ?? mask) as RadialMask),
        center: [p[0] - d.offset[0], p[1] - d.offset[1]],
      })
    }
  }

  function onPointerUp(e: ReactPointerEvent<HTMLDivElement>) {
    const d = drag.current
    if (!d) return
    e.stopPropagation()
    drag.current = null
    if (!draft) return
    const label =
      d.kind === 'brush'
        ? tool.erase
          ? 'Erase from mask'
          : 'Paint mask'
        : d.kind === 'line'
          ? 'Place gradient'
          : d.kind === 'ellipse-move'
            ? 'Move radial gradient'
            : 'Draw radial gradient'
    onChange(clampMask(draft), label)
  }

  return (
    <div
      className={`mask-overlay ${mask.kind}`}
      data-testid="mask-overlay"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => {
        drag.current = null
        setDraft(null)
      }}
    >
      {tool.show && maskSrc && (
        <div
          className="mask-tint"
          style={{
            maskImage: `url("${maskSrc}")`,
            WebkitMaskImage: `url("${maskSrc}")`,
          }}
        />
      )}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
      >
        {draft?.kind === 'brush' &&
          mask.kind === 'brush' &&
          draft.strokes.length > mask.strokes.length && (
            <StrokePreview
              stroke={draft.strokes[draft.strokes.length - 1]}
              px={px}
              longEdge={Math.max(width, height)}
            />
          )}
        {shown.kind === 'linear' && <LinearGuide mask={shown} px={px} />}
        {shown.kind === 'radial' && (
          <RadialGuide mask={shown} px={px} width={width} height={height} />
        )}
      </svg>
    </div>
  )
}

function StrokePreview({
  stroke,
  px,
  longEdge,
}: {
  stroke: BrushMask['strokes'][number]
  px: (p: readonly number[]) => Point
  longEdge: number
}) {
  return (
    <polyline
      className={`stroke-preview${stroke.erase ? ' erase' : ''}`}
      points={stroke.points.map((p) => px(p).join(',')).join(' ')}
      strokeWidth={stroke.size * 2 * longEdge}
    />
  )
}

function withStroke(mask: BrushMask, points: Point[], tool: MaskTool): Mask {
  return {
    ...mask,
    strokes: [
      ...mask.strokes,
      {
        points: [...points],
        size: tool.size,
        hardness: tool.hardness,
        erase: tool.erase,
      },
    ],
  }
}

/** Keep positions within what the server accepts. */
function clampMask(mask: Mask): Mask {
  const c = (v: number) => Math.min(2, Math.max(-1, v))
  const cp = (p: readonly number[]): Point => [c(p[0]), c(p[1])]
  switch (mask.kind) {
    case 'brush':
      return {
        ...mask,
        strokes: mask.strokes.map((s) => ({ ...s, points: s.points.map(cp) })),
      }
    case 'linear':
      return { ...mask, start: cp(mask.start), end: cp(mask.end) }
    case 'radial':
      return { ...mask, center: cp(mask.center) }
    default:
      return mask
  }
}

function LinearGuide({
  mask,
  px,
}: {
  mask: LinearMask
  px: (p: readonly number[]) => Point
}) {
  const [sx, sy] = px(mask.start)
  const [ex, ey] = px(mask.end)
  // Lines across the photo at both ends, perpendicular to the gradient direction.
  const dx = ex - sx
  const dy = ey - sy
  const len = Math.hypot(dx, dy) || 1
  const nx = (-dy / len) * 4000
  const ny = (dx / len) * 4000
  return (
    <g className="guide">
      <line x1={sx - nx} y1={sy - ny} x2={sx + nx} y2={sy + ny} />
      <line
        x1={ex - nx}
        y1={ey - ny}
        x2={ex + nx}
        y2={ey + ny}
        className="dashed"
      />
      <line x1={sx} y1={sy} x2={ex} y2={ey} />
      <circle cx={sx} cy={sy} r={6} className="handle" />
      <circle cx={ex} cy={ey} r={6} className="handle" />
    </g>
  )
}

function RadialGuide({
  mask,
  px,
  width,
  height,
}: {
  mask: RadialMask
  px: (p: readonly number[]) => Point
  width: number
  height: number
}) {
  const [cx, cy] = px(mask.center)
  const rx = mask.radius_x * width
  const ry = mask.radius_y * height
  const inner = 1 - mask.feather / 100
  return (
    <g className="guide">
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} />
      {inner > 0.02 && inner < 1 && (
        <ellipse
          cx={cx}
          cy={cy}
          rx={rx * inner}
          ry={ry * inner}
          className="dashed"
        />
      )}
      <circle cx={cx} cy={cy} r={6} className="handle" />
    </g>
  )
}
