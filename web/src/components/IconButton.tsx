import {
  type ButtonHTMLAttributes,
  type ReactNode,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'

interface Props extends Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-label' | 'title'
> {
  /** The action's name: the button's accessible name and the tooltip's heading. */
  label: string
  /** A sentence on what the action does, shown under the name. */
  hint?: string
  children: ReactNode
}

/** Keeps a tooltip this far from the window's edges. */
const EDGE = 8

/** A square icon button whose tooltip names the action, on hover or keyboard focus. */
export function IconButton({
  label,
  hint,
  children,
  className,
  ...rest
}: Props) {
  const button = useRef<HTMLButtonElement>(null)
  const tip = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [at, setAt] = useState<{ left: number; top: number } | null>(null)
  const tipId = useId()

  // Placed below the button and kept inside the window, so tooltips at the
  // panel's edges stay readable.
  useLayoutEffect(() => {
    if (!open || !button.current || !tip.current) return
    const b = button.current.getBoundingClientRect()
    const t = tip.current.getBoundingClientRect()
    const centered = b.left + b.width / 2 - t.width / 2
    const left = Math.max(
      EDGE,
      Math.min(centered, window.innerWidth - t.width - EDGE),
    )
    const below = b.bottom + 6
    const top =
      below + t.height > window.innerHeight - EDGE
        ? b.top - t.height - 6
        : below
    setAt({ left, top })
  }, [open])

  const show = () => setOpen(true)
  const hide = () => {
    setOpen(false)
    setAt(null)
  }

  return (
    <>
      <button
        ref={button}
        type="button"
        {...rest}
        className={['icon-button', className].filter(Boolean).join(' ')}
        aria-label={label}
        aria-describedby={hint ? tipId : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={(e) => {
          if (e.currentTarget.matches(':focus-visible')) show()
        }}
        onBlur={hide}
        onKeyDown={(e) => {
          if (e.key === 'Escape' && open) hide()
          rest.onKeyDown?.(e)
        }}
      >
        <span aria-hidden="true">{children}</span>
      </button>
      {/* Always rendered (hidden when closed) so aria-describedby resolves to the hint. */}
      <span
        ref={tip}
        role="tooltip"
        className="tooltip"
        hidden={!open}
        style={
          at
            ? { left: at.left, top: at.top }
            : { left: 0, top: 0, visibility: 'hidden' }
        }
      >
        <strong>{label}</strong>
        {hint && <span id={tipId}>{hint}</span>}
      </span>
    </>
  )
}
