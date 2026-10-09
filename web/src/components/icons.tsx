import type { ReactNode } from 'react'

/** A 16px line icon drawn in the current text color. */
function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {children}
    </svg>
  )
}

/** Scissors. */
export const CutOutIcon = () => (
  <Icon>
    <circle cx="6" cy="6" r="3" />
    <circle cx="6" cy="18" r="3" />
    <path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12" />
  </Icon>
)

/** An eraser. */
export const RemoveIcon = () => (
  <Icon>
    <path d="m7 21-4.3-4.3a1 1 0 0 1 0-1.4l9.6-9.6a1 1 0 0 1 1.4 0l5.6 5.6a1 1 0 0 1 0 1.4L13 21" />
    <path d="M22 21H7M5 11l9 9" />
  </Icon>
)

/** Sparkles. */
export const GenerateIcon = () => (
  <Icon>
    <path d="M10 3 8.5 7.5 4 9l4.5 1.5L10 15l1.5-4.5L16 9l-4.5-1.5z" />
    <path d="M18 14v6M15 17h6M5 18v3M3.5 19.5h3" />
  </Icon>
)

/** Mountains in a frame. */
export const BackgroundIcon = () => (
  <Icon>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="9" cy="9" r="2" />
    <path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21" />
  </Icon>
)

/** The sun. */
export const RelightIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Icon>
)

/** A paintbrush. */
export const RestyleIcon = () => (
  <Icon>
    <path d="M18.4 2.6a2 2 0 0 1 3 3L14 13l-3-3z" />
    <path d="M11 10c-2 0-4 1-4 4 0 2-1.5 3-4 3 1.5 2 3.5 4 7 4 3 0 5-2 5-4.5 0-1.5-.5-2.5-1-3.5" />
  </Icon>
)

/** A clock with an arrow turning back. */
export const RestoreIcon = () => (
  <Icon>
    <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
    <path d="M3 3v5h5M12 7v5l3 2" />
  </Icon>
)

/** A smiling face. */
export const RetouchIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="9" />
    <path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01" />
  </Icon>
)

/** A plus. */
export const NewLayerIcon = () => (
  <Icon>
    <path d="M12 5v14M5 12h14" />
  </Icon>
)
