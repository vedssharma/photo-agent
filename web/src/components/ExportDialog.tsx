import { useEffect, useState } from 'react'

import {
  type DocumentView,
  type ExportOptions,
  type ExportedFile,
  exportDocument,
  saveFile,
} from '../api/documents'

const FORMATS: {
  value: ExportOptions['format']
  name: string
  about: string
}[] = [
  { value: 'jpeg', name: 'JPEG', about: 'smaller, best for sharing' },
  { value: 'png', name: 'PNG', about: 'lossless, larger file' },
  {
    value: 'tiff',
    name: 'TIFF',
    about: 'lossless, for printing or editing elsewhere',
  },
  {
    value: 'avif',
    name: 'AVIF',
    about: 'smallest for the same quality, for the web',
  },
  { value: 'webp', name: 'WebP', about: 'small, for the web' },
]

/** Formats that take a quality setting. */
const LOSSY: ExportOptions['format'][] = ['jpeg', 'avif', 'webp']

interface Props {
  doc: DocumentView
  onClose: () => void
  /** Overridable for tests; defaults to triggering a browser download. */
  save?: (file: ExportedFile) => void
}

/** Choose a format and quality, then render at full resolution and download. */
export function ExportDialog({ doc, onClose, save = saveFile }: Props) {
  const cutout = doc.state.cutout
  const transparent = !!cutout && cutout.visible && !cutout.background
  const [format, setFormat] = useState<ExportOptions['format']>(
    transparent ? 'png' : 'jpeg',
  )
  const [quality, setQuality] = useState(92)
  const [bitDepth, setBitDepth] = useState<ExportOptions['bit_depth']>(
    doc.format === 'RAW' || doc.format === 'TIFF' ? 16 : 8,
  )
  const [keepLocation, setKeepLocation] = useState(false)
  const generated =
    doc.state.framing.some((op) => 'seed' in op) ||
    doc.state.layers.some(
      (layer) =>
        layer.visible &&
        layer.opacity > 0 &&
        layer.operations.some((op) => 'seed' in op),
    )
  const [upscale, setUpscale] = useState<ExportOptions['upscale']>(1)
  const wide = doc.color_space !== 'srgb'
  const [colorSpace, setColorSpace] =
    useState<ExportOptions['color_space']>('srgb')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function download() {
    setWorking(true)
    setError(null)
    try {
      save(
        await exportDocument(doc, {
          format,
          quality,
          bit_depth: bitDepth,
          keep_location: keepLocation,
          upscale,
          color_space: colorSpace,
        }),
      )
      onClose()
    } catch (err) {
      setError(`Export failed: ${(err as Error).message}`)
    } finally {
      setWorking(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="export-title">Download photo</h2>
        <p className="status">
          Saved at full resolution ({doc.width} × {doc.height} before any crop).
        </p>

        <fieldset>
          <legend>Format</legend>
          {FORMATS.map(({ value, name, about }) => (
            <label key={value}>
              <input
                type="radio"
                name="format"
                checked={format === value}
                onChange={() => setFormat(value)}
              />
              {name}{' '}
              <span className="status">
                {transparent && value !== 'jpeg'
                  ? 'keeps the transparent background'
                  : about}
              </span>
            </label>
          ))}
        </fieldset>
        {transparent && format === 'jpeg' && (
          <p className="hint">
            JPEG cannot be transparent, so the background will be white.
          </p>
        )}

        {(format === 'png' || format === 'tiff') && (
          <label className="bit-depth">
            Bits per channel{' '}
            <select
              value={bitDepth}
              onChange={(e) =>
                setBitDepth(
                  Number(e.target.value) as ExportOptions['bit_depth'],
                )
              }
            >
              <option value={8}>8, for viewing and printing</option>
              <option value={16}>16, for further editing</option>
            </select>
          </label>
        )}

        {LOSSY.includes(format) && (
          <label className="quality">
            Quality {quality}
            <input
              type="range"
              min={50}
              max={100}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
            />
          </label>
        )}

        <label className="upscale">
          Enlarge{' '}
          <select
            value={upscale}
            onChange={(e) =>
              setUpscale(Number(e.target.value) as ExportOptions['upscale'])
            }
          >
            <option value={1}>No</option>
            <option value={2}>2× with AI upscaling</option>
            <option value={4}>4× with AI upscaling</option>
          </select>
        </label>
        {upscale > 1 && (
          <p className="hint">
            Good for prints. Enlarging adds new fine detail and takes longer.
          </p>
        )}

        {wide && (
          <>
            <label className="color-space">
              Colors{' '}
              <select
                value={colorSpace}
                onChange={(e) =>
                  setColorSpace(e.target.value as ExportOptions['color_space'])
                }
              >
                <option value="srgb">sRGB, for sharing and the web</option>
                <option value="display-p3">
                  Display P3, for phone and laptop screens
                </option>
                <option value="rec2020">
                  Rec.2020, every color, for further editing
                </option>
              </select>
            </label>
            <p className="hint">
              {colorSpace === 'srgb'
                ? 'This photo has colors beyond sRGB; they are gently toned down to fit.'
                : 'The file keeps colors beyond sRGB and says so in its color profile.'}
            </p>
          </>
        )}

        <label>
          <input
            type="checkbox"
            checked={keepLocation}
            onChange={(e) => setKeepLocation(e.target.checked)}
          />
          Keep GPS location{' '}
          <span className="status">(removed by default for privacy)</span>
        </label>

        {generated && (
          <p className="hint">
            This photo has AI-generated edits, so the file includes Content
            Credentials saying so.
          </p>
        )}

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            onClick={download}
            disabled={working}
          >
            {working ? (upscale > 1 ? 'Enlarging…' : 'Rendering…') : 'Download'}
          </button>
        </div>
      </div>
    </div>
  )
}
