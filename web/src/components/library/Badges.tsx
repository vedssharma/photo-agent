import type { LibraryPhoto } from '../../api/library'

/** Rating, flag, and color label drawn over a thumbnail. */
export function Badges({ photo }: { photo: LibraryPhoto }) {
  return (
    <>
      {photo.flag !== 'none' && (
        <span
          className={`thumb-flag ${photo.flag}`}
          aria-label={photo.flag === 'picked' ? 'Picked' : 'Rejected'}
          title={photo.flag === 'picked' ? 'Picked' : 'Rejected'}
        >
          {photo.flag === 'picked' ? '⚑' : '✕'}
        </span>
      )}
      {(photo.rating > 0 || photo.label) && (
        <span className="thumb-marks">
          {photo.rating > 0 && (
            <span className="thumb-stars" aria-label={`${photo.rating} stars`}>
              {'★'.repeat(photo.rating)}
            </span>
          )}
          {photo.label && (
            <span
              className={`label-dot ${photo.label}`}
              aria-label={`${photo.label} label`}
            />
          )}
        </span>
      )}
    </>
  )
}
