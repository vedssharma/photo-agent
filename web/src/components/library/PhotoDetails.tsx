import type { LibraryPhoto } from '../../api/library'
import {
  coordinatesText,
  mapUrl,
  settingsText,
  sizeText,
  takenText,
} from '../../lib/details'

/** What the file says about a photo: when, with what, how, and where. */
export function PhotoDetails({ photo }: { photo: LibraryPhoto }) {
  const settings = settingsText(photo)
  const located = photo.latitude != null && photo.longitude != null
  return (
    <section className="info-details">
      <h3>Details</h3>
      <dl>
        <dt>Taken</dt>
        <dd>{photo.taken_at ? takenText(photo.taken_at) : 'Unknown'}</dd>
        {photo.camera && (
          <>
            <dt>Camera</dt>
            <dd>{photo.camera}</dd>
          </>
        )}
        {photo.lens && (
          <>
            <dt>Lens</dt>
            <dd>{photo.lens}</dd>
          </>
        )}
        {settings && (
          <>
            <dt>Settings</dt>
            <dd>{settings}</dd>
          </>
        )}
        <dt>Size</dt>
        <dd>{sizeText(photo)}</dd>
        {located && (
          <>
            <dt>Location</dt>
            <dd>
              {coordinatesText(photo.latitude!, photo.longitude!)}{' '}
              <a
                href={mapUrl(photo.latitude!, photo.longitude!)}
                target="_blank"
                rel="noreferrer noopener"
              >
                Map
              </a>
            </dd>
          </>
        )}
      </dl>
    </section>
  )
}
