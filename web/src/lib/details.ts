import type { LibraryPhoto } from '../api/library'

const when = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

/** When a photo was taken, as the camera's clock said (EXIF has no time zone). */
export function takenText(takenAt: string): string {
  // Read it as local time: it is the camera's wall clock, not a UTC instant.
  const [date, time = '00:00:00'] = takenAt.split('T')
  const [y, m, d] = date.split('-').map(Number)
  const [hh, mm, ss] = time.split(':').map(Number)
  return when.format(new Date(y, m - 1, d, hh, mm, ss || 0))
}

/** "1/250 s" for short exposures, "2 s" for long ones. */
export function shutterText(seconds: number): string {
  if (seconds >= 0.5) return `${Number(seconds.toFixed(1))} s`
  return `1/${Math.round(1 / seconds)} s`
}

/** Aperture, shutter, ISO, and focal length on one line, as cameras show them. */
export function settingsText(p: LibraryPhoto): string | null {
  const parts = [
    p.f_number ? `ƒ/${Number(p.f_number.toFixed(1))}` : null,
    p.exposure_time ? shutterText(p.exposure_time) : null,
    p.iso ? `ISO ${p.iso}` : null,
    p.focal_length ? `${Number(p.focal_length.toFixed(1))} mm` : null,
  ].filter(Boolean)
  return parts.length ? parts.join(' · ') : null
}

export function sizeText(p: LibraryPhoto): string {
  const mb = p.file_size / (1024 * 1024)
  const megapixels = (p.width * p.height) / 1e6
  return `${p.width} × ${p.height} (${megapixels.toFixed(1)} MP), ${mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(p.file_size / 1024))} KB`}`
}

export function coordinatesText(lat: number, lon: number): string {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? 'E' : 'W'}`
}

export function mapUrl(lat: number, lon: number): string {
  return `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=13/${lat}/${lon}`
}
