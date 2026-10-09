import { api } from './client'
import type { components } from './schema'

export type DocumentView = components['schemas']['DocumentView']
export type EditState = components['schemas']['EditState']
export type Layer = components['schemas']['Layer']
export type Cutout = components['schemas']['Cutout']
export type BlendMode = Layer['blend_mode']
export type Mask = NonNullable<Layer['mask']>
export type MaskKind = Mask['kind']
export type FramingOperation = EditState['framing'][number]
export type AdjustmentOperation = Layer['operations'][number]
export type Operation = FramingOperation | AdjustmentOperation
export type StepView = DocumentView['history'][number]
export type ManualEdit = components['schemas']['ManualEdit']
export type JobStatus = components['schemas']['JobStatus']

/** Photo types the backend can decode. HEIC often has no MIME type in browsers. */
export const PHOTO_TYPES =
  '.jpg,.jpeg,.png,.heic,.heif,image/jpeg,image/png,image/heic'

/** Saved projects, for the file dialog. */
export const PROJECT_TYPES = '.photoagent'

const ACCEPTED_EXTENSIONS = /\.(jpe?g|png|heic|heif|photoagent)$/i

/** Saved projects (original photo plus edits) use this extension. */
export function isProjectFile(file: File): boolean {
  return /\.photoagent$/i.test(file.name)
}

export function isSupportedFile(file: File): boolean {
  return (
    ['image/jpeg', 'image/png', 'image/heic', 'image/heif'].includes(
      file.type,
    ) || ACCEPTED_EXTENSIONS.test(file.name)
  )
}

export class ApiError extends Error {}

function detail(error: unknown, status: number): string {
  if (error && typeof error === 'object' && 'detail' in error) {
    const d = (error as { detail: unknown }).detail
    if (typeof d === 'string') return d
  }
  return `The server answered ${status}.`
}

export async function uploadDocument(file: File): Promise<DocumentView> {
  const form = new FormData()
  form.append('file', file, file.name)
  const { data, error, response } = await api.POST('/api/documents', {
    // openapi-fetch would JSON-encode the body; send the form as-is.
    body: form as unknown as { file: string },
    bodySerializer: (body) => body as unknown as FormData,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Reopen a saved `.photoagent` project file. */
export async function openProjectFile(file: File): Promise<DocumentView> {
  const form = new FormData()
  form.append('file', file, file.name)
  return postProject(form)
}

/** Recreate a document from the photo and edits the browser saved for it. */
export async function restoreProject(
  original: Blob,
  filename: string,
  graph: string,
): Promise<DocumentView> {
  const form = new FormData()
  form.append('original', original, filename)
  form.append('graph', graph)
  return postProject(form)
}

async function postProject(form: FormData): Promise<DocumentView> {
  const { data, error, response } = await api.POST('/api/projects', {
    body: form as unknown as Record<string, never>,
    bodySerializer: (body) => body as unknown as FormData,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** The document if the server still has it, or null. */
export async function fetchDocument(
  docId: string,
): Promise<DocumentView | null> {
  const { data, error, response } = await api.GET('/api/documents/{doc_id}', {
    params: { path: { doc_id: docId } },
  })
  if (response.status === 404) return null
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** The stored document (history, layers, chat) as JSON text, for autosave. */
export async function fetchGraph(docId: string): Promise<string> {
  const res = await fetch(
    new Request(
      new URL(`/api/documents/${docId}/graph`, window.location.origin),
    ),
  )
  if (!res.ok) throw new ApiError(`The server answered ${res.status}.`)
  return res.text()
}

/** The original file exactly as uploaded, for autosave. */
export async function fetchSource(docId: string): Promise<Blob> {
  const res = await fetch(
    new Request(
      new URL(`/api/documents/${docId}/source`, window.location.origin),
    ),
  )
  if (!res.ok) throw new ApiError(`The server answered ${res.status}.`)
  return res.blob()
}

/** The project (original plus edits) as a `.photoagent` file to download. */
export async function downloadProject(
  doc: DocumentView,
): Promise<ExportedFile> {
  const res = await fetch(
    new Request(
      new URL(`/api/documents/${doc.id}/project`, window.location.origin),
    ),
  )
  if (!res.ok) throw new ApiError(`The server answered ${res.status}.`)
  const stem = doc.filename.replace(/\.[^.]+$/, '') || 'photo'
  return {
    blob: await res.blob(),
    filename: dispositionFilename(
      res.headers.get('content-disposition'),
      `${stem}.photoagent`,
    ),
  }
}

/** URL of the unedited photo at preview size. */
export function originalUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/original`
}

/** URL of the unedited look with the current crop and rotation, to compare against. */
export function beforeUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/before?revision=${doc.revision}`
}

/** URL of a grayscale image of where a layer applies (white is full effect). */
export function layerMaskUrl(doc: DocumentView, layerId: string): string {
  return `/api/documents/${doc.id}/layers/${layerId}/mask?revision=${doc.revision}`
}

/** URL of the preview with one take on a generative operation (see `offerOptions`). */
export function optionUrl(
  doc: DocumentView,
  opId: string,
  seed: number,
): string {
  return `/api/documents/${doc.id}/operations/${opId}/options/${seed}?revision=${doc.revision}`
}

/** Offer several takes on a generative operation to pick from, as one history step. */
export async function offerOptions(
  docId: string,
  opId: string,
  count = 3,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/operations/{op_id}/options',
    { params: { path: { doc_id: docId, op_id: opId } }, body: { count } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** URL of the edited photo at preview size; changes whenever the edits do. */
export function previewUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/preview?revision=${doc.revision}`
}

/** AI model jobs running (or just finished) for a document. */
export async function fetchJobs(docId: string): Promise<JobStatus[]> {
  const { data, error, response } = await api.GET(
    '/api/documents/{doc_id}/jobs',
    { params: { path: { doc_id: docId } } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function undo(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/undo',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function redo(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/redo',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Add the portrait retouch layers (skin, eyes, teeth) as one step in the history. */
export async function retouchPortrait(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/retouch',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Level the photo and square up converging verticals, as one step in the history. */
export async function autoStraighten(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/straighten',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Record a change made with the manual controls as a step in the history. */
export async function editByHand(
  docId: string,
  edit: ManualEdit,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/edits',
    {
      params: { path: { doc_id: docId } },
      body: edit,
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Show any step in the history, or the original photo when `stepId` is null. */
export async function checkout(
  docId: string,
  stepId: string | null,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/checkout',
    {
      params: { path: { doc_id: docId } },
      body: { step_id: stepId },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export type ExportOptions = components['schemas']['ExportOptions']

export interface ExportedFile {
  blob: Blob
  filename: string
}

/** Parse the download name from a Content-Disposition header. */
export function dispositionFilename(
  header: string | null,
  fallback: string,
): string {
  const encoded = header?.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
  if (encoded) return decodeURIComponent(encoded)
  return header?.match(/filename="?([^";]+)"?/i)?.[1] ?? fallback
}

export async function exportDocument(
  doc: DocumentView,
  options: ExportOptions,
): Promise<ExportedFile> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/export',
    {
      params: { path: { doc_id: doc.id } },
      body: options,
      parseAs: 'blob',
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  const ext = options.format === 'png' ? 'png' : 'jpg'
  return {
    blob: data as Blob,
    filename: dispositionFilename(
      response.headers.get('content-disposition'),
      `photo.${ext}`,
    ),
  }
}

/** Hand a file to the browser's download manager. */
export function saveFile({ blob, filename }: ExportedFile) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}
