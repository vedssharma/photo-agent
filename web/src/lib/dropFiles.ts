/**
 * Files from a drop, walking into any dropped folders (and their subfolders).
 * Browsers list a dropped folder in `files` as one empty entry, so folders are
 * read through the entries API instead.
 */
export async function droppedFiles(transfer: DataTransfer): Promise<File[]> {
  const entries = droppedEntries(transfer)
  if (!entries.some((e) => e.isDirectory)) return Array.from(transfer.files)
  const found: File[] = []
  for (const entry of entries) found.push(...(await walk(entry)))
  return found
}

/** Whether a drop holds a folder or several files (an import, not one photo). */
export function isMultiDrop(transfer: DataTransfer): boolean {
  return (
    transfer.files.length > 1 ||
    droppedEntries(transfer).some((e) => e.isDirectory)
  )
}

function droppedEntries(transfer: DataTransfer): FileSystemEntry[] {
  const items = transfer.items as DataTransferItemList | undefined
  if (!items) return []
  const entries: FileSystemEntry[] = []
  for (let i = 0; i < items.length; i++) {
    const entry = items[i].webkitGetAsEntry?.()
    if (entry) entries.push(entry)
  }
  return entries
}

async function walk(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    return new Promise((resolve) =>
      (entry as FileSystemFileEntry).file(
        (file) => resolve([file]),
        () => resolve([]),
      ),
    )
  }
  if (!entry.isDirectory || entry.name.startsWith('.')) return []
  const reader = (entry as FileSystemDirectoryEntry).createReader()
  const children: FileSystemEntry[] = []
  // readEntries hands back folders in chunks; an empty chunk means the end.
  for (;;) {
    const chunk = await new Promise<FileSystemEntry[]>((resolve) =>
      reader.readEntries(resolve, () => resolve([])),
    )
    if (chunk.length === 0) break
    children.push(...chunk)
  }
  const files: File[] = []
  for (const child of children) files.push(...(await walk(child)))
  return files
}
