import { type FormEvent, useState } from 'react'

interface Props {
  /** Tags of the photo shown (empty when several are selected). */
  tags: string[]
  onAdd: (tag: string) => void
  onRemove: (tag: string) => void
  /** Show every photo with this tag. */
  onShow: (tag: string) => void
}

/** What is in the photo: tags found when it was indexed, plus any added by hand. */
export function TagEditor({ tags, onAdd, onRemove, onShow }: Props) {
  const [draft, setDraft] = useState('')
  function submit(e: FormEvent) {
    e.preventDefault()
    if (draft.trim()) onAdd(draft.trim())
    setDraft('')
  }
  return (
    <section className="info-tags">
      <h3>Tags</h3>
      {tags.length > 0 && (
        <ul className="chips">
          {tags.map((tag) => (
            <li key={tag}>
              <button
                type="button"
                className="link chip-name"
                onClick={() => onShow(tag)}
                title={`Show every photo tagged ${tag}`}
              >
                {tag}
              </button>
              <button
                type="button"
                className="link"
                aria-label={`Remove tag ${tag}`}
                onClick={() => onRemove(tag)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={submit} className="tag-form">
        <input
          aria-label="Add a tag"
          placeholder="Add a tag"
          value={draft}
          maxLength={40}
          onChange={(e) => setDraft(e.target.value)}
        />
      </form>
    </section>
  )
}
