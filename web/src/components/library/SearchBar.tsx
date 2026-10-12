import { type FormEvent, useState } from 'react'

interface Props {
  onSearch: (query: string) => void
  busy?: boolean
}

/** Search the library in everyday words. */
export function SearchBar({ onSearch, busy = false }: Props) {
  const [query, setQuery] = useState('')
  function submit(e: FormEvent) {
    e.preventDefault()
    if (query.trim()) onSearch(query.trim())
  }
  return (
    <form className="search-bar" role="search" onSubmit={submit}>
      <input
        type="search"
        aria-label="Search photos"
        placeholder="Search, e.g. beach photos from last summer with the dog"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button type="submit" disabled={busy || !query.trim()}>
        {busy ? 'Searching…' : 'Search'}
      </button>
    </form>
  )
}
