import { describe, expect, it } from 'vitest'

import {
  MAX_PROJECTS,
  lastOpenProject,
  memoryProjectStore,
  rememberOpenProject,
} from './projectStore'

const project = (id: string, savedAt: number) => ({
  id,
  filename: `${id}.jpg`,
  savedAt,
  graph: '{}',
  original: new Blob(['x']),
})

describe('memoryProjectStore', () => {
  it('lists newest first and keeps only the most recent projects', async () => {
    const store = memoryProjectStore()
    for (let i = 0; i < MAX_PROJECTS + 2; i++)
      await store.put(project(`p${i}`, i))
    const listed = await store.list()
    expect(listed).toHaveLength(MAX_PROJECTS)
    expect(listed[0]).toEqual({
      id: `p${MAX_PROJECTS + 1}`,
      filename: `p${MAX_PROJECTS + 1}.jpg`,
      savedAt: MAX_PROJECTS + 1,
    })
    expect(await store.get('p0')).toBeUndefined()
    await store.remove(listed[0].id)
    expect(await store.list()).toHaveLength(MAX_PROJECTS - 1)
  })
})

describe('open project memory', () => {
  it('remembers and forgets the open project', () => {
    rememberOpenProject('abc')
    expect(lastOpenProject()).toBe('abc')
    rememberOpenProject(null)
    expect(lastOpenProject()).toBeNull()
  })
})
