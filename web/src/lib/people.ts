/** How to show a person who may not have been named yet. */
export const personName = (person: { name?: string | null }) =>
  person.name ?? 'Unnamed'
