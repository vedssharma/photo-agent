/** Which library photos are selected, and which one has the keyboard focus. */
export interface Selection {
  ids: ReadonlySet<string>
  /** The photo the keyboard and the loupe are on. */
  focus: string | null
  /** Where a shift-click range starts. */
  anchor: string | null
}

export const NOTHING: Selection = { ids: new Set(), focus: null, anchor: null }

export interface ClickKeys {
  /** Extend the selection from the anchor (shift). */
  range?: boolean
  /** Add or remove one photo (ctrl or cmd). */
  toggle?: boolean
}

/** Which modifier keys a click was made with. */
export function clickKeys(e: {
  shiftKey: boolean
  metaKey: boolean
  ctrlKey: boolean
}): ClickKeys {
  return { range: e.shiftKey, toggle: e.metaKey || e.ctrlKey }
}

/** The selection after clicking a photo, the way file managers do it. */
export function clickSelect(
  sel: Selection,
  order: readonly string[],
  id: string,
  keys: ClickKeys = {},
): Selection {
  if (keys.range && sel.anchor && order.includes(sel.anchor)) {
    const a = order.indexOf(sel.anchor)
    const b = order.indexOf(id)
    const span = order.slice(Math.min(a, b), Math.max(a, b) + 1)
    const ids = keys.toggle ? new Set([...sel.ids, ...span]) : new Set(span)
    return { ids, focus: id, anchor: sel.anchor }
  }
  if (keys.toggle) {
    const ids = new Set(sel.ids)
    if (ids.has(id)) ids.delete(id)
    else ids.add(id)
    return { ids, focus: id, anchor: id }
  }
  return { ids: new Set([id]), focus: id, anchor: id }
}

/** Move the focus by `delta` places (a row is `columns` places), selecting it. */
export function moveFocus(
  sel: Selection,
  order: readonly string[],
  delta: number,
  extend = false,
): Selection {
  if (order.length === 0) return NOTHING
  const at = sel.focus ? order.indexOf(sel.focus) : -1
  const next =
    order[Math.max(0, Math.min(order.length - 1, at < 0 ? 0 : at + delta))]
  return clickSelect(sel, order, next, { range: extend })
}

/** Forget photos that are gone (removed, or filtered out of view). */
export function keepOnly(sel: Selection, order: readonly string[]): Selection {
  const present = new Set(order)
  const ids = new Set([...sel.ids].filter((id) => present.has(id)))
  const focus = sel.focus && present.has(sel.focus) ? sel.focus : null
  const anchor = sel.anchor && present.has(sel.anchor) ? sel.anchor : null
  if (ids.size === sel.ids.size && focus === sel.focus && anchor === sel.anchor)
    return sel
  return { ids, focus, anchor }
}

export function selectAll(order: readonly string[], sel: Selection): Selection {
  return {
    ids: new Set(order),
    focus: sel.focus ?? order[0] ?? null,
    anchor: sel.anchor,
  }
}

/** The selected ids in display order. */
export function selectedIds(
  sel: Selection,
  order: readonly string[],
): string[] {
  return order.filter((id) => sel.ids.has(id))
}
