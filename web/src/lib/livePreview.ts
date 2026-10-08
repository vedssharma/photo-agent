import type { EditState, Operation } from '../api/documents'
import type { Preview } from './state'

/**
 * Live slider previews. While a slider is dragged, the browser applies the *change* from
 * the saved value to the dragged value on top of the current server preview, with a WebGL
 * shader. That is instant, and close to what the server will render; when the slider is
 * released the server renders the real result, which stays the source of truth.
 *
 * The math mirrors server/src/photo_agent/render.py for the core adjustments.
 */

/** Uniforms for one adjustment; everything at its neutral value changes nothing. */
export interface Adjustment {
  exposure: number
  /** Linear-light gains per channel, from white balance. */
  gains: [number, number, number]
  highlights: number
  shadows: number
  whites: number
  blacks: number
  contrast: number
  /** Multiplier on chroma; 1 is no change. */
  saturation: number
  vibrance: number
}

export const NEUTRAL: Adjustment = {
  exposure: 0,
  gains: [1, 1, 1],
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  contrast: 0,
  saturation: 1,
  vibrance: 0,
}

/** Operations the live preview can show. Others update when the slider is released. */
export const LIVE_OPS = new Set([
  'exposure',
  'contrast',
  'highlights',
  'shadows',
  'whites',
  'blacks',
  'white_balance',
  'saturation',
  'vibrance',
])

const LUMA: [number, number, number] = [0.2126, 0.7152, 0.0722]

function wbGains(temperature: number, tint: number): [number, number, number] {
  const t = temperature / 100
  const g = tint / 100
  const gains: [number, number, number] = [
    1 + 0.25 * t,
    1 - 0.2 * g,
    1 - 0.25 * t,
  ]
  const norm = gains[0] * LUMA[0] + gains[1] * LUMA[1] + gains[2] * LUMA[2]
  return gains.map((v) => v / norm) as [number, number, number]
}

function num(op: Operation, key: string): number {
  const v = (op as unknown as Record<string, unknown>)[key]
  return typeof v === 'number' ? v : 0
}

/** The adjustment that turns `before` into `after` (same kind of operation). */
export function deltaAdjustment(
  before: Operation,
  after: Operation,
): Adjustment | null {
  if (!LIVE_OPS.has(before.op) || before.op !== after.op) return null
  const d = (key: string) => (num(after, key) - num(before, key)) / 100
  switch (before.op) {
    case 'exposure':
      return {
        ...NEUTRAL,
        exposure: num(after, 'stops') - num(before, 'stops'),
      }
    case 'white_balance': {
      const a = wbGains(num(after, 'temperature'), num(after, 'tint'))
      const b = wbGains(num(before, 'temperature'), num(before, 'tint'))
      return { ...NEUTRAL, gains: [a[0] / b[0], a[1] / b[1], a[2] / b[2]] }
    }
    case 'saturation': {
      const b = 1 + num(before, 'amount') / 100
      const a = 1 + num(after, 'amount') / 100
      // From fully desaturated there is no color left to scale; show the target instead.
      return { ...NEUTRAL, saturation: b > 1e-3 ? a / b : a }
    }
    case 'contrast':
    case 'highlights':
    case 'shadows':
    case 'whites':
    case 'blacks':
    case 'vibrance':
      return { ...NEUTRAL, [before.op]: d('amount') }
    default:
      return null
  }
}

export interface LiveTarget {
  adjustment: Adjustment
  /** Layer opacity, 0..1. */
  opacity: number
  /** The layer whose mask limits the preview, or null for none. */
  maskLayerId: string | null
}

/** Work out what to draw for a slider being dragged, or null if it cannot be previewed. */
export function liveTarget(
  state: EditState,
  preview: Preview,
): LiveTarget | null {
  if (preview.layerId === null) return null // framing changes the geometry
  const layer = state.layers.find((l) => l.id === preview.layerId)
  if (!layer || !layer.visible) return null
  const op = layer.operations.find((o) => o.id === preview.opId)
  if (!op) return null
  const after = { ...op, [preview.param]: preview.value } as Operation
  const adjustment = deltaAdjustment(op, after)
  if (!adjustment) return null
  return {
    adjustment,
    opacity: layer.opacity / 100,
    maskLayerId: layer.mask ? layer.id : null,
  }
}

// CPU reference of the shader, used when WebGL is unavailable and in tests.

const toLin = (c: number) => {
  c = Math.max(c, 0)
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}
const toSrgb = (c: number) => {
  c = Math.max(c, 0)
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
}
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const smooth = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0))
  return t * t * (3 - 2 * t)
}
const lumaOf = (c: number[]) => c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]

function gainLinear(c: number[], stops: number[]): number[] {
  return c.map((v, i) => toSrgb(toLin(v) * 2 ** stops[i]))
}

function hueSat(c: number[]): [number, number] {
  const [r, g, b] = c.map(clamp01)
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const sat = max > 0 ? (max - min) / max : 0
  if (max === min) return [0, sat]
  let h: number
  if (max === r) h = ((g - b) / (max - min)) % 6
  else if (max === g) h = (b - r) / (max - min) + 2
  else h = (r - g) / (max - min) + 4
  return [(h * 60 + 360) % 360, sat]
}

/** Apply an adjustment to one sRGB pixel, blended in by `weight` (opacity × mask). */
export function adjustPixel(
  rgb: [number, number, number],
  a: Adjustment,
  weight = 1,
): [number, number, number] {
  let x: number[] = [...rgb]
  const e = a.exposure
  x = gainLinear(x, [e, e, e])
  x = x.map((v, i) => toSrgb(toLin(v) * a.gains[i]))
  let l = clamp01(lumaOf(x))
  const hs = smooth(0.45, 1, l) * a.highlights * 1.2
  x = gainLinear(x, [hs, hs, hs])
  l = clamp01(lumaOf(x))
  const sh = (1 - smooth(0, 0.55, l)) * a.shadows * 1.5
  x = gainLinear(x, [sh, sh, sh])
  l = clamp01(lumaOf(x))
  const wh = 1 + a.whites * 0.35 * smooth(0.35, 1, l)
  x = x.map((v) => v * wh)
  x = x.map((v) => v + a.blacks * 0.12 * (1 - smooth(0, 0.4, clamp01(v))))
  x = x.map((v) => {
    const c = clamp01(v)
    const y =
      a.contrast >= 0
        ? c + a.contrast * (c * c * (3 - 2 * c) - c)
        : 0.5 + (c - 0.5) * (1 + 0.6 * a.contrast)
    return v + (y - c)
  })
  l = lumaOf(x)
  x = x.map((v) => l + (v - l) * a.saturation)
  if (a.vibrance !== 0) {
    const [hue, sat] = hueSat(x)
    const hueW = clamp01(1 - Math.abs(hue - 25) / 25)
    const skin = hueW * smooth(0.1, 0.25, sat) * (1 - smooth(0.55, 0.8, sat))
    const factor = 1 + a.vibrance * 1.2 * (1 - sat) * (1 - 0.6 * skin)
    l = lumaOf(x)
    x = x.map((v) => l + (v - l) * factor)
  }
  return x.map((v, i) => clamp01(rgb[i] + (v - rgb[i]) * weight)) as [
    number,
    number,
    number,
  ]
}

export const VERTEX_SHADER = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = vec2((a_pos.x + 1.0) * 0.5, (1.0 - a_pos.y) * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`

export const FRAGMENT_SHADER = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_image;
uniform sampler2D u_mask;
uniform bool u_has_mask;
uniform float u_opacity;
uniform float u_exposure;
uniform vec3 u_gains;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_vibrance;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float luma(vec3 c) { return dot(c, LUMA); }
vec3 toLin(vec3 c) {
  c = max(c, 0.0);
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 toSrgb(vec3 c) {
  c = max(c, 0.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 gain(vec3 c, float stops) { return toSrgb(toLin(c) * exp2(stops)); }

vec2 hueSat(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float sat = mx > 0.0 ? (mx - mn) / mx : 0.0;
  if (mx == mn) return vec2(0.0, sat);
  float h;
  if (mx == c.r) h = mod((c.g - c.b) / (mx - mn), 6.0);
  else if (mx == c.g) h = (c.b - c.r) / (mx - mn) + 2.0;
  else h = (c.r - c.g) / (mx - mn) + 4.0;
  return vec2(mod(h * 60.0 + 360.0, 360.0), sat);
}

void main() {
  vec3 orig = texture2D(u_image, v_uv).rgb;
  vec3 x = gain(orig, u_exposure);
  x = toSrgb(toLin(x) * u_gains);
  float l = clamp(luma(x), 0.0, 1.0);
  x = gain(x, smoothstep(0.45, 1.0, l) * u_highlights * 1.2);
  l = clamp(luma(x), 0.0, 1.0);
  x = gain(x, (1.0 - smoothstep(0.0, 0.55, l)) * u_shadows * 1.5);
  l = clamp(luma(x), 0.0, 1.0);
  x *= 1.0 + u_whites * 0.35 * smoothstep(0.35, 1.0, l);
  x += u_blacks * 0.12 * (1.0 - smoothstep(vec3(0.0), vec3(0.4), clamp(x, 0.0, 1.0)));
  vec3 c = clamp(x, 0.0, 1.0);
  vec3 y = u_contrast >= 0.0
    ? c + u_contrast * (c * c * (3.0 - 2.0 * c) - c)
    : 0.5 + (c - 0.5) * (1.0 + 0.6 * u_contrast);
  x += y - c;
  l = luma(x);
  x = l + (x - l) * u_saturation;
  if (u_vibrance != 0.0) {
    vec2 hs = hueSat(x);
    float hueW = clamp(1.0 - abs(hs.x - 25.0) / 25.0, 0.0, 1.0);
    float skin = hueW * smoothstep(0.1, 0.25, hs.y) * (1.0 - smoothstep(0.55, 0.8, hs.y));
    float factor = 1.0 + u_vibrance * 1.2 * (1.0 - hs.y) * (1.0 - 0.6 * skin);
    l = luma(x);
    x = l + (x - l) * factor;
  }
  float w = u_opacity;
  if (u_has_mask) w *= texture2D(u_mask, v_uv).r;
  gl_FragColor = vec4(clamp(orig + (x - orig) * w, 0.0, 1.0), 1.0);
}
`
