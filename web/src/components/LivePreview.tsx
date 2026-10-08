import { useEffect, useRef, useState } from 'react'

import {
  type Adjustment,
  FRAGMENT_SHADER,
  VERTEX_SHADER,
  adjustPixel,
} from '../lib/livePreview'

interface Props {
  /** The current server preview, which the adjustment is applied on top of. */
  src: string
  /** Where the dragged layer applies (grayscale), if it has a mask. */
  maskSrc?: string
  adjustment: Adjustment
  opacity: number
}

interface Images {
  image: HTMLImageElement
  mask: HTMLImageElement | null
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Could not load ${src}`))
    img.src = src
  })
}

/**
 * Draws the photo with a slider's pending change applied, instantly, while it is dragged.
 * Uses WebGL when available and a (slower) 2D canvas otherwise.
 */
export function LivePreview({ src, maskSrc, adjustment, opacity }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [images, setImages] = useState<Images | null>(null)
  const renderer = useRef<Renderer | null>(null)

  useEffect(() => {
    let live = true
    Promise.all([loadImage(src), maskSrc ? loadImage(maskSrc) : null])
      .then(([image, mask]) => {
        if (live) setImages({ image, mask })
      })
      .catch(() => {
        // No preview; the server render still arrives when the slider is released.
      })
    return () => {
      live = false
    }
  }, [src, maskSrc])

  useEffect(() => {
    const el = canvas.current
    if (!el || !images) return
    if (!renderer.current || renderer.current.images !== images)
      renderer.current = createRenderer(el, images)
    renderer.current?.draw(adjustment, opacity)
  }, [images, adjustment, opacity])

  return (
    <canvas
      ref={canvas}
      className="live-preview"
      aria-hidden="true"
      style={{ visibility: images ? 'visible' : 'hidden' }}
    />
  )
}

interface Renderer {
  images: Images
  draw: (adjustment: Adjustment, opacity: number) => void
}

function createRenderer(
  el: HTMLCanvasElement,
  images: Images,
): Renderer | null {
  el.width = images.image.naturalWidth
  el.height = images.image.naturalHeight
  const gl = el.getContext('webgl', { premultipliedAlpha: false })
  if (gl) {
    const draw = webglRenderer(gl, images)
    if (draw) return { images, draw }
  }
  const ctx = el.getContext('2d')
  return ctx ? { images, draw: canvasRenderer(ctx, images) } : null
}

function compile(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)
  if (!shader) return null
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null
}

function texture(
  gl: WebGLRenderingContext,
  unit: number,
  img: HTMLImageElement,
) {
  const tex = gl.createTexture()
  gl.activeTexture(gl.TEXTURE0 + unit)
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img)
}

function webglRenderer(
  gl: WebGLRenderingContext,
  images: Images,
): Renderer['draw'] | null {
  const vs = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER)
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER)
  const program = gl.createProgram()
  if (!vs || !fs || !program) return null
  gl.attachShader(program, vs)
  gl.attachShader(program, fs)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null
  gl.useProgram(program)

  const buffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  )
  const pos = gl.getAttribLocation(program, 'a_pos')
  gl.enableVertexAttribArray(pos)
  gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0)

  texture(gl, 0, images.image)
  if (images.mask) texture(gl, 1, images.mask)
  const u = (name: string) => gl.getUniformLocation(program, name)
  gl.uniform1i(u('u_image'), 0)
  gl.uniform1i(u('u_mask'), 1)
  gl.uniform1i(u('u_has_mask'), images.mask ? 1 : 0)
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)

  return (a, opacity) => {
    gl.uniform1f(u('u_opacity'), opacity)
    gl.uniform1f(u('u_exposure'), a.exposure)
    gl.uniform3f(u('u_gains'), ...a.gains)
    gl.uniform1f(u('u_highlights'), a.highlights)
    gl.uniform1f(u('u_shadows'), a.shadows)
    gl.uniform1f(u('u_whites'), a.whites)
    gl.uniform1f(u('u_blacks'), a.blacks)
    gl.uniform1f(u('u_contrast'), a.contrast)
    gl.uniform1f(u('u_saturation'), a.saturation)
    gl.uniform1f(u('u_vibrance'), a.vibrance)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
}

function canvasRenderer(
  ctx: CanvasRenderingContext2D,
  images: Images,
): Renderer['draw'] {
  const { width, height } = ctx.canvas
  ctx.drawImage(images.image, 0, 0)
  const source = ctx.getImageData(0, 0, width, height)
  let mask: Uint8ClampedArray | null = null
  if (images.mask) {
    ctx.drawImage(images.mask, 0, 0, width, height)
    mask = ctx.getImageData(0, 0, width, height).data
  }
  return (a, opacity) => {
    const out = ctx.createImageData(width, height)
    const src = source.data
    for (let i = 0; i < src.length; i += 4) {
      const w = opacity * (mask ? mask[i] / 255 : 1)
      const [r, g, b] = adjustPixel(
        [src[i] / 255, src[i + 1] / 255, src[i + 2] / 255],
        a,
        w,
      )
      out.data[i] = r * 255
      out.data[i + 1] = g * 255
      out.data[i + 2] = b * 255
      out.data[i + 3] = 255
    }
    ctx.putImageData(out, 0, 0)
  }
}
