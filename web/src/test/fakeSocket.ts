import type { ChatEvent, ChatSocket } from '../hooks/useChat'

/** An in-memory stand-in for the chat WebSocket. */
export class FakeSocket implements ChatSocket {
  static last: FakeSocket | null = null
  readyState = 0
  sent: unknown[] = []
  closed = false
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null

  url: string

  constructor(url: string) {
    this.url = url
    FakeSocket.last = this
  }

  static factory = (url: string) => new FakeSocket(url)

  open() {
    this.readyState = 1
    this.onopen?.(new Event('open'))
  }

  send(data: string) {
    this.sent.push(JSON.parse(data))
  }

  emit(event: ChatEvent) {
    this.onmessage?.(
      new MessageEvent('message', { data: JSON.stringify(event) }),
    )
  }

  close() {
    this.closed = true
    this.readyState = 3
  }

  drop() {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close'))
  }
}
