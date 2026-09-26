/** Build a backend URL for local proxying or a separately hosted API. */
export function apiUrl(path: string): string {
  const base = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/$/, '')
  return `${base}${path}`
}
