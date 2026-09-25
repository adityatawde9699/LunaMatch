import type { JobResult, JobSummary, Match, Matcher, RegistrationOptions, Sensor } from '../types'

export async function registerImages(
  source: File,
  reference: File,
  sourceSensor: Sensor,
  referenceSensor: Sensor,
  matcher: Matcher,
  options: RegistrationOptions
): Promise<JobSummary> {
  const form = new FormData()
  form.append('source', source)
  form.append('reference', reference)
  form.append('source_sensor', sourceSensor)
  form.append('reference_sensor', referenceSensor)
  form.append('matcher', matcher)
  form.append('preprocessing', JSON.stringify(options))
  const response = await fetch('/api/v1/register', { method: 'POST', body: form })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Registration failed (${response.status})`)
  }
  return response.json() as Promise<JobSummary>
}

export async function fetchResult(jobId: string): Promise<JobResult> {
  const response = await fetch(`/api/v1/results/${jobId}`)
  if (!response.ok) throw new Error(`Unable to load result (${response.status})`)
  return response.json() as Promise<JobResult>
}

export async function fetchCandidates(jobId: string): Promise<Match[]> {
  const response = await fetch(`/api/v1/results/${jobId}/candidates`)
  if (!response.ok) throw new Error(`Unable to load matches (${response.status})`)
  return response.json() as Promise<Match[]>
}
