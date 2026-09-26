import type { GroundTruthEvaluation, GroundTruthPoint, JobResult, JobSummary, Match, Matcher, RegistrationOptions, Sensor } from '../types'
import { apiUrl } from './urls'

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
  const response = await fetch(apiUrl('/api/v1/register'), { method: 'POST', body: form })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Registration failed (${response.status})`)
  }
  return response.json() as Promise<JobSummary>
}

export async function fetchResult(jobId: string): Promise<JobResult> {
  const response = await fetch(apiUrl(`/api/v1/results/${jobId}`))
  if (!response.ok) throw new Error(`Unable to load result (${response.status})`)
  const result = await response.json() as JobResult
  result.artifacts = Object.fromEntries(Object.entries(result.artifacts).map(
    ([name, path]) => [name, apiUrl(path)]))
  return result
}

export async function fetchCandidates(jobId: string): Promise<Match[]> {
  const response = await fetch(apiUrl(`/api/v1/results/${jobId}/candidates`))
  if (!response.ok) throw new Error(`Unable to load matches (${response.status})`)
  return response.json() as Promise<Match[]>
}

export async function evaluateGroundTruth(jobId: string, points: GroundTruthPoint[]): Promise<GroundTruthEvaluation> {
  const response = await fetch(apiUrl(`/api/v1/results/${jobId}/ground-truth`), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points })
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Tie-point evaluation failed (${response.status})`)
  }
  return response.json() as Promise<GroundTruthEvaluation>
}
