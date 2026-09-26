import { useState } from 'react'
import type { MouseEvent } from 'react'
import { evaluateGroundTruth } from '../services/api'
import type { GroundTruthEvaluation, GroundTruthPoint } from '../types'

type PixelPoint = { x: number, y: number }

function clickedPixel(event: MouseEvent<HTMLImageElement>): PixelPoint {
  const image = event.currentTarget
  const rect = image.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left) * image.naturalWidth / rect.width,
    y: (event.clientY - rect.top) * image.naturalHeight / rect.height
  }
}

export default function GroundTruthAnnotator({ jobId, sourceUrl, referenceUrl }: {
  jobId: string, sourceUrl?: string, referenceUrl?: string
}) {
  const [pending, setPending] = useState<PixelPoint | null>(null)
  const [points, setPoints] = useState<GroundTruthPoint[]>([])
  const [report, setReport] = useState<GroundTruthEvaluation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [sourceSize, setSourceSize] = useState<PixelPoint | null>(null)
  const [referenceSize, setReferenceSize] = useState<PixelPoint | null>(null)

  function addReference(event: MouseEvent<HTMLImageElement>) {
    if (!pending) return
    const reference = clickedPixel(event)
    setPoints(current => [...current, { id: current.length + 1,
      source_x: pending.x, source_y: pending.y,
      reference_x: reference.x, reference_y: reference.y }])
    setPending(null)
    setReport(null)
  }

  function downloadCsv() {
    const columns = 'id,source_x,source_y,reference_x,reference_y\n'
    const rows = points.map(point => [point.id, point.source_x, point.source_y,
      point.reference_x, point.reference_y].join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([columns + rows + '\n'], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `lunamatch_${jobId}_tie_points.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  async function evaluate() {
    setRunning(true)
    setError(null)
    try {
      setReport(await evaluateGroundTruth(jobId, points))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Evaluation failed')
    } finally {
      setRunning(false)
    }
  }

  if (!sourceUrl || !referenceUrl) return null
  const markers = (side: 'source' | 'reference', size: PixelPoint | null) => size &&
    <svg className="tie-point-markers" viewBox={`0 0 ${size.x} ${size.y}`} aria-hidden="true">
      {points.map(point => <g key={point.id} transform={`translate(${point[`${side}_x`]},${point[`${side}_y`]})`}>
        <circle r="5" fill="#a6dce0" stroke="#0a0e16" strokeWidth="2" />
        <text x="8" y="-8" fill="white" stroke="#0a0e16" strokeWidth="2" paintOrder="stroke">{point.id}</text>
      </g>)}
      {side === 'source' && pending && <circle cx={pending.x} cy={pending.y} r="7"
        fill="none" stroke="#ffd78c" strokeWidth="2" />}
    </svg>
  return <section className="ground-truth panel" aria-label="Independent tie-point evaluation">
    <div className="panel-heading"><h3>Independent tie points</h3><span>{points.length} point pair(s)</span></div>
    <p>Click a feature in the source, then the same feature in the reference. Use crater centers or sharp terrain corners. These points are evaluated against the saved transform and are never used to fit it.</p>
    <div className="tie-point-grid">
      <div><strong>1 · Source</strong><div className="tie-image"><img src={sourceUrl} alt="Source for manual tie-point annotation"
        onLoad={event => setSourceSize({ x: event.currentTarget.naturalWidth, y: event.currentTarget.naturalHeight })}
        onClick={event => { setPending(clickedPixel(event)); setReport(null) }} />
        {markers('source', sourceSize)}</div></div>
      <div><strong>2 · Reference</strong><div className="tie-image"><img src={referenceUrl} alt="Reference for manual tie-point annotation"
        onLoad={event => setReferenceSize({ x: event.currentTarget.naturalWidth, y: event.currentTarget.naturalHeight })}
        onClick={addReference} />{markers('reference', referenceSize)}</div></div>
    </div>
    <p aria-live="polite">{pending
      ? `Source (${pending.x.toFixed(1)}, ${pending.y.toFixed(1)}) selected. Click the matching reference feature.`
      : 'Select a source feature to add a pair.'}</p>
    <div className="tie-point-actions">
      <button type="button" onClick={() => { setPoints(current => current.slice(0, -1)); setReport(null) }} disabled={!points.length}>Undo last pair</button>
      <button type="button" onClick={downloadCsv} disabled={!points.length}>Export tie-point CSV</button>
      <button type="button" onClick={evaluate} disabled={!points.length || running}>{running ? 'Evaluating…' : 'Evaluate saved transform'}</button>
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {report && <div className="ground-truth-report" aria-live="polite">
      <strong>Supplied-point transform error</strong>
      <span>RMSE {report.ground_truth_rmse_px.toFixed(2)} px</span>
      <span>Median {report.ground_truth_median_px.toFixed(2)} px</span>
      <span>95th percentile {report.ground_truth_p95_px.toFixed(2)} px</span>
      <span>{report.point_count} point(s) · user supplied, independently verified status unknown</span>
      <a href={`/api/v1/results/${jobId}/artifacts/ground_truth_evaluation.json`} download>Download evaluation JSON</a>
      <a href={`/api/v1/results/${jobId}/artifacts/ground_truth_points.csv`} download>Download evaluated points CSV</a>
    </div>}
  </section>
}
