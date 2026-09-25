import { useState } from 'react'
import MatchCanvas from './MatchCanvas'
import type { JobResult, JobSummary, Match, Matcher, RegistrationOptions, Sensor } from '../types'

const panels = [
  ['source_preview.png', 'Source image'],
  ['reference_preview.png', 'Reference image'],
  ['match_visualization.png', 'Correspondence lines'],
  ['registered_preview.png', 'Registered image'],
  ['overlay.png', 'Registration overlay'],
  ['error_map.png', 'Sparse residual markers'],
  ['confidence_map.png', 'Match confidence map'],
  ['distribution.png', 'Spatial distribution']
] as const

type CompletedRun = {
  sourceName: string, referenceName: string, sourceSensor: Sensor,
  referenceSensor: Sensor, matcher: Matcher, options: RegistrationOptions
}

const matcherNames: Record<Matcher, string> = {
  sift: 'SIFT', orb: 'ORB', akaze: 'AKAZE', loftr: 'LoFTR',
  lightglue: 'SuperPoint + LightGlue', hybrid: 'Hybrid (SIFT + LoFTR)'
}

export default function ResultsPanel({ summary, result, matches, run }: {
  summary: JobSummary, result: JobResult, matches: Match[], run: CompletedRun
}) {
  const [split, setSplit] = useState(50)
  const artifact = (name: string) => result.artifacts[name]
  const cards = [
    ['Candidate matches', summary.matches.toLocaleString()],
    ['RANSAC inliers', summary.inliers.toLocaleString()],
    ['Inlier ratio', `${(summary.inlier_ratio * 100).toFixed(1)}%`],
    ['Residual RMSE', summary.rmse == null ? 'Not available' : `${summary.rmse.toFixed(2)} px`],
    ['Selected points', matches.filter(match => match.selected).length.toLocaleString()],
    ['Selected coverage', `${(summary.coverage * 100).toFixed(1)}%`],
    ['Runtime', `${summary.runtime_seconds.toFixed(2)} s`]
  ]
  const optionalMetric = (key: string) => {
    const value = result.metrics[key]
    return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)} px` : 'Not available'
  }

  return <section className="results" aria-label="Registration results">
    <div className="section-head"><div><p className="eyebrow">02 / ANALYSIS</p><h2>Registration results</h2></div><span className="status">Completed · {summary.device.toUpperCase()}</span></div>
    <details className="run-provenance">
      <summary>Run record <code>{summary.job_id}</code></summary>
      <dl>
        <div><dt>Source</dt><dd>{run.sourceName} · {run.sourceSensor}</dd></div>
        <div><dt>Reference</dt><dd>{run.referenceName} · {run.referenceSensor}</dd></div>
        <div><dt>Matcher / geometry</dt><dd>{matcherNames[run.matcher]} / {run.options.geometry_model}</dd></div>
        <div><dt>Representation</dt><dd>{run.options.iirs_mode === 'pca'
          ? 'IIRS PCA when applicable'
          : `IIRS band ${run.options.source_band} → ${run.options.reference_band} when applicable`}
          {' · '}{run.options.pyramid_levels} pyramid level(s)</dd></div>
        <div><dt>Preprocessing</dt><dd>{[
          run.options.clahe && 'CLAHE', run.options.local_normalization && 'local normalization',
          run.options.gradient && 'gradient', run.options.shadow_mask && 'shadow mask',
          run.options.subpixel_refinement && 'sub-pixel estimate'
        ].filter(Boolean).join(', ') || 'Default intensity representation'}</dd></div>
        <div><dt>Point selection</dt><dd>{run.options.spatial_selection
          ? `${run.options.grid_rows} × ${run.options.grid_cols} grid · max ${run.options.max_matches_per_cell} per cell`
          : 'All geometric inliers'}</dd></div>
      </dl>
    </details>
    <div className="metrics">{cards.map(([label, value]) =>
      <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <p className="scientific-note">Residual RMSE is computed from RANSAC inlier reprojection errors in reference pixels; it is not ground-truth registration accuracy. Ground-truth error has not been measured. Error and confidence maps mark sparse correspondences.</p>
    <div className="residual-details" aria-label="Registration residual details">
      <span>Residual median <strong>{optionalMetric('registration_residual_median_px')}</strong></span>
      <span>Residual 95th percentile <strong>{optionalMetric('registration_residual_p95_px')}</strong></span>
      <span>Ground-truth RMSE <strong>Not measured</strong></span>
    </div>

    <div className="comparison panel">
      <div className="panel-heading"><h3>Reference / registered overlay</h3><span>Both views use reference-image coordinates</span></div>
      {artifact('reference_preview.png') && artifact('overlay.png') ? <>
        <div className="comparison-frame">
          <img src={artifact('reference_preview.png')} alt="Reference image" />
          <img className="comparison-after" src={artifact('overlay.png')}
            alt="Overlay of the warped source and reference" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
          <div className="comparison-rule" style={{ left: `${split}%` }} />
          <span className="comparison-label before">REFERENCE</span><span className="comparison-label after">REGISTERED OVERLAY</span>
        </div>
        <input type="range" min="0" max="100" value={split} onChange={event => setSplit(Number(event.target.value))}
          aria-label="Reference and registered overlay comparison slider" />
      </> : <div className="artifact-unavailable">Comparison images were not saved for this run.</div>}
    </div>

    <div className="visual-grid">{panels.map(([name, label]) =>
      <div className="visual panel" key={name}><div className="panel-heading"><h3>{label}</h3></div>
        {artifact(name) ? <img src={artifact(name)} alt={label} loading="lazy" />
          : <div className="artifact-unavailable">This visualization is unavailable for the run.</div>}</div>)}</div>
    <div className="panel"><MatchCanvas imageUrl={artifact('reference_preview.png')} matches={matches} /></div>

    <div className="exports panel"><div><h3>Export results</h3><p>Registered GeoTIFF pixels, selected correspondences, metrics, and visual evidence.</p></div>
      <div className="export-buttons">
        {([['registered_image.tif', 'Registered TIFF'], ['matches.csv', 'Selected correspondence CSV'],
           ['candidate_matches.csv', 'All candidate matches CSV'], ['metrics.json', 'Metrics JSON'],
           ['transformation.json', 'Transformation JSON'], ['job_log.json', 'Run log JSON'],
           ['match_visualization.png', 'Match visualization']] as const)
          .filter(([name]) => artifact(name))
          .map(([name, label]) => <a key={name} href={artifact(name)} download={name}>{label} ↓</a>)}
      </div>
    </div>
  </section>
}
