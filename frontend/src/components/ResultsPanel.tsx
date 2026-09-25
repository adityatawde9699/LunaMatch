import { useState } from 'react'
import MatchCanvas from './MatchCanvas'
import type { JobResult, JobSummary, Match } from '../types'

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

export default function ResultsPanel({ summary, result, matches }: {
  summary: JobSummary, result: JobResult, matches: Match[]
}) {
  const [split, setSplit] = useState(50)
  const artifact = (name: string) => result.artifacts[name]
  const cards = [
    ['Candidate matches', summary.matches.toLocaleString()],
    ['RANSAC inliers', summary.inliers.toLocaleString()],
    ['Inlier ratio', `${(summary.inlier_ratio * 100).toFixed(1)}%`],
    ['Residual RMSE', `${summary.rmse.toFixed(2)} px`],
    ['Selected coverage', `${(summary.coverage * 100).toFixed(1)}%`],
    ['Runtime', `${summary.runtime_seconds.toFixed(2)} s`]
  ]

  return <section className="results" aria-label="Registration results">
    <div className="section-head"><div><p className="eyebrow">02 / ANALYSIS</p><h2>Registration results</h2></div><span className="status">Completed · {summary.device.toUpperCase()}</span></div>
    <div className="metrics">{cards.map(([label, value]) =>
      <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <p className="scientific-note">RMSE is the RANSAC inlier reprojection residual in reference pixels. Ground-truth registration error has not been measured. Error and confidence maps mark sparse correspondences.</p>

    <div className="comparison panel">
      <div className="panel-heading"><h3>Before / after alignment</h3><span>Drag to compare source and registered view</span></div>
      <div className="comparison-frame">
        <img src={artifact('source_preview.png')} alt="Source image before registration" />
        <img className="comparison-after" src={artifact('registered_preview.png')}
          alt="Source warped into reference coordinates" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
        <div className="comparison-rule" style={{ left: `${split}%` }} />
        <span className="comparison-label before">BEFORE</span><span className="comparison-label after">AFTER</span>
      </div>
      <input type="range" min="0" max="100" value={split} onChange={event => setSplit(Number(event.target.value))}
        aria-label="Before and after alignment slider" />
    </div>

    <div className="visual-grid">{panels.map(([name, label]) =>
      <div className="visual panel" key={name}><div className="panel-heading"><h3>{label}</h3></div>
        <img src={artifact(name)} alt={label} loading="lazy" /></div>)}</div>
    <div className="panel"><MatchCanvas imageUrl={artifact('reference_preview.png')} matches={matches} /></div>

    <div className="exports panel"><div><h3>Export results</h3><p>Registered GeoTIFF pixels, selected correspondences, metrics, and visual evidence.</p></div>
      <div className="export-buttons">
        {([['registered_image.tif', 'Registered TIFF'], ['matches.csv', 'Correspondence CSV'],
           ['metrics.json', 'Metrics JSON'], ['match_visualization.png', 'Match visualization']] as const)
          .map(([name, label]) => <a key={name} href={artifact(name)} download={name}>{label} ↓</a>)}
      </div>
    </div>
  </section>
}
