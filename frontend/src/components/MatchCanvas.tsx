import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, MouseEvent } from 'react'
import type { Match } from '../types'

type Filter = 'all' | 'inliers' | 'high'

function confidenceColor(confidence: number): string {
  const hue = Math.round(Math.max(0, Math.min(1, confidence)) * 120)
  return `hsl(${hue} 72% 58%)`
}

function visibleForFilter(match: Match, filter: Filter): boolean {
  return filter === 'all' || (filter === 'inliers' && match.inlier) ||
    (filter === 'high' && match.inlier && match.confidence >= 0.7)
}

export default function MatchCanvas({ imageUrl, matches }: { imageUrl?: string, matches: Match[] }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [filter, setFilter] = useState<Filter>('inliers')
  const [focusedMatch, setFocusedMatch] = useState<Match | null>(null)
  const visible = useMemo(() => matches.filter(match => visibleForFilter(match, filter)), [matches, filter])

  useEffect(() => {
    if (!imageUrl) return
    const image = new Image()
    image.onload = () => {
      const surface = canvas.current
      const context = surface?.getContext('2d')
      if (!surface || !context) return
      surface.width = image.naturalWidth
      surface.height = image.naturalHeight
      context.drawImage(image, 0, 0)
      for (const match of visible) {
        const x = match.reference_x
        const y = match.reference_y
        context.beginPath()
        context.arc(x, y, match.selected ? 4 : 3, 0, Math.PI * 2)
        context.fillStyle = match.inlier ? confidenceColor(match.confidence) : '#ee6b79'
        context.fill()
        if (focusedMatch?.id === match.id) {
          context.beginPath()
          context.arc(x, y, 8, 0, Math.PI * 2)
          context.strokeStyle = '#ffffff'
          context.lineWidth = 2
          context.stroke()
        }
      }
    }
    image.onerror = () => {
      const context = canvas.current?.getContext('2d')
      if (context && canvas.current) {
        context.clearRect(0, 0, canvas.current.width, canvas.current.height)
      }
    }
    image.src = imageUrl
    return () => {
      image.onload = null
      image.onerror = null
    }
  }, [imageUrl, visible, focusedMatch])

  function inspectPoint(event: MouseEvent<HTMLCanvasElement>) {
    const surface = canvas.current
    if (!surface) return
    const rect = surface.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const x = (event.clientX - rect.left) / rect.width * surface.width
    const y = (event.clientY - rect.top) / rect.height * surface.height
    const tolerance = 12 / rect.width * surface.width
    let closest: Match | null = null
    let minDistance = tolerance * tolerance
    for (const match of visible) {
      const dx = match.reference_x - x
      const dy = match.reference_y - y
      const distance = dx * dx + dy * dy
      if (distance < minDistance) {
        minDistance = distance
        closest = match
      }
    }
    setFocusedMatch(closest)
  }

  function navigateMatches(event: KeyboardEvent<HTMLCanvasElement>) {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    if (!visible.length) return
    const currentIndex = visible.findIndex(match => match.id === focusedMatch?.id)
    const direction = event.key === 'ArrowRight' ? 1 : -1
    const nextIndex = currentIndex < 0
      ? (direction > 0 ? 0 : visible.length - 1)
      : (currentIndex + direction + visible.length) % visible.length
    setFocusedMatch(visible[nextIndex])
  }

  const confidenceText = focusedMatch ? `${(focusedMatch.confidence * 100).toFixed(1)}%` : '—'

  return <div className="match-canvas">
    <div className="panel-heading"><div><h3>Correspondences by confidence</h3><span>{visible.length.toLocaleString()} shown · click a point to inspect</span></div>
      <div className="segmented" aria-label="Match filter">
        {(['all', 'inliers', 'high'] as const).map(value =>
          <button key={value} type="button" className={filter === value ? 'active' : ''}
            aria-pressed={filter === value}
            onClick={() => { setFilter(value); setFocusedMatch(null) }}>
            {value === 'high' ? 'High confidence' : value === 'inliers' ? 'Inliers' : 'All'}
          </button>)}
      </div>
    </div>
    {imageUrl ? <canvas ref={canvas} role="img" tabIndex={0}
      aria-label="Reference image with interactive correspondence points. Use left and right arrow keys to inspect matches."
      onClick={inspectPoint} onKeyDown={navigateMatches} />
      : <div className="artifact-unavailable">Reference preview is unavailable.</div>}
    <div className="confidence-legend"><span>LOW</span><i aria-hidden="true" /><span>HIGH</span><small>Inlier confidence</small><span className="outlier-key" /> <small>Outlier</small></div>
    {focusedMatch && <dl className="match-details" aria-live="polite">
      <div><dt>Match</dt><dd>#{focusedMatch.id}</dd></div>
      <div><dt>Source (px)</dt><dd>{focusedMatch.source_x.toFixed(2)}, {focusedMatch.source_y.toFixed(2)}</dd></div>
      <div><dt>Reference (px)</dt><dd>{focusedMatch.reference_x.toFixed(2)}, {focusedMatch.reference_y.toFixed(2)}</dd></div>
      <div><dt>Confidence</dt><dd>{confidenceText}</dd></div>
      <div><dt>Geometry</dt><dd>{focusedMatch.inlier ? 'RANSAC inlier' : 'Outlier'}</dd></div>
      <div><dt>Residual</dt><dd>{Number.isFinite(focusedMatch.error) ? `${focusedMatch.error.toFixed(3)} px` : '—'}</dd></div>
      <div><dt>Grid selection</dt><dd>{focusedMatch.selected ? 'Selected' : 'Not selected'}</dd></div>
    </dl>}
    {!focusedMatch && <p className="match-hint">Select a point, or focus the image and use the arrow keys, to inspect coordinates, confidence, and geometric residual.</p>}
  </div>
}
