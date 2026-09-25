import { useEffect, useRef, useState } from 'react'
import type { Match } from '../types'

type Filter = 'all' | 'inliers' | 'high'

export default function MatchCanvas({ imageUrl, matches }: { imageUrl: string, matches: Match[] }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [filter, setFilter] = useState<Filter>('inliers')

  useEffect(() => {
    const image = new Image()
    image.onload = () => {
      const surface = canvas.current
      const context = surface?.getContext('2d')
      if (!surface || !context) return
      surface.width = image.naturalWidth
      surface.height = image.naturalHeight
      context.drawImage(image, 0, 0)
      const visible = matches.filter(match =>
        filter === 'all' || (filter === 'inliers' && match.inlier) ||
        (filter === 'high' && match.inlier && match.confidence >= 0.7))
      for (const match of visible) {
        context.beginPath()
        context.arc(match.reference_x, match.reference_y, 3, 0, Math.PI * 2)
        context.fillStyle = match.inlier ? (match.selected ? '#7ce6ad' : '#e5ba69') : '#ee6b79'
        context.fill()
      }
    }
    image.src = imageUrl
  }, [imageUrl, matches, filter])

  return <div className="match-canvas">
    <div className="panel-heading"><h3>Correspondence confidence</h3>
      <div className="segmented" aria-label="Match filter">
        {(['all', 'inliers', 'high'] as const).map(value =>
          <button key={value} type="button" className={filter === value ? 'active' : ''}
            onClick={() => setFilter(value)}>{value === 'high' ? 'High confidence' : value === 'inliers' ? 'Inliers' : 'All'}</button>)}
      </div>
    </div>
    <canvas ref={canvas} aria-label="Reference image with match points" />
    <div className="legend"><span><i className="green" /> Selected</span><span><i className="amber" /> Other inlier</span><span><i className="red" /> Outlier</span></div>
  </div>
}
