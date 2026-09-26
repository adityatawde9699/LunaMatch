import { useState } from 'react'
import ResultsPanel from './components/ResultsPanel'
import { fetchCandidates, fetchResult, registerImages } from './services/api'
import type { JobResult, JobSummary, Match, Matcher, RegistrationOptions, Sensor } from './types'

const sensors: Sensor[] = ['OHRC', 'TMC-2', 'IIRS']
const matchers: { value: Matcher, label: string }[] = [
  { value: 'sift', label: 'SIFT baseline' }, { value: 'orb', label: 'ORB' },
  { value: 'akaze', label: 'AKAZE' }, { value: 'loftr', label: 'LoFTR' },
  { value: 'descriptor', label: 'LunaPatchDescriptor (trained)' },
  { value: 'lightglue', label: 'SuperPoint + LightGlue' },
  { value: 'hybrid', label: 'Hybrid: SIFT + LoFTR' }
]
const initialOptions: RegistrationOptions = {
  clahe: false, local_normalization: false, gradient: false, shadow_mask: false,
  pyramid_levels: 1, geometry_model: 'homography', spatial_selection: true,
  subpixel_refinement: false, source_band: 0, reference_band: 0, iirs_mode: 'pca',
  grid_rows: 8, grid_cols: 8, max_matches_per_cell: 10
}

const maxUploadBytes = 100 * 1024 * 1024

export default function App() {
  const [source, setSource] = useState<File | null>(null)
  const [reference, setReference] = useState<File | null>(null)
  const [sourceSensor, setSourceSensor] = useState<Sensor>('OHRC')
  const [referenceSensor, setReferenceSensor] = useState<Sensor>('OHRC')
  const [matcher, setMatcher] = useState<Matcher>('sift')
  const [options, setOptions] = useState<RegistrationOptions>(initialOptions)
  const [summary, setSummary] = useState<JobSummary | null>(null)
  const [result, setResult] = useState<JobResult | null>(null)
  const [completedConfig, setCompletedConfig] = useState<{
    sourceName: string, referenceName: string, sourceSensor: Sensor,
    referenceSensor: Sensor, matcher: Matcher, options: RegistrationOptions
  } | null>(null)
  const [matches, setMatches] = useState<Match[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function setOption<K extends keyof RegistrationOptions>(key: K, value: RegistrationOptions[K]) {
    setOptions(current => ({ ...current, [key]: value }))
  }

  function chooseImage(file: File | undefined, side: 'source' | 'reference') {
    if (!file) return
    if (file.size > maxUploadBytes) {
      if (side === 'source') setSource(null)
      else setReference(null)
      setError(`${file.name} is larger than the 100 MB upload limit.`)
      return
    }
    setError(null)
    if (side === 'source') setSource(file)
    else setReference(file)
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!source || !reference) return
    setRunning(true)
    setError(null)
    setSummary(null)
    setResult(null)
    setCompletedConfig(null)
    const submittedConfig = {
      sourceName: source.name, referenceName: reference.name,
      sourceSensor, referenceSensor, matcher, options: { ...options }
    }
    try {
      const job = await registerImages(source, reference, sourceSensor, referenceSensor, matcher, options)
      const [details, candidates] = await Promise.all([fetchResult(job.job_id), fetchCandidates(job.job_id)])
      setSummary(job)
      setResult(details)
      setCompletedConfig(submittedConfig)
      setMatches(candidates)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Registration failed')
    } finally {
      setRunning(false)
    }
  }

  return <div className="app-shell">
    <header className="site-header"><div className="brand"><span className="brand-mark" aria-hidden="true">◐</span><div><strong>LunaMatch</strong><small>PLANETARY IMAGE LAB</small></div></div><span className="header-tag">SIH26166 · RESEARCH PROTOTYPE</span></header>
    <main>
      <section className="hero"><div><p className="eyebrow">LUNAR IMAGE CORRESPONDENCE</p><h1>See the same terrain<br /><em>in a new light.</em></h1><p className="hero-copy">Register Chandrayaan-2 optical images across changes in illumination, scale, and sensor representation. AI proposes. Geometry verifies. Sub-pixel optimization refines.</p></div><div className="orbital" aria-hidden="true"><div className="moon" /></div></section>

      <form onSubmit={submit} className="workspace panel">
        <div className="section-head"><div><p className="eyebrow">01 / CONFIGURE</p><h2>New registration</h2></div><span className="step-status">Image pair → correspondences → geometry</span></div>
        <div className="upload-grid">
          <label className="upload-zone"><span className="upload-icon" aria-hidden="true">↥</span><strong>Source image</strong><span>{source?.name || 'Choose a lunar image'}</span><small>PNG, JPEG, TIFF, GeoTIFF · up to 100 MB</small><input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" required onChange={event => chooseImage(event.target.files?.[0], 'source')} /></label>
          <label className="upload-zone"><span className="upload-icon" aria-hidden="true">↥</span><strong>Reference image</strong><span>{reference?.name || 'Choose a reference image'}</span><small>PNG, JPEG, TIFF, GeoTIFF · up to 100 MB</small><input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" required onChange={event => chooseImage(event.target.files?.[0], 'reference')} /></label>
        </div>
        <div className="controls-grid">
          <label>Source sensor<select value={sourceSensor} onChange={event => setSourceSensor(event.target.value as Sensor)}>{sensors.map(sensor => <option key={sensor}>{sensor}</option>)}</select></label>
          <label>Reference sensor<select value={referenceSensor} onChange={event => setReferenceSensor(event.target.value as Sensor)}>{sensors.map(sensor => <option key={sensor}>{sensor}</option>)}</select></label>
          <label>Matcher<select value={matcher} onChange={event => setMatcher(event.target.value as Matcher)}>{matchers.map(item => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label>
          <label>Geometry<select value={options.geometry_model} onChange={event => setOption('geometry_model', event.target.value as 'affine' | 'homography')}><option value="homography">Homography</option><option value="affine">Affine</option></select></label>
          <label>Pyramid levels<input type="number" min="1" max="5" value={options.pyramid_levels} onChange={event => setOption('pyramid_levels', Number(event.target.value))} /></label>
        </div>
        <div className="option-grid">
          {([['clahe', 'Local contrast (CLAHE)'], ['local_normalization', 'Illumination normalization'],
             ['gradient', 'Gradient representation'], ['shadow_mask', 'Shadow heuristic'],
             ['spatial_selection', 'Uniform point selection'], ['subpixel_refinement', 'Sub-pixel refinement']] as const)
            .map(([key, label]) => <label className="check" key={key}><input type="checkbox" checked={options[key]} onChange={event => setOption(key, event.target.checked)} /><span>{label}</span></label>)}
        </div>
        <details className="advanced-settings">
          <summary>Advanced correspondence settings</summary>
          <div className="advanced-grid">
            <label>Grid rows<input type="number" min="1" max="32" value={options.grid_rows} onChange={event => setOption('grid_rows', Number(event.target.value))} /></label>
            <label>Grid columns<input type="number" min="1" max="32" value={options.grid_cols} onChange={event => setOption('grid_cols', Number(event.target.value))} /></label>
            <label>Maximum matches per cell<input type="number" min="1" max="100" value={options.max_matches_per_cell} onChange={event => setOption('max_matches_per_cell', Number(event.target.value))} /></label>
            {sourceSensor === 'IIRS' && <label>Source IIRS band (zero-based)<input type="number" min="0" value={options.source_band} onChange={event => setOption('source_band', Number(event.target.value))} disabled={options.iirs_mode === 'pca'} /></label>}
            {referenceSensor === 'IIRS' && <label>Reference IIRS band (zero-based)<input type="number" min="0" value={options.reference_band} onChange={event => setOption('reference_band', Number(event.target.value))} disabled={options.iirs_mode === 'pca'} /></label>}
            {(sourceSensor === 'IIRS' || referenceSensor === 'IIRS') && <label>IIRS spatial representation<select value={options.iirs_mode} onChange={event => setOption('iirs_mode', event.target.value as 'pca' | 'band')}><option value="pca">PCA plane</option><option value="band">Select band</option></select></label>}
          </div>
          <p>Grid selection ranks geometrically verified points per reference-image cell. IIRS PCA and band selection are spatial representations; neither claims spectral calibration.</p>
        </details>
        <div className="form-footer"><p>Input images are processed locally by the connected API. Synthetic examples are software tests, not lunar validation.</p><button type="submit" disabled={running || !source || !reference}>{running ? 'PROCESSING…' : 'REGISTER IMAGES  →'}</button></div>
        {running && <div className="pipeline-progress" role="status">Registration is running. The API processes the pipeline synchronously; results will appear here when complete.</div>}
        {error && <p className="error" role="alert">{error}</p>}
      </form>
      {summary && result && completedConfig && <ResultsPanel
        summary={summary} result={result} matches={matches} run={completedConfig} />}
    </main>
    <footer>Independent research prototype · No real Chandrayaan-2 accuracy claim · Ground-truth validation pending</footer>
  </div>
}
