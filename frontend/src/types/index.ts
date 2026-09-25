export type Sensor = 'OHRC' | 'TMC-2' | 'IIRS'
export type Matcher = 'sift' | 'orb' | 'akaze' | 'loftr' | 'hybrid'

export interface Match {
  id: number
  source_x: number
  source_y: number
  reference_x: number
  reference_y: number
  confidence: number
  inlier: boolean
  selected: boolean
  error: number
}

export interface RegistrationOptions {
  clahe: boolean
  local_normalization: boolean
  gradient: boolean
  shadow_mask: boolean
  pyramid_levels: number
  geometry_model: 'homography' | 'affine'
  spatial_selection: boolean
  subpixel_refinement: boolean
}

export interface JobSummary {
  job_id: string
  status: string
  matches: number
  inliers: number
  inlier_ratio: number
  rmse: number
  rmse_type: string
  coverage: number
  runtime_seconds: number
  device: string
}

export interface JobResult {
  job_id: string
  status: string
  metrics: Record<string, number | string | null | boolean | number[][]>
  artifacts: Record<string, string>
}
