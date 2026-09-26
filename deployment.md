# Deploy LunaMatch on Render and Vercel

LunaMatch has two deployable parts:

| Part | Repository path | Host | Purpose |
| --- | --- | --- | --- |
| API | `api/`, `src/`, root `Dockerfile` | Render Docker web service | Image loading, registration, and result files |
| Dashboard | `frontend/` | Vercel Vite project | Browser upload and result display |

The dashboard calls the API directly. `VITE_API_BASE_URL` is compiled into the frontend build, and the API allows the dashboard origin through `LUNAMATCH_CORS_ORIGINS`. These steps describe a prototype deployment; no public LunaMatch instance is verified by this document.

## Before deploying

Configuration templates are [`.env.example`](.env.example) for the API and [`frontend/.env.example`](frontend/.env.example) for Vite. The API reads exported environment variables and does not load `.env` files automatically. For local development, copy the root template to `.env` and export it with `set -a; . ./.env; set +a` in a trusted shell; copy the frontend template to `frontend/.env.local`. Render and Vercel environment settings take the corresponding values for deployment. Keep actual `.env` files out of Git.

1. Push the code to a GitHub repository. Selectively stage code and documentation: raw lunar products, model checkpoints, and generated results must stay out of Git. This repository's `.dockerignore` also excludes `data/` and `results/` from the API image.
2. Confirm the baseline locally:

   ```bash
   python -m pytest -q
   cd frontend && npm install && npm run build
   ```

3. Decide whether the deployment is only a small SIFT demonstration or needs persistent result storage. The API image installs CPU-only PyTorch and planetary dependencies, making it comparatively large. Check the available memory of the selected [Render compute plan](https://render.com/docs/compute-plans) before using learned matchers or large rasters.

## 1. Deploy the API on Render

Create a **New → Web Service** connected to the GitHub repository. Use these settings:

| Render setting | Value |
| --- | --- |
| Branch | `master`, or the branch containing the code |
| Language/runtime | Docker |
| Root Directory | Repository root (`.`) |
| Dockerfile Path | `./Dockerfile` |
| Docker build context | Repository root (`.`) |
| Docker Command | Leave blank; use the Dockerfile `CMD` |
| Health Check Path | `/health` |

The Dockerfile starts Uvicorn on Render's `PORT`, so no custom start command or fixed port is needed. Render documents [Docker web service configuration](https://render.com/docs/docker) and [health checks](https://render.com/docs/web-services). The first build needs outbound access to install Python packages and the pinned LightGlue source dependency.

Set or review these environment variables under the service's **Environment** settings:

| Variable | Example | Purpose |
| --- | --- | --- |
| `LUNAMATCH_CORS_ORIGINS` | `https://your-site.vercel.app` | Exact allowed dashboard origin; set after Vercel assigns its URL |
| `LUNAMATCH_MAX_UPLOAD_BYTES` | `104857600` | Per-file upload cap; default is 100 MiB |
| `LUNAMATCH_MAX_CONCURRENT_JOBS` | `1` | Maximum simultaneous registrations per API process |
| `LUNAMATCH_RESULTS_DIR` | `/app/results/jobs` | Job artifact directory; this is ephemeral without a disk |
| `LUNAMATCH_MODEL_CACHE` | `/app/model-cache` | Cache for downloaded LoFTR/LightGlue weights |
| `LUNAMATCH_DESCRIPTOR_CHECKPOINT` | `/app/storage/models/lunar_patch_descriptor.pt` | Optional trained local descriptor checkpoint |
| `LUNAMATCH_DESCRIPTOR_DEVICE` | `cpu` | Optional descriptor inference device; this Dockerfile installs CPU PyTorch |

Do not set `LUNAMATCH_DESCRIPTOR_CHECKPOINT` unless that file actually exists inside the running container. The checkpoint and ISRO products are not included in the Git repository or Docker image. A checkpoint trained locally must be transferred to storage accessible to the service through an appropriate, authorized process.

After Render reports a successful deploy, open:

```text
https://YOUR-SERVICE.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```

Keep the Render service URL for the frontend configuration. A health response verifies that the API process started; it does not verify registration or model weights.

### Persisting jobs and model weights

Render's default filesystem is ephemeral. Files saved under `/app/results/jobs` and `/app/model-cache` can disappear after a restart or redeploy. Render's [free web service](https://render.com/docs/free) also spins down after inactivity and does not support persistent disks. For retained jobs or cached weights, use a paid service with a [persistent disk](https://render.com/docs/disks), for example mounted at `/app/storage`, and set:

```text
LUNAMATCH_RESULTS_DIR=/app/storage/jobs
LUNAMATCH_MODEL_CACHE=/app/storage/model-cache
LUNAMATCH_DESCRIPTOR_CHECKPOINT=/app/storage/models/lunar_patch_descriptor.pt
```

Only set the final variable after provisioning the checkpoint on the disk. A disk preserves files beneath its mount path; it does not add a job database, retention policy, or backup workflow to LunaMatch.

## 2. Deploy the dashboard on Vercel

Import the same GitHub repository as a Vercel project. In the project configuration choose:

| Vercel setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | Vercel default, or `npm install` |

Set the **Production** environment variable:

```text
VITE_API_BASE_URL=https://YOUR-SERVICE.onrender.com
```

Use the API origin only: omit `/api`, `/api/v1`, and a trailing slash. The frontend code appends API paths itself. Vite reads `VITE_` variables at build time, so redeploy the Vercel project after changing this value. Vercel documents [Vite environment variables](https://vercel.com/docs/frameworks/frontend/vite), [monorepo root directories](https://vercel.com/docs/monorepos), and [build settings](https://vercel.com/docs/builds/configure-a-build).

Open the deployed dashboard and copy its exact origin, such as `https://your-site.vercel.app`. Set that value as `LUNAMATCH_CORS_ORIGINS` on Render, then redeploy or restart the API. For several trusted origins, separate them with commas. The origin must match the browser's address bar scheme and host; do not add a path or trailing slash.

## 3. Check the deployed flow

1. Open the API `/health` URL.
2. Open the Vercel dashboard in a browser.
3. Upload a **small** overlapping PNG/JPEG/TIFF pair, select `OHRC` for both sensors and `SIFT`, then run registration. Use synthetic images only as a software smoke test; they are not Chandrayaan-2 validation.
4. Confirm that match points, metrics, registered image, overlay, and download links appear. A registration response includes a `job_id`; `/api/v1/results/{job_id}` lists artifacts that actually exist.

You can check the API without the dashboard from a local checkout containing the generated sample pair:

```bash
curl -fsS -X POST "https://YOUR-SERVICE.onrender.com/api/v1/register" \
  -F "source=@data/samples/synthetic_source.png" \
  -F "reference=@data/samples/synthetic_reference.png" \
  -F "source_sensor=OHRC" \
  -F "reference_sensor=OHRC" \
  -F "matcher=sift"
```

Create that local pair with `python experiments/generate_sample.py` if it is absent. Do not upload the generated files to GitHub merely to run this check. The API accepts PNG, JPEG, TIFF, and GeoTIFF uploads; PDS4 XML labels with external binary files must currently be processed through the local CLI.

## Operational limits and troubleshooting

| Symptom | Check |
| --- | --- |
| Render build fails | Confirm Docker runtime, repository-root context, root `Dockerfile`, and build logs. The image installs PyTorch and LightGlue and may need substantial build time and storage. |
| API health fails | Inspect Render logs and verify the service listens on Render's `PORT`; the Dockerfile already does this. |
| Dashboard cannot call API | Check `VITE_API_BASE_URL` in the Vercel production build, browser network errors, and the exact Vercel origin in Render's `LUNAMATCH_CORS_ORIGINS`. Rebuild after changing Vite variables. |
| HTTP 413 or 422 on upload | Check the per-file cap, supported format, whether the image decodes, and whether the pair overlaps enough to match. |
| Results disappear | Use a persistent disk and put `LUNAMATCH_RESULTS_DIR` below its mount path. |
| Learned matcher fails | Verify CPU memory, dependency/model download logs, and cache write access. Start with SIFT to isolate deployment from model availability. |
| Trained descriptor unavailable | Ensure the checkpoint exists in the container and `LUNAMATCH_DESCRIPTOR_CHECKPOINT` points to it. |

The current API returns registration results in the same request, keeps job artifacts on the local filesystem, and has no user accounts or job retention policy. Its random job IDs do not replace access control. Restrict exposure and avoid uploading sensitive or restricted imagery to an unrestricted public instance. Render's free plan is suitable for small demonstrations; its [resource and idle limits](https://render.com/docs/free) make large lunar products and learned inference uncertain. No deployed result should be presented as independently validated lunar accuracy without ground-truth tie points.
