# SSH GPU Test Report (Remote ROCm Droplet)

## Context
- Test date: February 18, 2026 (UTC)
- Target host: DigitalOcean ROCm droplet (public IPv4 `129.212.190.231`)
- Access method: SSH (`root` user), intermittent connection-refused windows observed

## Remote Infrastructure Verification
- OS: Ubuntu 24.04.3 LTS
- GPU stack: ROCm driver `6.16.6`
- GPU device: `AMD Instinct MI300X VF`
- GPU visibility command (`rocm-smi`) succeeded

## Runtime Environment Probes
- Host Python: `3.12.3`
- Container Python: `3.12.3`
- Host module availability:
  - `torch`: not available
  - `transformers`: not available
  - `flask`: not available
  - `aiortc`: not available
  - `numpy`: not available
  - `requests`: available
- Container module availability:
  - `torch`: not available
  - `transformers`: not available
  - `numpy`: available
  - `requests`: available

## Service Availability Probes
- Remote LLM endpoint probe (`http://localhost:11434`): connection refused
- Remote voice-pipeline endpoint probe (`http://localhost:8080`):
  - Automated connectivity suite result: `0/6` checks passed
  - Health, readiness, signaling, ICE, and metrics endpoints all unreachable

## Artifact Generation on Remote Host
- Synchronized project to remote host under `/root/solvathon_layer1-main`
- Executed remote evidence extractor and generated:
  - `metrics_snapshot.json`
  - `evidence_summary.md`
- Pulled artifacts back to local workspace under:
  - `paper/evidence_remote/`

## Interpretation
Remote testing confirms that GPU infrastructure is provisioned and detectable, but the application stack was not fully operational at probe time due to missing runtime dependencies and unavailable local services. This supports a research conclusion of **infrastructure readiness without service readiness**.
