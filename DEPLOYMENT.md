# DigitalOcean GPU Deployment (MI300X)

## MI300X Plan Specs
- 1x MI300X GPU (192 GB VRAM)
- 20 vCPU, 240 GB RAM
- Boot disk: 720 GB NVMe
- Scratch disk: 5 TB NVMe (non-persistent, not snapshotted)

## 1) Create Droplet
- Use the **GPU Droplet** flow and pick an **MI300X** plan.
- Choose the **AI/ML-ready AMD** image so ROCm is preinstalled.

## 2) Bootstrap Host (Scratch + Docker)
Run the bootstrap script to mount the scratch disk and install Docker:
```bash
sudo bash deploy/do_gpu_bootstrap.sh
```
This script:
- Mounts the scratch disk at `/mnt/scratch` using the `DOSCRATCH` label.
- Creates cache dirs at `/mnt/scratch/hf` and `/mnt/scratch/torch`.
- Installs Docker + Compose plugin.

## 3) Clone and Configure
```bash
git clone <repo-url>
cd solvathon_layer1
cp .env.example .env
```
Edit `.env`:
- **Edge TTS** (Tamil/Kannada): set voices/rate/volume/pitch as needed.
- **Caches** (recommended):
  - `HF_CACHE_DIR=/mnt/scratch/hf`
  - `TORCH_CACHE_DIR=/mnt/scratch/torch`

## 4) Start Services
```bash
docker compose up --build -d
```

## 5) (Optional) Run via systemd
1) Copy the unit file and update the repo path inside:
```bash
sudo cp deploy/solvathon.service /etc/systemd/system/solvathon.service
sudo nano /etc/systemd/system/solvathon.service
```
2) Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now solvathon
```

## 6) Verify
```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
```

## Notes
- **Do not store models on the scratch disk**. It is non‑persistent.
- Edge TTS requires **outbound internet** access.
- For TLS, update `deploy/nginx.conf` with your domain and cert paths, then restart `nginx`.
