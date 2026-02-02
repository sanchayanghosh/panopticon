# The Panopticon: Self-Healing Cognitive Supply Chain

This project implements a forensic cognitive architecture for LLMs/AI Agents, consisting of:
1.  **The Ledger**: A Merkle Tree blockchain for data/weight provenance with Hardware ID.
2.  **The Cortex Monitor**: Real-time 3D visualization of "thought trajectories" (attractor dynamics).
3.  **The Immune System**: An investigator agent that generates adversarial probes to map unsafe regions.

## 🚀 Deployment Instructions (Docker)

You have a Docker container `pytorch_cont`. These files should be moved there.

### 1. Automated Deployment
Run the included script to copy files and install dependencies:
```bash
./deploy.sh
```

### 1. Transfer Files
Copy the `panopticon` folder to your container:

```bash
docker cp /home/balgav/.gemini/antigravity/scratch/panopticon pytorch-cont:/home/torch/tensorvis
```

### 2. Enter Container
```bash
docker exec -it pytorch_cont bash
cd /home/torch/tensorvis
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Control Room
```bash
streamlit run app.py --server.port 9000 --server.address 0.0.0.0
```

## 🔍 Modules

- `ledger.py`: Handles cryptographic provenance, Hardware ID, and Merkle logging.
- `cortex.py`: Uses `transformer_lens` to extract activations and `sklearn` for PCA.
- `immune.py`: Generates adversarial probes to verify attractors.
- `app.py`: The dashboard that brings it all together.

## 🧪 Simulation
To verify the backend without the UI:
```bash
python verify_setup.py
```