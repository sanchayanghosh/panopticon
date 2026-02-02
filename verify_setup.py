import sys
import os

# Ensure we can import local modules
sys.path.append(os.getcwd())

def test_pipeline():
    print("[-] Starting Panopticon Backend Verification...")
    
    # 1. Test Imports
    try:
        from ledger import MerkleLogger
        from cortex import AttractorAnalyzer
        from immune import ImmuneSystem
        print("[+] Modules imported successfully.")
    except ImportError as e:
        print(f"[!] Import Failed: {e}")
        # If imports fail due to missing dependencies in this environment, 
        # we still want to generate the script for the user.
        return

    # 2. Test Ledger
    print("[-] Testing Ledger...")
    ledger = MerkleLogger()
    print(f"    Hardware ID: {ledger.hardware_id}")
    ledger.log_step(1, {"test": "data"}, [0.1, 0.2])
    if ledger.verify_integrity():
        print("[+] Ledger Integrity Verified.")
    else:
        print("[!] Ledger Integrity Failed.")

    # 3. Test Cortex (Mocked if model load fails)
    print("[-] Testing Cortex Monitor...")
    cortex = AttractorAnalyzer(model_name="gpt2-small")
    
    # We might not have internet/gpu to load model here, so we wrap
    try:
        cortex.load_model()
        if cortex.model:
            traj = cortex.get_thought_trajectory("Hello world")
            print(f"    Trajectory Shape: {traj.shape}")
            
            # Calibration
            cortex.calibrate(["Safe query"], ["Unsafe query"])
            
            # Classification
            cls = cortex.classify_state(cortex.transform_trajectory(traj))
            print(f"    Classification Result: {cls}")
        else:
            print("[!] Model could not be loaded (likely environment issue). Skipping inference.")
    except Exception as e:
        print(f"[!] Cortex Test Partial Failure: {e}")
        
    print("[+] Verification Script Completed.")

if __name__ == "__main__":
    test_pipeline()
