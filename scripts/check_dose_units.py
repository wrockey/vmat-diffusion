import numpy as np

# Load prediction
pred_path = "predictions/structure_weighted_test/case_0007_pred.npz"
pred_data = np.load(pred_path)
pred_dose = pred_data['dose']

print(f"Prediction dose:")
print(f"  Shape: {pred_dose.shape}")
print(f"  Min: {pred_dose.min():.4f}")
print(f"  Max: {pred_dose.max():.4f}")
print(f"  Mean: {pred_dose.mean():.4f}")

# Load target
target_path = r"I:\processed_npz\case_0007.npz"
target_data = np.load(target_path)
target_dose = target_data['dose']

print(f"\nTarget dose (normalized):")
print(f"  Shape: {target_dose.shape}")
print(f"  Min: {target_dose.min():.4f}")
print(f"  Max: {target_dose.max():.4f}")
print(f"  Mean: {target_dose.mean():.4f}")

target_dose_gy = target_dose * 70.0
print(f"\nTarget dose (Gy = normalized * 70):")
print(f"  Min: {target_dose_gy.min():.4f}")
print(f"  Max: {target_dose_gy.max():.4f}")
print(f"  Mean: {target_dose_gy.mean():.4f}")

# Check PTV region
masks_sdf = target_data['masks_sdf']
ptv70_mask = masks_sdf[0] < 0

print(f"\nPTV70 region analysis:")
print(f"  N voxels: {np.sum(ptv70_mask)}")
print(f"  Pred in PTV: min={pred_dose[ptv70_mask].min():.2f}, max={pred_dose[ptv70_mask].max():.2f}, mean={pred_dose[ptv70_mask].mean():.2f}")
print(f"  Target in PTV (Gy): min={target_dose_gy[ptv70_mask].min():.2f}, max={target_dose_gy[ptv70_mask].max():.2f}, mean={target_dose_gy[ptv70_mask].mean():.2f}")
print(f"  Difference (pred - target): mean={np.mean(pred_dose[ptv70_mask] - target_dose_gy[ptv70_mask]):.2f} Gy")
