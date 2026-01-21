"""Quick evaluation of test predictions."""
import numpy as np
from pathlib import Path

# Configuration
test_cases = ['case_0007', 'case_0021']
pred_dir = Path(r'C:\Users\Bill\vmat-diffusion-project\predictions\grad_loss_0.1_test')
data_dir = Path(r'I:\processed_npz')
rx_dose = 70.0

results = []
for case in test_cases:
    pred_data = np.load(pred_dir / f'{case}_pred.npz')
    gt_data = np.load(data_dir / f'{case}.npz')

    pred_dose = pred_data['dose']
    gt_dose = gt_data['dose']

    # Compute MAE in Gy (dose is normalized 0-1, Rx=70 Gy)
    mae_gy = np.mean(np.abs(pred_dose - gt_dose)) * rx_dose

    # Get body mask for masked metrics
    body_mask = gt_dose > 0.01  # > 1% of max
    mae_gy_body = np.mean(np.abs(pred_dose[body_mask] - gt_dose[body_mask])) * rx_dose

    print(f'{case}: MAE = {mae_gy:.2f} Gy (full), {mae_gy_body:.2f} Gy (body)')
    results.append({'case': case, 'mae_gy': mae_gy, 'mae_gy_body': mae_gy_body})

print(f'\n=== Summary ===')
print(f'Mean MAE: {np.mean([r["mae_gy"] for r in results]):.2f} +/- {np.std([r["mae_gy"] for r in results]):.2f} Gy')
print(f'Mean MAE (body): {np.mean([r["mae_gy_body"] for r in results]):.2f} +/- {np.std([r["mae_gy_body"] for r in results]):.2f} Gy')

# Try gamma computation
try:
    from pymedphys import gamma as calc_gamma
    print('\n=== Gamma Analysis (3%/3mm) ===')

    for case in test_cases:
        pred_data = np.load(pred_dir / f'{case}_pred.npz')
        gt_data = np.load(data_dir / f'{case}.npz')

        pred_dose = pred_data['dose'] * rx_dose
        gt_dose = gt_data['dose'] * rx_dose

        # Take central slices for faster computation
        z_mid = pred_dose.shape[0] // 2
        pred_slice = pred_dose[z_mid, :, :]
        gt_slice = gt_dose[z_mid, :, :]

        # Create coordinate axes (assuming 1mm spacing)
        y_coords = np.arange(pred_slice.shape[0])
        x_coords = np.arange(pred_slice.shape[1])

        # Compute gamma
        gamma = calc_gamma(
            (y_coords, x_coords), gt_slice,
            (y_coords, x_coords), pred_slice,
            dose_percent_threshold=3,
            distance_mm_threshold=3,
            lower_percent_dose_cutoff=10,
        )

        pass_rate = np.sum(gamma <= 1) / np.sum(~np.isnan(gamma)) * 100
        print(f'{case}: Gamma pass rate = {pass_rate:.1f}% (central slice)')

except Exception as e:
    print(f'\nGamma computation failed: {e}')
