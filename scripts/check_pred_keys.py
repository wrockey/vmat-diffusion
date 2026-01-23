import numpy as np
import os

pred_path = "predictions/structure_weighted_test/case_0007_pred.npz"
data = np.load(pred_path)
print("Prediction file keys:", list(data.keys()))
for key in data.keys():
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

# Also check ground truth
gt_path = r"I:\processed_npz\case_0007.npz"
if os.path.exists(gt_path):
    gt = np.load(gt_path)
    print("\nGround truth file keys:", list(gt.keys()))
