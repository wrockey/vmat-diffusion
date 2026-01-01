import os
import json
import numpy as np
import pydicom
from skimage.draw import polygon
import warnings
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk

# Relax pydicom validation for Pinnacle DS VR precision issues
import pydicom.config
pydicom.config.settings.reading_validation_mode = pydicom.config.WARN

# Fixed AAPM constraints for prostate VMAT (QUANTEC/TG-101; normalized to [0,1])
AAPM_CONSTRAINTS = np.array([
    70.0,  # PTV70 mean (Gy)
    50.0,  # Rectum V50 (%)
    35.0,  # Rectum V60 (%)
    20.0,  # Rectum V70 (%)
    50.0,  # Bladder V65 (%)
    35.0,  # Bladder V70 (%)
    25.0,  # Bladder V75 (%)
    10.0,  # Femur V50 (%)
    195.0, # Bowel V45 (cc)
    45.0   # Spinal cord max (Gy)
]) / 100.0

def load_json_mapping(mapping_file='oar_mapping.json'):
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"{mapping_file} not found; run generate_mapping_json.py and edit")
    with open(mapping_file, 'r') as f:
        return json.load(f)

def get_ct_volume_and_metadata(plan_dir):
    ct_files = [f for f in os.listdir(plan_dir) if f.startswith('CT')]
    if not ct_files:
        raise ValueError(f"No CT files in {plan_dir}")
    ct_paths = [os.path.join(plan_dir, f) for f in ct_files]
    ct_ds = [pydicom.dcmread(p) for p in ct_paths]
    # Sort by increasing z-position
    sorted_indices = np.argsort([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    ct_ds = [ct_ds[i] for i in sorted_indices]
    slice_z = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_ds])
    print(f"CT z range: min {slice_z.min():.2f}, max {slice_z.max():.2f}, slices {len(slice_z)}, spacing {np.mean(np.diff(slice_z)):.2f}")
    ct_volume = np.stack([ds.pixel_array.astype(np.float32) for ds in ct_ds], axis=2)
    spacing = np.array([float(ct_ds[0].PixelSpacing[0]), float(ct_ds[0].PixelSpacing[1]), np.mean(np.diff(slice_z))])
    position = np.array([float(x) for x in ct_ds[0].ImagePositionPatient])
    return ct_volume, spacing, position, slice_z, ct_ds

def get_roi_numbers(rtstruct, variations):
    roi_nums = []
    for roi in rtstruct.StructureSetROISequence:
        name = roi.ROIName.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '')
        if name in [v.lower().replace('_', '').replace(' ', '').replace('gy', '').replace('-', '') for v in variations]:
            roi_nums.append(int(roi.ROINumber))
    return roi_nums

def create_mask_from_contours(rtstruct, roi_nums, ct_shape, position, spacing, slice_z, ct_ds):
    mask = np.zeros(ct_shape, dtype=np.uint8)
    for roi_num in roi_nums:
        try:
            roi_contour = next(c for c in rtstruct.ROIContourSequence if c.ReferencedROINumber == roi_num)
            for cont_seq in roi_contour.ContourSequence:
                points = np.array(cont_seq.ContourData).reshape(-1, 3)
                z = points[0, 2]
                slice_idx = np.argmin(np.abs(slice_z - z))
                if not 0 <= slice_idx < ct_shape[2]:
                    warnings.warn(f"Slice idx {slice_idx} out of bounds for z {z}; clamping")
                    slice_idx = np.clip(slice_idx, 0, ct_shape[2] - 1)
                slice_ds = ct_ds[slice_idx]
                slice_position = np.array([float(x) for x in slice_ds.ImagePositionPatient])
                slice_spacing = np.array([float(x) for x in slice_ds.PixelSpacing])
                pix_x = (points[:, 0] - slice_position[0]) / slice_spacing[0]
                pix_y = (points[:, 1] - slice_position[1]) / slice_spacing[1]
                rr, cc = polygon(pix_y, pix_x, (ct_shape[0], ct_shape[1]))
                rr = np.clip(rr, 0, ct_shape[0] - 1)
                cc = np.clip(cc, 0, ct_shape[1] - 1)
                mask[rr, cc, slice_idx] = 1
        except StopIteration:
            warnings.warn(f"No contours found for ROI {roi_num}")
    return mask

def resample_image(image, reference, interpolator=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)

def preprocess_dicom_rt(plan_dir, output_dir, structure_map, target_shape=(512, 512, 256), target_spacing=(1.0, 1.0, 2.0), use_gamma=False, relax_filter=False):
    try:
        ct_volume, ct_spacing, position, slice_z, ct_ds = get_ct_volume_and_metadata(plan_dir)
        
        rtstruct_file = [f for f in os.listdir(plan_dir) if f.startswith('RS')][0]
        rtstruct = pydicom.dcmread(os.path.join(plan_dir, rtstruct_file))
        
        masks = np.zeros((len(structure_map),) + ct_volume.shape, dtype=np.uint8)
        for ch_str, info in structure_map.items():
            ch = int(ch_str)
            roi_nums = get_roi_numbers(rtstruct, info['variations'])
            channel_mask = create_mask_from_contours(rtstruct, roi_nums, ct_volume.shape, position, ct_spacing, slice_z, ct_ds)
            masks[ch] = channel_mask
        
        ptv70_sum = masks[0].sum()
        ptv56_sum = masks[1].sum()
        print(f"PTV70 sum: {ptv70_sum}, PTV56 sum: {ptv56_sum}")
        if (ptv70_sum == 0 or ptv56_sum == 0) and not relax_filter:
            warnings.warn(f"Skipping {plan_dir}: Missing PTV70/PTV56")
            return None
        
        prostate_mask = masks[2]
        
        # Compute physical centroid
        nonzero = np.nonzero(prostate_mask)
        if len(nonzero[0]) == 0:
            warnings.warn("No prostate mask; centering on mid-volume")
            mid_y = prostate_mask.shape[0] / 2
            mid_x = prostate_mask.shape[1] / 2
            mid_z = prostate_mask.shape[2] / 2
            centroid_phys_x = position[0] + mid_x * ct_spacing[0]
            centroid_phys_y = position[1] + mid_y * ct_spacing[1]
            centroid_phys_z = position[2] + mid_z * ct_spacing[2]
        else:
            phys_x = position[0] + nonzero[1] * ct_spacing[0]
            phys_y = position[1] + nonzero[0] * ct_spacing[1]
            phys_z = slice_z[nonzero[2]]
            centroid_phys_x = np.mean(phys_x)
            centroid_phys_y = np.mean(phys_y)
            centroid_phys_z = np.mean(phys_z)
        
        # Create CT image
        ct_array_sitk = ct_volume.transpose(2, 0, 1)  # z y x
        ct_image = sitk.GetImageFromArray(ct_array_sitk)
        ct_image.SetSpacing((ct_spacing[0], ct_spacing[1], ct_spacing[2]))
        ct_image.SetOrigin((position[0], position[1], position[2]))
        
        # Load dose
        rtdose_file = [f for f in os.listdir(plan_dir) if f.startswith('RD')][0]
        dose_ds = pydicom.dcmread(os.path.join(plan_dir, rtdose_file))
        dose_scaling = float(dose_ds.DoseGridScaling)
        dose_array = dose_ds.pixel_array.astype(np.float32) * dose_scaling
        if dose_array.ndim != 3:
            raise ValueError("Dose not 3D")
        dose_array_sitk = dose_array  # (z, y, x)
        dose_position = [float(x) for x in dose_ds.ImagePositionPatient]
        dose_pixel_spacing = [float(x) for x in dose_ds.PixelSpacing]
        if 'GridFrameOffsetVector' in dose_ds:
            offsets = [float(o) for o in dose_ds.GridFrameOffsetVector]
            if len(offsets) != dose_array.shape[0]:
                raise ValueError("GridFrameOffsetVector length mismatch with dose frames")
            dose_z_positions = [dose_position[2] + offsets[i] for i in range(len(offsets))]
        else:
            warnings.warn("No GridFrameOffsetVector; assuming single frame or CT-like z")
            dose_z_positions = [dose_position[2]]
            if dose_array.shape[0] != 1:
                raise ValueError("Multi-frame dose without GridFrameOffsetVector")
        if len(dose_z_positions) > 1:
            diffs = np.diff(dose_z_positions)
            if not np.allclose(diffs, diffs[0], rtol=1e-3):
                warnings.warn("Non-uniform dose z-spacing; using mean")
            dose_z_spacing = np.mean(diffs)
        else:
            dose_z_spacing = ct_spacing[2]
        dose_spacing_sitk = (dose_pixel_spacing[0], dose_pixel_spacing[1], dose_z_spacing)
        dose_origin_sitk = (dose_position[0], dose_position[1], min(dose_z_positions))  # Use min for origin
        # Sort z-positions if not increasing
        if len(dose_z_positions) > 1 and dose_z_positions[1] < dose_z_positions[0]:
            warnings.warn("Decreasing dose z; sorting")
            sort_idx = np.argsort(dose_z_positions)
            dose_array_sitk = dose_array_sitk[sort_idx]
            dose_z_positions = np.array(dose_z_positions)[sort_idx]
            dose_z_spacing = np.abs(dose_z_spacing)  # Ensure positive
            dose_origin_sitk = (dose_position[0], dose_position[1], min(dose_z_positions))
        dose_image = sitk.GetImageFromArray(dose_array_sitk)
        dose_image.SetSpacing(dose_spacing_sitk)
        dose_image.SetOrigin(dose_origin_sitk)
        
        # Define reference image centered on prostate
        target_size_sitk = (target_shape[1], target_shape[0], target_shape[2])  # x y z
        target_spacing_sitk = (target_spacing[0], target_spacing[1], target_spacing[2])  # x y z
        target_origin_sitk = (
            centroid_phys_x - (target_size_sitk[0] * target_spacing_sitk[0]) / 2,
            centroid_phys_y - (target_size_sitk[1] * target_spacing_sitk[1]) / 2,
            centroid_phys_z - (target_size_sitk[2] * target_spacing_sitk[2]) / 2
        )
        reference_image = sitk.Image(target_size_sitk, sitk.sitkFloat32)
        reference_image.SetSpacing(target_spacing_sitk)
        reference_image.SetOrigin(target_origin_sitk)
        
        # Resample CT
        ct_resampled = resample_image(ct_image, reference_image, sitk.sitkLinear)
        ct_array = sitk.GetArrayFromImage(ct_resampled).transpose(1, 2, 0)  # y x z
        ct_volume = np.clip(ct_array, -1000, 3000) / 4000 + 0.5
        
        # Resample dose
        dose_resampled = resample_image(dose_image, reference_image, sitk.sitkLinear)
        dose_volume = sitk.GetArrayFromImage(dose_resampled).transpose(1, 2, 0) / 70.0
        
        # Resample masks
        masks_resampled = np.zeros((len(structure_map),) + target_shape, dtype=np.uint8)
        for ch_str, _ in structure_map.items():
            ch = int(ch_str)
            mask_original = masks[ch]  # y x z
            mask_array_sitk = mask_original.transpose(2, 0, 1)  # z y x
            mask_image = sitk.GetImageFromArray(mask_array_sitk.astype(np.float32))
            mask_image.SetSpacing(ct_image.GetSpacing())
            mask_image.SetOrigin(ct_image.GetOrigin())
            mask_res = resample_image(mask_image, reference_image, sitk.sitkNearestNeighbor)
            mask_array = sitk.GetArrayFromImage(mask_res).transpose(1, 2, 0)
            masks_resampled[ch] = (mask_array > 0.5).astype(np.uint8)
        
        print(f"Resampled PTV70 sum: {masks_resampled[0].sum()}, PTV56 sum: {masks_resampled[1].sum()}")
        
        ptv_type = np.array([1.0, 0.0, 0.0])
        constraints = np.concatenate([AAPM_CONSTRAINTS, ptv_type])
        
        os.makedirs(output_dir, exist_ok=True)
        plan_id = os.path.basename(plan_dir)
        output_path = os.path.join(output_dir, f'{plan_id}.npz')
        np.savez(output_path, ct=ct_volume, masks=masks_resampled, dose=dose_volume, constraints=constraints)
        
        stats = {'mask_sums': [int(masks_resampled[ch].sum()) for ch in range(len(structure_map))],
                 'dose_mean': float(dose_volume.mean())}
        print(f"Processed {plan_dir} to {output_path}: {stats}")
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        mid = target_shape[2] // 2
        axs[0].imshow(ct_volume[:, :, mid], cmap='gray')
        axs[1].imshow(dose_volume[:, :, mid], cmap='hot')
        axs[2].imshow(masks_resampled[0][:, :, mid], cmap='binary')
        plt.savefig(os.path.join(output_dir, f'debug_{plan_id}.png'))
        plt.close()
        
        return output_path
    except Exception as e:
        warnings.warn(f"Error processing {plan_dir}: {e}")
        return None

def batch_preprocess(input_base_dir, output_dir, mapping_file='oar_mapping.json', use_gamma=False, relax_filter=False):
    structure_map = load_json_mapping(mapping_file)
    plan_dirs = sorted([os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))])
    processed = []
    for plan_dir in plan_dirs:
        result = preprocess_dicom_rt(plan_dir, output_dir, structure_map, use_gamma=use_gamma, relax_filter=relax_filter)
        if result:
            processed.append(result)
    print(f"Batch complete: {len(processed)}/{len(plan_dirs)} cases processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM-RT to .npz with PTV70/PTV56 filter")
    parser.add_argument("--input_dir", default="~/vmat-diffusion-project/data/raw")
    parser.add_argument("--output_dir", default="~/vmat-diffusion-project/processed_npz")
    parser.add_argument("--mapping_file", default="oar_mapping.json")
    parser.add_argument("--use_gamma", action="store_true")
    parser.add_argument("--relax_filter", action="store_true")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    batch_preprocess(input_dir, output_dir, args.mapping_file, args.use_gamma, args.relax_filter)