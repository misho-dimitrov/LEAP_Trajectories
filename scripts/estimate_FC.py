"""estimate_FC.py
==================
Estimate whole-brain functional connectivity (FC) from preprocessed rs-fMRI
and compute weighted degree centrality (wDC) per brain region.

Atlas
-----
Three atlases are combined into a single whole-brain parcellation:
  - Cortical:     Schaefer (200 parcels, Yeo 7-network labels)
  - Cerebellar:   FSL Cerebellum-MNIfnirt (probabilistic, 2 mm)
  - Subcortical:  Harvard-Oxford subcortical (6 bilateral structures + brainstem)

FC computation per subject:
  - Pairwise Pearson correlation across all parcels
  - Threshold at r >= --threshold (default 0.25); wDC = sum of retained edge weights
  - Subjects flagged exclude_stringent==1 in the LEAP QC CSVs are skipped

Inputs
------
- Preprocessed 4D NIfTI files under Data/<Timepoint>/<Site>/<SubjectID>/
- QC CSVs auto-resolved per timepoint via rs_paths.default_qc_csv()
- Cerebellar atlas: Atlas/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz
- Cerebellar labels: Atlas/Cerebellum_MNIfnirt.xml

Outputs
-------
- Per-subject wDC: reports/fc/<Timepoint>/<Site>/<SubjectID>/wdc.csv
- Master summary:  reports/fc/master_wdc.csv  (all subjects / timepoints)

Usage
-----
    python estimate_FC.py                            # all sites, T1-T3
    python estimate_FC.py --timepoints T1 --site KCL --n-jobs 8
    python estimate_FC.py --validate-only            # check atlas only
    python estimate_FC.py --help
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Union

try:
    from rs_paths import default_fc_output_dir, default_qc_csv, rs_project_root
except ImportError:  # pragma: no cover
    from scripts.rs_paths import default_fc_output_dir, default_qc_csv, rs_project_root  # type: ignore

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import load_img, new_img_like, resample_to_img
from concurrent.futures import ProcessPoolExecutor, as_completed


SITE_FOLDER_MAP = {
    # Accept centre names (CSV-style) and map to Data folder names
    "KINGS_COLLEGE": "KCL",
    "CAMBRIDGE": "Cambridge",
    "MANNHEIM": "Mannheim",
    "NIJMEGEN": "Nijmegen",
    "UTRECHT": "Utrecht",
    "ROME": "Rome",
    # Also accept canonical folder names directly
    "KCL": "KCL",
    # T3 uses "London_KCL" as both CSV site name and folder name
    "LONDON_KCL": "London_KCL",
}


def parse_args() -> argparse.Namespace:
    base_dir = rs_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whole-brain FC from preprocessed rs-fMRI using Schaefer cortical, "
            "FSL cerebellum, and Harvard-Oxford subcortical atlases. "
            "Threshold FC at r>=THRESHOLD and save matrices plus "
            "weighted degree centrality (wDC) per region. "
            "Assumes subjects are directly under Data/<SiteFolder>/<SubjectId>."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=base_dir / "Data",
        help="Root data folder containing site folders (Data/<SiteFolder>).",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Single site to process (e.g., KCL, CAMBRIDGE, MANNHEIM). If omitted, uses --sites.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=["KINGS_COLLEGE", "CAMBRIDGE", "MANNHEIM", "NIJMEGEN", "UTRECHT", "ROME", "LONDON_KCL"],
        help=(
            "Space-separated list of centres/sites to process. Accepts centre names (e.g., CAMBRIDGE) or folder codes "
            "(e.g., KCL). Missing folders are warned and skipped."
        ),
    )
    parser.add_argument(
        "--timepoints",
        "--timepoint",
        nargs="+",
        type=str,
        choices=["T1", "T2", "T3"],
        default=["T1", "T2", "T3"],
        help="Timepoint(s) to process (T1, T2, T3). Default: T1 T2 T3.",
    )
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help=(
            "QC CSV file path used to exclude subjects (exclude_stringent==1). "
            "If omitted, auto-resolves per timepoint (LEAP1 for T1, LEAP2 for T2, etc.)."
        ),
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help=(
            "Limit number of subjects to process (for testing). "
            "Set to 0 or negative to process all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_fc_output_dir(),
        help="Output directory to save matrices and wDC outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--atlas-n",
        type=int,
        default=200,
        help="Number of parcels for Schaefer atlas (e.g., 200).",
    )
    parser.add_argument(
        "--atlas-resolution",
        type=int,
        choices=[1, 2],
        default=2,
        help="Atlas resolution in mm (affects Schaefer + Harvard-Oxford defaults).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="FC threshold applied to the full symmetric correlation matrix (keep r>=threshold).",
    )
    parser.add_argument(
        "--cereb-atlas",
        type=Path,
        default=base_dir / "Atlas" / "Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz",
        help="Cerebellar atlas NIfTI path.",
    )
    parser.add_argument(
        "--cereb-xml",
        type=Path,
        default=base_dir / "Atlas" / "Cerebellum_MNIfnirt.xml",
        help="Cerebellar atlas XML labels path.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes to use (subjects processed in parallel).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the combined atlas and exit (don't process subjects).",
    )

    parser.add_argument(
        "--sub-exclude-labels",
        nargs="*",
        default=[
            "Left Cerebral White Matter",
            "Right Cerebral White Matter",
            "Left Cerebral Cortex",
            "Right Cerebral Cortex",
            "Left Lateral Ventricle",
            "Right Lateral Ventricle",
        ],
        help=(
            "Harvard-Oxford 'sub' labels to zero-out before combining with cortical/cerebellar atlases. "
            "This avoids cortex/white-matter/ventricle labels leaking into the whole-brain parcellation."
        ),
    )
    return parser.parse_args()


def resolve_site_folder(site_token: str) -> str:
    key = site_token.strip().upper()
    return SITE_FOLDER_MAP.get(key, site_token)


def get_site_subjects_root(data_root: Path, site_token: str, timepoint: str = "T1") -> Path:
    # Workspace data layout: Data/<Timepoint>/<SiteFolder>/<SubjectId>
    folder = resolve_site_folder(site_token)
    return data_root / timepoint / folder


def discover_subject_ids(preprocessed_tp_dir: Path) -> List[str]:
    ids: List[str] = []
    if not preprocessed_tp_dir.exists():
        return ids
    for entry in preprocessed_tp_dir.iterdir():
        if entry.is_dir() and re.fullmatch(r"\d{9,}", entry.name):
            ids.append(entry.name)
    return sorted(ids)


def discover_t3_subjects(data_root: Path, site_folder: str) -> List[Tuple[str, Path]]:
    """Discover all T3 subjects across batch folders for a given site.

    T3 layout: Data/T3/<batch>/<Site>/BOLD/Preprocessed/<SubjectId>
    Returns a list of (subject_id, subject_dir) tuples.
    """
    seen: Dict[str, Path] = {}
    t3_dir = data_root / "T3"
    if not t3_dir.exists():
        return []
    for batch_dir in sorted(t3_dir.iterdir(), key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else 0)):
        if not batch_dir.is_dir():
            continue
        preprocessed = batch_dir / site_folder / "BOLD" / "Preprocessed"
        if not preprocessed.exists():
            continue
        for entry in preprocessed.iterdir():
            if entry.is_dir() and re.fullmatch(r"\d{9,}", entry.name):
                if entry.name not in seen:
                    seen[entry.name] = entry
    return sorted(seen.items())


def round_id_to_6sig(sid: str) -> str:
    """Round a 12-digit subject ID to 6 significant figures (matches Excel scientific notation truncation)."""
    n = int(sid)
    return str(round(n, -(len(sid) - 6)))


def qc_excluded_subjects(qc_csv: Path) -> set:
    # Detect delimiter (LEAP1 is tab-separated, LEAP2 is comma-separated)
    with open(qc_csv, "r") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(qc_csv, sep=sep)
    # Normalise column names for lookup (LEAP1 uses 'subject', LEAP2 uses 'subjects')
    col_map = {c.strip().lower(): c for c in df.columns}
    subject_col = col_map.get("subject") or col_map.get("subjects")
    # T3 CSV uses 'exclude' instead of 'exclude_stringent'
    excl_col = col_map.get("exclude_stringent") or col_map.get("exclude")
    if not subject_col or not excl_col:
        raise KeyError(
            "QC CSV must contain 'subject'/'subjects' and 'exclude_stringent'/'exclude' columns "
            f"(found columns: {list(df.columns)})"
        )
    raw = df.loc[df[excl_col] == 1, subject_col].dropna()
    excluded = set()
    for v in raw:
        try:
            excluded.add(str(int(float(v))))
        except (ValueError, TypeError):
            excluded.add(str(v).strip())
    return excluded


def find_nifti_path(subject_dir: Path) -> Path:
    """Find the denoised NIfTI file, allowing for some path flexibility."""
    # Try T1/T2 filename first
    glob_pattern = "**/denoised_func_data_nonaggr_res_pp2MNI152.nii.gz"
    found_files = sorted(subject_dir.glob(glob_pattern))

    # Also try T3 filename
    if not found_files:
        glob_pattern_t3 = "**/filtered_func_data_standard.nii.gz"
        found_files = sorted(subject_dir.glob(glob_pattern_t3))

    if not found_files:
        raise FileNotFoundError(f"Denoised NIfTI file not found in {subject_dir}")
    if len(found_files) > 1:
        print(f"Warning: multiple matching NIfTI files found in {subject_dir}; using first: {found_files[0]}")
    
    return found_files[0]

def normalize_labels(labels: List[str]) -> List[str]:
    cleaned = [lab for lab in labels if lab and "background" not in str(lab).lower()]
    return cleaned


def labels_for_values(labels: List[str], values: List[int]) -> List[str]:
    if not labels:
        return [f"label_{v}" for v in values]
    out = []
    for v in values:
        idx = v - 1
        out.append(labels[idx] if 0 <= idx < len(labels) else f"label_{v}")
    return out


def parse_fsl_xml_labels(xml_path: Path, target_img_name: str) -> List[str]:
    if not xml_path.exists():
        raise RuntimeError(f"FSL atlas XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for atlas in root.findall("atlas"):
        images = [img.text.strip() for img in atlas.findall(".//imagefile") if img.text]
        if any(img.endswith(target_img_name) for img in images):
            labels = [lbl.text.strip() for lbl in atlas.findall(".//label") if lbl.text]
            return labels
    labels_with_index = []
    for label in root.iter("label"):
        if label.text and "index" in label.attrib:
            try:
                idx = int(label.attrib["index"])
            except ValueError:
                continue
            labels_with_index.append((idx, label.text.strip()))
    if labels_with_index:
        print(f"Warning: using fallback label parsing for {xml_path}. This may be incorrect if the XML contains multiple atlases.")
        labels_with_index.sort(key=lambda x: x[0])
        return [name for _, name in labels_with_index]
    raise RuntimeError(f"Could not find labels for {target_img_name} in {xml_path}")


def fetch_cerebellum_atlas() -> Tuple[Path, List[str]]:
    raise RuntimeError("Use fetch_cerebellum_atlas_from_paths with explicit atlas paths.")


def fetch_cerebellum_atlas_from_paths(atlas_path: Path, xml_path: Path) -> Tuple[Path, List[str]]:
    if not atlas_path.exists():
        raise RuntimeError(f"Cerebellum atlas not found at {atlas_path}")
    if not xml_path.exists():
        raise RuntimeError(f"Cerebellum atlas XML not found at {xml_path}")
    labels = parse_fsl_xml_labels(xml_path, atlas_path.name)
    return atlas_path, labels


def fetch_subcortical_atlas(resolution_mm: int) -> Tuple[Union[Path, object], List[str]]:
    atlas = datasets.fetch_atlas_harvard_oxford(f"sub-maxprob-thr25-{int(resolution_mm)}mm")
    return atlas["maps"], list(atlas["labels"])


def label_name_for_value(labels: List[str], value: int) -> str:
    if 0 <= int(value) < len(labels):
        return str(labels[int(value)]).strip()
    return f"label_{value}"


def prepare_label_atlas(atlas_img: Union[Path, object], atlas_labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    img = load_img(atlas_img)
    data = np.round(img.get_fdata()).astype(int)
    unique_vals = sorted(np.unique(data))
    unique_vals = [v for v in unique_vals if v != 0]
    cleaned_labels = normalize_labels(atlas_labels)
    if cleaned_labels and len(cleaned_labels) >= max(unique_vals, default=0):
        labels_by_value = [
            cleaned_labels[v - 1] if v - 1 < len(cleaned_labels) else f"label_{v}"
            for v in unique_vals
        ]
    else:
        labels_by_value = [f"label_{v}" for v in unique_vals]

    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
    relabeled = np.zeros_like(data, dtype=int)
    for old_val, new_val in mapping.items():
        relabeled[data == old_val] = new_val
    return relabeled, labels_by_value


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    # timeseries: shape (n_time, n_parcels)
    # Pearson correlation across parcels
    if timeseries.ndim != 2:
        raise ValueError("timeseries must be 2D (n_time, n_parcels)")
    ts = timeseries
    # np.corrcoef expects variables in rows, observations in columns when rowvar=True
    # Here we want parcels x parcels, so transpose
    r = np.corrcoef(ts.T)
    return r


def compute_weighted_degree(fc_r: np.ndarray) -> np.ndarray:
    fc = np.array(fc_r, dtype=float)
    np.fill_diagonal(fc, 0.0)
    fc = np.nan_to_num(fc, nan=0.0)
    return fc.sum(axis=0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_subject(
    subj_id: str,
    nifti_path: Path,
    atlas_img: Union[Path, object],
    atlas_labels: List[str],
    output_root: Path,
    overwrite: bool,
    expected_parcels: int,
    threshold: float,
) -> Dict[str, Path]:
    out_dir = output_root / subj_id
    ensure_dir(out_dir)
    outputs: Dict[str, Path] = {}

    # Skip if already done unless overwrite
    fc_r_path = out_dir / "fc_r.npy"
    upper_r_path = out_dir / "fc_upper_r.npy"
    wdc_path = out_dir / "wdc.npy"
    wdc_csv = out_dir / "wdc.csv"
    if (
        not overwrite
        and fc_r_path.exists()
        and upper_r_path.exists()
        and wdc_path.exists()
        and wdc_csv.exists()
    ):
        outputs.update({
            "fc_r": fc_r_path,
            "fc_upper_r": upper_r_path,
            "wdc": wdc_path,
            "wdc_csv": wdc_csv,
        })
        return outputs

    # Extract parcel time series
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize="zscore_sample",
        detrend=False,
        verbose=0,
    )
    img = load_img(nifti_path)
    ts = masker.fit_transform(img)  # shape (n_time, n_parcels)
    if ts is None or ts.size == 0:
        raise RuntimeError(f"Failed to extract time series for {subj_id} at {nifti_path}")

    # Validate that extracted parcels match expected atlas count
    n_time, n_parcels = ts.shape
    if n_parcels != expected_parcels:
        raise RuntimeError(
            f"Mismatch between atlas parcels ({expected_parcels}) and extracted time series ({n_parcels}) "
            f"for subject {subj_id} at {nifti_path}"
        )

    # Save time series
    ts_path = out_dir / "timeseries.npy"
    np.save(ts_path, ts)
    outputs["timeseries"] = ts_path

    # Compute FC (r) and apply thresholding
    fc_r = compute_fc(ts)
    fc_r = np.nan_to_num(fc_r, nan=0.0)
    np.fill_diagonal(fc_r, 0.0)
    # Ensure symmetry before thresholding (Option B: threshold full symmetric matrix)
    fc_r = (fc_r + fc_r.T) / 2.0
    fc_r[fc_r < float(threshold)] = 0.0

    # Save full thresholded matrices
    np.save(fc_r_path, fc_r)
    outputs["fc_r"] = fc_r_path

    # Upper triangle vector
    n = fc_r.shape[0]
    iu = np.triu_indices(n, k=1)
    upper_r = fc_r[iu]
    np.save(upper_r_path, upper_r)
    outputs["fc_upper_r"] = upper_r_path

    # Weighted degree centrality (after thresholding)
    wdc = compute_weighted_degree(fc_r)
    np.save(wdc_path, wdc)
    outputs["wdc"] = wdc_path

    labels_for_df = atlas_labels[: len(wdc)]
    wdc_df = pd.DataFrame({"label": labels_for_df, "wdc": wdc})
    wdc_df.to_csv(wdc_csv, index=False)
    outputs["wdc_csv"] = wdc_csv
    return outputs


def main() -> None:
    args = parse_args()
    # Determine sites to process
    sites: Iterable[str]
    if args.site:
        sites = [args.site]
    else:
        sites = args.sites

    # Fetch atlases and build combined whole-brain label image
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=args.atlas_n,
        yeo_networks=7,
        resolution_mm=args.atlas_resolution,
    )
    schaefer_img_path = Path(schaefer_atlas["maps"])
    schaefer_labels = [
        lab.decode("utf-8") if isinstance(lab, bytes) else str(lab)
        for lab in schaefer_atlas["labels"]
    ]

    cereb_img_path, cereb_labels = fetch_cerebellum_atlas_from_paths(
        args.cereb_atlas,
        args.cereb_xml,
    )
    sub_img_path, sub_labels = fetch_subcortical_atlas(args.atlas_resolution)

    schaefer_img = load_img(schaefer_img_path)
    schaefer_data, schaefer_labels_by_value = prepare_label_atlas(schaefer_img_path, schaefer_labels)
    cereb_img_res = resample_to_img(load_img(cereb_img_path), schaefer_img, interpolation="nearest")
    sub_img_res = resample_to_img(load_img(sub_img_path), schaefer_img, interpolation="nearest")

    cereb_data_res = np.round(cereb_img_res.get_fdata()).astype(int)
    sub_data_res = np.round(sub_img_res.get_fdata()).astype(int)

    # Harvard-Oxford "sub" atlas includes large non-subcortical labels (e.g., Cerebral Cortex/WM/Ventricles).
    # Exclude them up front so they cannot fill gaps in the cortical parcellation after resampling.
    if getattr(args, "sub_exclude_labels", None):
        exclude_norm = {str(x).strip().lower() for x in args.sub_exclude_labels}
        excluded_vals = [
            idx
            for idx, name in enumerate(sub_labels)
            if str(name).strip().lower() in exclude_norm
        ]
        if excluded_vals:
            excluded_names = [label_name_for_value(sub_labels, v) for v in excluded_vals]
            print(
                "Info: excluding Harvard-Oxford labels from subcortical atlas: "
                + ", ".join(excluded_names)
            )
            for v in excluded_vals:
                sub_data_res[sub_data_res == int(v)] = 0

    # Relabel cerebellum and subcortical to contiguous indices
    cereb_unique = [v for v in sorted(np.unique(cereb_data_res)) if v != 0]
    cereb_map = {v: i + 1 for i, v in enumerate(cereb_unique)}
    cereb_data_rel = np.zeros_like(cereb_data_res, dtype=int)
    for old_val, new_val in cereb_map.items():
        cereb_data_rel[cereb_data_res == old_val] = new_val

    sub_unique = [v for v in sorted(np.unique(sub_data_res)) if v != 0]
    sub_map = {v: i + 1 for i, v in enumerate(sub_unique)}
    sub_data_rel = np.zeros_like(sub_data_res, dtype=int)
    for old_val, new_val in sub_map.items():
        sub_data_rel[sub_data_res == old_val] = new_val

    if not cereb_unique:
        print("Warning: cerebellum atlas resampled to Schaefer has no nonzero labels.")
    if not sub_unique:
        print("Warning: subcortical atlas resampled to Schaefer has no nonzero labels.")

    cereb_total_vox = {v: int(np.sum(cereb_data_rel == cereb_map[v])) for v in cereb_unique}
    sub_total_vox = {v: int(np.sum(sub_data_rel == sub_map[v])) for v in sub_unique}

    combined = schaefer_data.copy()
    offset = combined.max()
    cereb_overlap = (combined > 0) & (cereb_data_rel > 0)
    if np.any(cereb_overlap):
        print("Warning: cerebellum atlas overlaps with cortical atlas; keeping cortical labels.")
    cereb_mask = (cereb_data_rel > 0) & (combined == 0)

    cereb_kept_vox = {
        v: int(np.sum(cereb_mask & (cereb_data_rel == cereb_map[v])))
        for v in cereb_unique
    }
    cereb_labels_normalized = normalize_labels(cereb_labels)
    for v in cereb_unique:
        total = cereb_total_vox.get(v, 0)
        kept = cereb_kept_vox.get(v, 0)
        # cereb_labels don't include Background; atlas values are 1-based
        name = cereb_labels_normalized[v - 1] if 0 <= v - 1 < len(cereb_labels_normalized) else f"label_{v}"
        if total > 0 and kept == 0:
            print(f"Warning: cerebellum region '{v}' ({name}) lost all voxels due to overlap.")
        if total > 0 and (total - kept) / total >= 0.5:
            print(f"Warning: cerebellum region '{v}' ({name}) lost >=50% voxels due to overlap ({kept}/{total} kept).")

    combined[cereb_mask] = cereb_data_rel[cereb_mask] + offset
    offset = combined.max()

    sub_overlap = (combined > 0) & (sub_data_rel > 0)
    if np.any(sub_overlap):
        print("Warning: subcortical atlas overlaps with existing labels; keeping existing labels.")
    sub_mask = (sub_data_rel > 0) & (combined == 0)

    sub_kept_vox = {
        v: int(np.sum(sub_mask & (sub_data_rel == sub_map[v])))
        for v in sub_unique
    }
    for v in sub_unique:
        total = sub_total_vox.get(v, 0)
        kept = sub_kept_vox.get(v, 0)
        if total > 0 and kept == 0:
            name = label_name_for_value(sub_labels, v)
            print(f"Warning: subcortical label '{v}' ({name}) lost all voxels due to overlap.")
        if total > 0 and (total - kept) / total >= 0.5:
            name = label_name_for_value(sub_labels, v)
            print(
                f"Warning: subcortical label '{v}' ({name}) lost >=50% voxels due to overlap "
                f"({kept}/{total} kept)."
            )

    combined[sub_mask] = sub_data_rel[sub_mask] + offset

    schaefer_labels_clean = schaefer_labels_by_value
    cereb_labels_clean = labels_for_values(normalize_labels(cereb_labels), cereb_unique)
    sub_labels_clean = labels_for_values(normalize_labels(sub_labels), sub_unique)
    combined_labels = schaefer_labels_clean + cereb_labels_clean + sub_labels_clean
    expected_parcels = len(combined_labels)
    
    # Validate that combined_labels length matches the actual number of unique values in combined
    unique_combined = len(np.unique(combined)) - 1  # exclude 0
    assert unique_combined == expected_parcels, f"Mismatch: {unique_combined} vs {expected_parcels}"
    
    combined_img = new_img_like(schaefer_img, combined)

    if args.validate_only:
        print(f"✓ Validation passed: {unique_combined} unique parcels match {expected_parcels} expected parcels")
        print("Exiting (--validate-only flag set)")
        return

    master_wdc_rows: List[Dict[str, object]] = []

    for tp in args.timepoints:
        tp_num = int(tp[1:])  # T1 -> 1, T2 -> 2, T3 -> 3
        qc_csv = args.qc_csv or default_qc_csv(timepoint=tp_num)
        print(f"\n{'=' * 60}")
        print(f"Processing timepoint {tp} (QC: {qc_csv})")
        print(f"{'=' * 60}")
        excluded = qc_excluded_subjects(qc_csv)
        tp_output_dir = args.output_dir / tp

        # For T3, build a set of rounded excluded IDs for matching
        excluded_rounded = set()
        if tp == "T3":
            for eid in excluded:
                try:
                    excluded_rounded.add(round_id_to_6sig(str(eid)))
                except (ValueError, IndexError):
                    excluded_rounded.add(str(eid))

        for site_token in sites:
            site_folder = resolve_site_folder(site_token)

            if tp == "T3":
                # T3: subjects are spread across batch folders
                t3_subjects = discover_t3_subjects(args.data_root, site_folder)
                if not t3_subjects:
                    print(f"Warning: no T3 subject folders found for {site_folder}; skipping")
                    continue

                # QC filtering via rounded ID matching
                eligible_pairs = []
                for sid, subj_dir in t3_subjects:
                    rounded = round_id_to_6sig(sid)
                    if rounded not in excluded_rounded:
                        eligible_pairs.append((sid, subj_dir))
                if args.max_subjects and args.max_subjects > 0:
                    eligible_pairs = eligible_pairs[: args.max_subjects]
                if not eligible_pairs:
                    print(f"No eligible subjects to process after QC filter for {site_folder}.")
                    continue

                output_root = tp_output_dir / site_folder
                ensure_dir(output_root)

                tasks = []
                for sid, subj_dir in eligible_pairs:
                    try:
                        nifti_path = find_nifti_path(subj_dir)
                    except FileNotFoundError as e:
                        print(f"Skipping {sid} ({site_folder}): {e}")
                        continue
                    tasks.append((sid, nifti_path))
            else:
                # T1/T2: standard layout
                site_root = get_site_subjects_root(args.data_root, site_token, tp)
                if not site_root.exists():
                    print(f"Warning: site folder missing: {site_root} (skipping)")
                    continue

                subject_ids = discover_subject_ids(site_root)
                if not subject_ids:
                    print(f"Warning: no subject folders found under {site_root}; skipping")
                    continue

                # QC filtering and optional cap
                eligible = [sid for sid in subject_ids if sid not in excluded]
                if args.max_subjects and args.max_subjects > 0:
                    eligible = eligible[: args.max_subjects]
                if not eligible:
                    print(f"No eligible subjects to process after QC filter for {site_folder}.")
                    continue

                # Outputs per site
                output_root = tp_output_dir / site_folder
                ensure_dir(output_root)

                # Build task list
                tasks = []
                for sid in eligible:
                    subj_dir = site_root / sid
                    try:
                        nifti_path = find_nifti_path(subj_dir)
                    except FileNotFoundError as e:
                        print(f"Skipping {sid} ({site_folder}): {e}")
                        continue
                    tasks.append((sid, nifti_path))

            subject_count = 0
            succeeded_sids = set()
            if args.n_jobs and args.n_jobs > 1:
                with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                    future_to_sid = {
                        ex.submit(
                            process_subject,
                            sid,
                            nifti_path,
                            combined_img,
                            combined_labels,
                            output_root,
                            args.overwrite,
                            expected_parcels,
                            args.threshold,
                        ): sid
                        for sid, nifti_path in tasks
                    }
                    for fut in as_completed(future_to_sid):
                        sid = future_to_sid[fut]
                        try:
                            fut.result()
                            subject_count += 1
                            succeeded_sids.add(sid)
                        except Exception as e:
                            print(f"Subject {sid} failed ({site_folder}): {e}", file=sys.stderr)
                            continue
            else:
                for sid, nifti_path in tasks:
                    try:
                        process_subject(
                            sid,
                            nifti_path,
                            combined_img,
                            combined_labels,
                            output_root,
                            overwrite=args.overwrite,
                            expected_parcels=expected_parcels,
                            threshold=args.threshold,
                        )
                        subject_count += 1
                        succeeded_sids.add(sid)
                    except Exception as e:
                        print(f"Subject {sid} failed ({site_folder}): {e}", file=sys.stderr)
                        continue

            print(f"Completed {subject_count} subjects for {site_folder} ({tp}).")

            # Collect wDC summaries only from subjects that succeeded
            for sid in succeeded_sids:
                wdc_csv_path = output_root / sid / "wdc.csv"
                if wdc_csv_path.exists():
                    wdc_df = pd.read_csv(wdc_csv_path)
                    wdc_df.insert(0, "subject", sid)
                    wdc_df.insert(1, "site", site_folder)
                    wdc_df.insert(2, "timepoint", tp)
                    master_wdc_rows.extend(wdc_df.to_dict(orient="records"))

    if master_wdc_rows:
        master_wdc_df = pd.DataFrame(master_wdc_rows)
        master_wdc_df["subject"] = master_wdc_df["subject"].astype(str)
        master_wdc_csv = args.output_dir / "master_wdc.csv"
        ensure_dir(args.output_dir)
        master_wdc_df.to_csv(master_wdc_csv, index=False)
        print(f"Saved master wDC summary: {master_wdc_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
