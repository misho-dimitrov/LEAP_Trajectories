#!/usr/bin/env python3
"""check_subjects.py
====================
Pre-analysis QC: verify that LEAP subjects listed in the QC spreadsheets exist
in the preprocessed-data directory tree, check NIfTI header integrity, and
identify which subjects to exclude based on QC flag columns.

Inputs
------
- LEAP QC CSVs in the project root:
    T1: LEAP1_RS-fMRI_QC_final.tsv
    T2: LEAP2_RS-fMRI_QC_final.csv
    T3: LEAP3_QC_raport_all_sites.csv
- Preprocessed subject folders under Data/<Timepoint>/<Site>/<SubjectID>/.

Outputs (written to reports/)
------------------------------
- <site>_<tp>_subjects_presence.csv        -- present/missing per subject
- <site>_<tp>_nifti_qc.csv                 -- header/volume QC per subject
- <site>_<tp>_remaining_missing_after_exclusions.csv -- subjects absent from
  disk that are not flagged for exclusion in the QC CSV (unexpected gaps)

Usage
-----
    python check_subjects.py                   # all sites, all timepoints
    python check_subjects.py --timepoints T1   # T1 only
    python check_subjects.py --help
"""

import argparse
import csv
import os
from typing import Any, Dict, List, Set, Tuple

try:
    import nibabel as nib
except Exception:
    nib = None


DEFAULT_CSV = "LEAP1_RS-fMRI_QC_final.csv"
DEFAULT_DATA_BASE = "Data"

# Map CSV centre values to folder names under Data/
CENTRE_TO_FOLDER = {
    "KINGS_COLLEGE": "KCL",
    "CAMBRIDGE": "Cambridge",
    "MANNHEIM": "Mannheim",
    "NIJMEGEN": "Nijmegen",
    "UTRECHT": "Utrecht",
    "ROME": "Rome",
    # T3 uses "London_KCL" as both the CSV site name and the folder name
    "LONDON_KCL": "London_KCL",
}

# Reverse mapping for T2 CSV which uses folder names
FOLDER_TO_CENTRE = {v: k for k, v in CENTRE_TO_FOLDER.items()}


def detect_delimiter(path: str) -> str:
    with open(path, "r", newline="") as f:
        head = f.readline()
    return "\t" if "\t" in head else ","


def read_subjects_by_centre(csv_path: str, centres: List[str]) -> Dict[str, List[str]]:
    delimiter = detect_delimiter(csv_path)
    subjects_by: Dict[str, List[str]] = {c: [] for c in centres}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        field_map = {k.strip().lower(): k for k in (reader.fieldnames or [])}
        # Try different column name variations (LEAP1 vs LEAP2 vs LEAP3)
        centre_key = field_map.get("centre") or field_map.get("t2_site") or field_map.get("site")
        subject_key = field_map.get("subject") or field_map.get("subjects")
        if not centre_key or not subject_key:
            raise KeyError("CSV must contain 'centre'/'t2_site' and 'subject'/'subjects' columns (case-insensitive).")
        for row in reader:
            centre_val = (row.get(centre_key) or "").strip()
            subject_val = (row.get(subject_key) or "").strip()
            if not subject_val:
                continue
            # Normalize centre name (T2 CSV uses folder names, T1 uses full names)
            # Try to map folder name to centre name if needed
            normalized_centre = FOLDER_TO_CENTRE.get(centre_val, centre_val.upper())
            if normalized_centre in centres and subject_val not in subjects_by[normalized_centre]:
                subjects_by[normalized_centre].append(subject_val)
    return subjects_by


def list_subject_dirs(site_root: str) -> Set[str]:
    names: Set[str] = set()
    if not os.path.isdir(site_root):
        return names
    for name in os.listdir(site_root):
        full = os.path.join(site_root, name)
        if os.path.isdir(full):
            names.add(name.strip())
    return names


def check_presence(subjects: List[str], folder_names: Set[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for sid in subjects:
        if sid in folder_names:
            present.append(sid)
        else:
            missing.append(sid)
    return present, missing


def read_exclude_subjects(
    csv_path: str,
    column_name: str,
    centre_filter: str | None,
) -> List[str]:
    subjects: List[str] = []
    delimiter = detect_delimiter(csv_path)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        field_map = {k.strip().lower(): k for k in (reader.fieldnames or [])}
        # Try different column name variations (LEAP1 vs LEAP2 vs LEAP3)
        centre_key = field_map.get("centre") or field_map.get("t2_site") or field_map.get("site")
        subject_key = field_map.get("subject") or field_map.get("subjects")
        excl_key = field_map.get(column_name.strip().lower())
        if not subject_key or not excl_key:
            raise KeyError(
                f"CSV must contain 'subject'/'subjects' and '{column_name}' columns (case-insensitive)."
            )
        for row in reader:
            centre_val = (row.get(centre_key) or "").strip() if centre_key else ""
            subject_val = (row.get(subject_key) or "").strip()
            excl_val = (row.get(excl_key) or "").strip()
            # Normalize centre name for comparison
            normalized_centre = FOLDER_TO_CENTRE.get(centre_val, centre_val.upper())
            if centre_filter and normalized_centre != centre_filter.upper():
                continue
            if subject_val and excl_val == "1":
                subjects.append(subject_val)
    return subjects


def build_expected_volumes(csv_path: str, centre_filter: str | None) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    delimiter = detect_delimiter(csv_path)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        field_map = {k.strip().lower(): k for k in (reader.fieldnames or [])}
        # Try different column name variations (LEAP1 vs LEAP2 vs LEAP3)
        centre_key = field_map.get("centre") or field_map.get("t2_site") or field_map.get("site")
        subject_key = field_map.get("subject") or field_map.get("subjects")
        volumes_key = field_map.get("volumes")
        if not subject_key or not volumes_key:
            raise KeyError("CSV must contain 'subject'/'subjects' and 'volumes' columns.")
        for row in reader:
            centre_val = (row.get(centre_key) or "").strip() if centre_key else ""
            subject_val = (row.get(subject_key) or "").strip()
            volumes_str = (row.get(volumes_key) or "").strip()
            # Normalize centre name for comparison
            normalized_centre = FOLDER_TO_CENTRE.get(centre_val, centre_val.upper())
            if centre_filter and normalized_centre != centre_filter.upper():
                continue
            if subject_val and volumes_str:
                try:
                    mapping[subject_val] = int(volumes_str)
                except ValueError:
                    pass
    return mapping


def subject_nifti_path(subject_id: str, site_root: str, timepoint: str = "T1") -> str:
    """Get the path to the NIfTI file for a subject.
    
    Args:
        subject_id: Subject identifier
        site_root: Path to the site folder (e.g., Data/T1/Cambridge)
        timepoint: Either 'T1', 'T2', or 'T3' to determine the correct subfolder structure
    """
    if timepoint == "T1":
        # T1 path: subject_id/model005/task006.pre/...
        return os.path.join(
            site_root,
            subject_id,
            "model005",
            "task006.pre",
            "RS_FMRI_4D.ica_aroma",
            "nr_wm_csf",
            "highpassfilter",
            "denoised_func_data_nonaggr_res_pp2MNI152.nii.gz",
        )
    elif timepoint == "T3":
        # T3 path: subject_id/Resting_state/ICA_AROMA/.../filtered_func_data_standard.nii.gz
        # site_root already points to the BOLD/Preprocessed dir for T3
        return os.path.join(
            site_root,
            subject_id,
            "Resting_state",
            "ICA_AROMA",
            "denoised_func_data_nonaggr.feat",
            "filtered_func_data_standard.nii.gz",
        )
    else:  # T2
        # T2 path: subject_id/Resting_state/...
        return os.path.join(
            site_root,
            subject_id,
            "Resting_state",
            "RS_FMRI_4D.ica_aroma",
            "nr_wm_csf",
            "highpassfilter",
            "denoised_func_data_nonaggr_res_pp2MNI152.nii.gz",
        )


def read_nifti_info(nifti_path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": os.path.exists(nifti_path),
        "path": nifti_path,
        "file_size_bytes": None,
        "voxel_size_mm": None,
        "shape": None,
        "volumes": None,
        "error": None,
    }
    if not info["exists"]:
        return info
    try:
        info["file_size_bytes"] = os.path.getsize(nifti_path)
        if nib is None:
            info["error"] = "nibabel_not_installed"
            return info
        img = nib.load(nifti_path)
        header = img.header
        pixdim = header.get_zooms()
        voxel_size = tuple(round(float(v), 3) for v in pixdim[:3])
        info["voxel_size_mm"] = voxel_size
        shape = img.shape
        info["shape"] = shape
        if len(shape) >= 4:
            info["volumes"] = int(shape[3])
        elif len(shape) == 3:
            info["volumes"] = 1
        else:
            info["volumes"] = None
    except Exception as e:
        info["error"] = f"load_error: {e}"
    return info


def format_mb(bytes_val: int | None) -> str:
    if bytes_val is None:
        return "-"
    return f"{bytes_val / (1024*1024):.1f}MB"


def round_id_to_6sig(sid: str) -> str:
    """Round a 12-digit subject ID to 6 significant figures (matches Excel scientific notation truncation)."""
    n = int(sid)
    return str(round(n, -(len(sid) - 6)))


def build_t3_id_lookup(data_base: str, folder: str) -> Dict[str, str]:
    """Build a mapping from rounded (CSV) subject IDs to real folder IDs for T3.

    T3 layout: Data/T3/<batch>/<Site>/BOLD/Preprocessed/<SubjectId>
    The QC CSV has IDs truncated to 6 significant figures by Excel, so we match by rounding.
    """
    import glob
    pattern = os.path.join(data_base, "T3", "*", folder, "BOLD", "Preprocessed", "*")
    real_ids: Set[str] = set()
    for p in glob.iglob(pattern):
        if os.path.isdir(p):
            real_ids.add(os.path.basename(p))
    lookup: Dict[str, str] = {}
    for rid in real_ids:
        rounded = round_id_to_6sig(rid)
        lookup[rounded] = rid
    return lookup


def list_subject_dirs_t3(data_base: str, folder: str) -> Set[str]:
    """List all subject directory names across all T3 batch folders for a given site.

    T3 layout: Data/T3/<batch>/<Site>/BOLD/Preprocessed/<SubjectId>
    """
    import glob
    pattern = os.path.join(data_base, "T3", "*", folder, "BOLD", "Preprocessed", "*")
    names: Set[str] = set()
    for p in glob.iglob(pattern):
        if os.path.isdir(p):
            names.add(os.path.basename(p).strip())
    return names


def write_report(output_path: str, centre: str, subjects: List[str], present_set: Set[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "centre", "present_in_data"])
        for sid in subjects:
            writer.writerow([sid, centre, "yes" if sid in present_set else "no"])


def _resolve_qc_csv(data_base: str, timepoint_num: int) -> str:
    """Auto-resolve QC CSV path for a timepoint, trying common naming conventions."""
    # The CSVs live one level above Data/
    root = os.path.dirname(data_base)
    if timepoint_num == 3:
        candidates = [
            os.path.join(root, "LEAP3_QC_raport_all_sites.csv"),
        ]
    else:
        stem = f"LEAP{timepoint_num}_RS-fMRI_QC_final"
        candidates = [
            os.path.join(root, f"{stem}.tsv"),
            os.path.join(root, f"{stem}.csv"),
        ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Fallback (will fail later with a clear error)
    return os.path.abspath(candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check subjects by centre in CSV against site folders in Data/."
        )
    )
    parser.add_argument(
        "--csv-t1",
        default=None,
        help="Path to T1 QC CSV. Auto-resolved if omitted.",
    )
    parser.add_argument(
        "--csv-t2",
        default=None,
        help="Path to T2 QC CSV. Auto-resolved if omitted.",
    )
    parser.add_argument(
        "--csv-t3",
        default=None,
        help="Path to T3 QC CSV (default: LEAP3_QC_raport_all_sites.csv)",
    )
    parser.add_argument(
        "--data-base",
        default=DEFAULT_DATA_BASE,
        help="Path to base data directory containing site folders (default: Data)",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=[
            "CAMBRIDGE",
            "KINGS_COLLEGE",
            "MANNHEIM",
            "NIJMEGEN",
            "UTRECHT",
            "ROME",
            "LONDON_KCL",
        ],
        help="Centres to include (CSV values). Defaults to common sites including future ones.",
    )
    parser.add_argument(
        "--timepoints",
        nargs="*",
        default=["T1", "T2", "T3"],
        help="Timepoints to process (default: T1 T2 T3)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory to write per-site reports (default: reports)",
    )
    args = parser.parse_args()

    data_base = os.path.abspath(args.data_base)
    reports_dir = os.path.abspath(args.reports_dir)

    csv_t1_path = os.path.abspath(args.csv_t1) if args.csv_t1 else _resolve_qc_csv(data_base, 1)
    csv_t2_path = os.path.abspath(args.csv_t2) if args.csv_t2 else _resolve_qc_csv(data_base, 2)
    csv_t3_path = os.path.abspath(args.csv_t3) if args.csv_t3 else _resolve_qc_csv(data_base, 3)

    print(f"T1 CSV: {csv_t1_path}")
    print(f"T2 CSV: {csv_t2_path}")
    print(f"T3 CSV: {csv_t3_path}")
    print(f"Data base: {data_base}")
    print(f"Sites: {', '.join(args.sites)}")

    # Prepare mapping centre -> folder name under Data/
    missing_map = [c for c in args.sites if c not in CENTRE_TO_FOLDER]
    if missing_map:
        print("\nWarning: No folder mapping for centres:", ", ".join(missing_map))

    # Process selected timepoints
    for timepoint in args.timepoints:
        # Select the appropriate CSV for this timepoint
        if timepoint == "T1":
            csv_path = csv_t1_path
        elif timepoint == "T2":
            csv_path = csv_t2_path
        else:
            csv_path = csv_t3_path
        
        print(f"\n{'='*60}")
        print(f"PROCESSING TIMEPOINT: {timepoint}")
        print(f"Using CSV: {csv_path}")
        print(f"{'='*60}")
        
        # Read subjects from the appropriate CSV
        subjects_by = read_subjects_by_centre(csv_path, args.sites)
        
        grand_total = 0
        grand_present = 0
        grand_missing = 0

        for centre in args.sites:
            centre_subjects = subjects_by.get(centre, [])
            folder = CENTRE_TO_FOLDER.get(centre)
            csv_to_real: Dict[str, str] = {}

            print("\n------------------------------")
            print(f"Centre: {centre} ({timepoint})")
            print(f"Subjects in CSV: {len(centre_subjects)}")

            # Exclusion sets for this centre
            # T3 QC CSV only has a single 'exclude' column (no liberal/stringent split)
            if timepoint == "T3":
                excl_str = read_exclude_subjects(csv_path, "exclude", centre)
                excl_lib = []  # not available for T3
            else:
                excl_lib = read_exclude_subjects(csv_path, "exclude_liberal", centre)
                excl_str = read_exclude_subjects(csv_path, "exclude_stringent", centre)

            # For T3 subjects live under Data/T3/<batch>/<Site>/BOLD/Preprocessed/<SubjectId>
            if timepoint == "T3":
                if not folder:
                    print(f"Warning: No folder mapping for centre {centre}, skipping.")
                    grand_total += len(centre_subjects)
                    grand_missing += len(centre_subjects)
                    continue
                folder_names = list_subject_dirs_t3(data_base, folder)
                # Build rounding lookup: CSV IDs are truncated to 6 sig figs by Excel
                id_lookup = build_t3_id_lookup(data_base, folder)
                # Map CSV subject IDs to real folder IDs via rounding
                resolved_subjects: List[str] = []
                unresolved: List[str] = []
                csv_to_real = {}
                for sid in centre_subjects:
                    real_id = id_lookup.get(sid)
                    if real_id:
                        resolved_subjects.append(real_id)
                        csv_to_real[sid] = real_id
                    else:
                        unresolved.append(sid)
                if unresolved:
                    print(f"Warning: {len(unresolved)} CSV subject(s) could not be matched to any data folder (likely no data delivered)")
                # For T3 we check presence of the resolved (real) IDs
                present, missing_real = check_presence(resolved_subjects, folder_names)
                # Also include CSV-only subjects as missing
                missing = [sid for sid in centre_subjects if csv_to_real.get(sid, sid) not in set(present)]
                site_path = None  # T3 doesn't have a single site_path
            else:
                # T1/T2: standard layout
                site_path = os.path.join(data_base, timepoint, folder) if folder else None

            if timepoint != "T3":
                if not folder or not site_path or not os.path.isdir(site_path):
                    where = site_path if site_path else f"{data_base}/{timepoint}/<unknown>"
                    print(f"Warning: site folder missing: {where} (skipping presence check)")
                    out_csv = os.path.join(reports_dir, f"{(folder or centre).lower()}_{timepoint.lower()}_subjects_presence.csv")
                    write_report(out_csv, centre, centre_subjects, set())
                    print(f"Report written to: {out_csv}")
                    missing = list(centre_subjects)
                    liberal_overlap = sorted(set(missing) & set(excl_lib))
                    stringent_overlap = sorted(set(missing) & set(excl_str))
                    excluded_union = set(excl_lib) | set(excl_str)
                    remaining_missing = sorted(set(missing) - excluded_union)

                    print("Exclude-liberal analysis:")
                    print(f"Total exclude_liberal == 1: {len(excl_lib)}")
                    print(f"Overlap with missing subjects: {len(liberal_overlap)}")
                    print("Exclude-stringent analysis:")
                    print(f"Total exclude_stringent == 1: {len(excl_str)}")
                    print(f"Overlap with missing subjects: {len(stringent_overlap)}")
                    print("Remaining missing after exclusions:")
                    print(f"Count: {len(remaining_missing)}")

                    rem_csv = os.path.join(reports_dir, f"{(folder or centre).lower()}_{timepoint.lower()}_remaining_missing_after_exclusions.csv")
                    os.makedirs(reports_dir, exist_ok=True)
                    with open(rem_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["subject"])    
                        for sid in remaining_missing:
                            w.writerow([sid])
                    print(f"Remaining-missing report written to: {rem_csv}")
                    grand_total += len(centre_subjects)
                    grand_missing += len(centre_subjects)
                    continue

                folder_names = list_subject_dirs(site_path)
                present, missing = check_presence(centre_subjects, folder_names)

            print(f"Present in {folder} {timepoint} data folders: {len(present)}")
            print(f"Missing from {folder} {timepoint} data folders: {len(missing)}")
            if missing:
                print("Missing subjects (up to 20):")
                for sid in missing[:20]:
                    print(f"  - {sid}")

            out_csv = os.path.join(reports_dir, f"{folder.lower()}_{timepoint.lower()}_subjects_presence.csv")
            # For T3, present contains real (full) IDs; map back to CSV IDs for the report
            if timepoint == "T3" and csv_to_real:
                real_to_csv = {v: k for k, v in csv_to_real.items()}
                present_csv = {real_to_csv.get(sid, sid) for sid in present}
                write_report(out_csv, centre, centre_subjects, present_csv)
            else:
                write_report(out_csv, centre, centre_subjects, set(present))
            print(f"Report written to: {out_csv}")

            # Exclusions and remaining-missing for this centre
            liberal_overlap = sorted(set(missing) & set(excl_lib))
            stringent_overlap = sorted(set(missing) & set(excl_str))
            excluded_union = set(excl_lib) | set(excl_str)
            remaining_missing = sorted(set(missing) - excluded_union)

            if timepoint == "T3":
                # T3 only has a single 'exclude' column
                print("Exclude analysis:")
                print(f"Total exclude == 1: {len(excl_str)}")
                print(f"Overlap with missing subjects: {len(stringent_overlap)}")
                if stringent_overlap:
                    print("Overlapping subject IDs (up to 20):")
                    for sid in stringent_overlap[:20]:
                        print(f"  - {sid}")
            else:
                print("Exclude-liberal analysis:")
                print(f"Total exclude_liberal == 1: {len(excl_lib)}")
                print(f"Overlap with missing subjects: {len(liberal_overlap)}")
                if liberal_overlap:
                    print("Overlapping subject IDs (up to 20):")
                    for sid in liberal_overlap[:20]:
                        print(f"  - {sid}")
                print("Exclude-stringent analysis:")
                print(f"Total exclude_stringent == 1: {len(excl_str)}")
                print(f"Overlap with missing subjects: {len(stringent_overlap)}")
                if stringent_overlap:
                    print("Overlapping subject IDs (up to 20):")
                    for sid in stringent_overlap[:20]:
                        print(f"  - {sid}")
            print("Remaining missing after exclusions:")
            print(f"Count: {len(remaining_missing)}")
            if remaining_missing:
                print("Subject IDs (up to 20):")
                for sid in remaining_missing[:20]:
                    print(f"  - {sid}")

            rem_csv = os.path.join(reports_dir, f"{folder.lower()}_{timepoint.lower()}_remaining_missing_after_exclusions.csv")
            os.makedirs(reports_dir, exist_ok=True)
            with open(rem_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["subject"])    
                for sid in remaining_missing:
                    w.writerow([sid])
            print(f"Remaining-missing report written to: {rem_csv}")

            # NIfTI QC for present subjects in this centre
            print("\nNIfTI QC:")
            if nib is None:
                print("nibabel not installed; skipping NIfTI QC.")
            else:
                expected_vols = build_expected_volumes(csv_path, centre)
                qc_rows: List[Dict[str, Any]] = []
                for sid in present:
                    if timepoint == "T3":
                        # For T3, find the batch folder containing this subject
                        import glob as _glob
                        matches = _glob.glob(os.path.join(data_base, "T3", "*", folder, "BOLD", "Preprocessed", sid))
                        subj_site_path = os.path.dirname(matches[0]) if matches else ""
                    else:
                        subj_site_path = site_path
                    nifti = subject_nifti_path(sid, subj_site_path, timepoint)
                    info = read_nifti_info(nifti)
                    # For T3, present has real IDs but expected_vols keys are CSV IDs
                    if timepoint == "T3" and csv_to_real:
                        real_to_csv_nifti = {v: k for k, v in csv_to_real.items()}
                        exp_vol = expected_vols.get(real_to_csv_nifti.get(sid, sid))
                    else:
                        exp_vol = expected_vols.get(sid)
                    row = {
                        "subject": sid,
                        "exists": info["exists"],
                        "file_size": format_mb(info["file_size_bytes"]),
                        "voxel_size": info["voxel_size_mm"] or (),
                        "shape": info["shape"] or (),
                        "volumes": info["volumes"],
                        "expected_volumes": exp_vol,
                        "flags": [],
                    }
                    if not info["exists"]:
                        row["flags"].append("missing_file")
                    if info.get("error"):
                        row["flags"].append(str(info["error"]))
                    vox = info["voxel_size_mm"]
                    if vox and len(vox) >= 3:
                        tol = 0.2
                        exp = (2.0, 2.0, 2.0)
                        if any(abs(vox[i] - exp[i]) > tol for i in range(3)):
                            row["flags"].append("voxel_size_unusual")
                    if row["expected_volumes"] is not None and row["volumes"] is not None:
                        if int(row["volumes"]) != int(row["expected_volumes"]):
                            row["flags"].append("volumes_mismatch")
                    fs_bytes = info["file_size_bytes"]
                    if fs_bytes is not None and fs_bytes < 20 * 1024 * 1024:
                        row["flags"].append("file_too_small")
                    qc_rows.append(row)

                qc_rows.sort(key=lambda r: (0 if r["flags"] else 1, r["volumes"] if r["volumes"] is not None else 0))
                print("subject | exists | file_size | voxel(mm) | shape | volumes | expected | flags")
                for r in qc_rows[:30]:
                    vox_str = "x".join(str(v) for v in (r["voxel_size"] or ())) or "-"
                    shape_str = "x".join(str(s) for s in (r["shape"] or ())) or "-"
                    flags_str = ",".join(r["flags"]) if r["flags"] else ""
                    vol_str = str(r['volumes']) if r['volumes'] is not None else '-'
                    exp_str = str(r['expected_volumes']) if r['expected_volumes'] is not None else '-'
                    print(
                        f"{r['subject']} | {r['exists']} | {r['file_size']} | {vox_str} | {shape_str} | {vol_str} | {exp_str} | {flags_str}"
                    )

                qc_csv = os.path.join(reports_dir, f"{folder.lower()}_{timepoint.lower()}_nifti_qc.csv")
                with open(qc_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "subject",
                        "exists",
                        "file_size_mb",
                        "voxel_size_mm",
                        "shape",
                        "volumes",
                        "expected_volumes",
                        "flags",
                    ])
                    for r in qc_rows:
                        writer.writerow([
                            r["subject"],
                            r["exists"],
                            (r["file_size"]).replace("MB", ""),
                            "x".join(str(v) for v in (r["voxel_size"] or ())) or "-",
                            "x".join(str(s) for s in (r["shape"] or ())) or "-",
                            r["volumes"] if r["volumes"] is not None else "-",
                            r["expected_volumes"] if r["expected_volumes"] is not None else "-",
                            ",".join(r["flags"]) if r["flags"] else "",
                        ])
                print(f"NIfTI QC report written to: {qc_csv}")

            grand_total += len(centre_subjects)
            grand_present += len(present)
            grand_missing += len(missing)

        print(f"\n{'='*60}")
        print(f"Summary for {timepoint}:")
        print(f"Total CSV subjects (selected centres): {grand_total}")
        print(f"Total present: {grand_present}")
        print(f"Total missing: {grand_missing}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
