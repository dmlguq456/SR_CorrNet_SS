"""Checkpoint migration utility for SR_CorrNet model variants.

Remaps old state_dict key names to new names after layer attribute renames.
Supports single-file, directory, and full-project batch migration.

Key mapping categories:
    A: frame_wise_block/freq_wise_block  -> freq_block/time_block  (Multi_Path_Block inner)
    B: vald_estim                        -> vad_doa_estim          (legacy; harmless on SS)
    H: .cs.                              -> .ca.                   (TransDecoderBlock cross-attn)
    I: .block.ega.                       -> .block.sa.             (CS_TransBlock ModuleDict)
    J: input_layer.                      -> encoder.               (all variants)
    K: mask_estim. / mask_estim_aux.     -> filter_estim. / filter_estim_aux.  (all variants)
"""

import argparse
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Key mapping tables
# ---------------------------------------------------------------------------

# Prefix-based mappings applied via str.startswith().
# IMPORTANT: mask_estim_aux. must come before mask_estim. in this list.
# Rationale: "mask_estim_aux.".startswith("mask_estim.") is False (the 12th
# character is '_', not '.'), so ordering is not strictly required for
# correctness. However, keeping aux first is defensive coding practice that
# makes intent explicit and avoids any future ambiguity.
PREFIX_KEY_MAP = [
    ("input_layer.", "encoder."),           # J
    ("mask_estim_aux.", "filter_estim_aux."),  # K (aux) — before mask_estim!
    ("mask_estim.", "filter_estim."),       # K (main)
    ("vald_estim.", "vad_doa_estim."),      # B
]

# Substring-based mappings applied via str.replace(..., count=1).
# Each pattern is dot-delimited to avoid false matches on partial substrings.
# Safety notes:
#   - ".cs."  matches only TransDecoderBlock keys (spk_split.dec.net*.cs.*).
#             It does NOT match dec_cs.* keys because "dec_cs" contains "_cs",
#             not ".cs." — the preceding character is '_', not '.'.
#   - ".block.ega." is unique to CS_TransBlock ModuleDict keys.
SUBSTRING_KEY_MAP = [
    (".frame_wise_block.", ".freq_block."),  # A
    (".freq_wise_block.", ".time_block."),   # A
    (".cs.", ".ca."),                        # H
    (".block.ega.", ".block.sa."),           # I
]


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------

def migrate_state_dict(state_dict, key_map=None):
    """Remap state_dict keys from old layer names to new layer names.

    Applies prefix-based mappings first (str.startswith), then substring-based
    mappings (str.replace, count=1). Each key is processed by at most one
    prefix mapping and at most one substring mapping.

    This function is idempotent: applying it to already-migrated keys is a
    no-op because old patterns do not appear in new key names. For example,
    "encoder." does not startswith "input_layer.", and ".freq_block." does not
    contain ".frame_wise_block.". Double-applying the migration is safe.

    Args:
        state_dict (OrderedDict): Model state dict with potentially old key names.
        key_map (list of tuple, optional): List of (old_pattern, new_pattern)
            tuples. If None, uses the combined default
            (PREFIX_KEY_MAP + SUBSTRING_KEY_MAP).

    Returns:
        tuple:
            new_state_dict (OrderedDict): State dict with remapped keys.
            migration_log (list of tuple): List of (old_key, new_key) pairs for
                keys that were changed. Unchanged keys are not included.
    """
    if key_map is None:
        key_map = PREFIX_KEY_MAP + SUBSTRING_KEY_MAP

    # Split key_map into prefix and substring maps by inspecting pattern shape.
    # A pattern is a prefix mapping if it does not start with '.' (dots indicate
    # substring context). This matches the structure of PREFIX_KEY_MAP and
    # SUBSTRING_KEY_MAP above.
    prefix_map = [(old, new) for old, new in key_map if not old.startswith(".")]
    substring_map = [(old, new) for old, new in key_map if old.startswith(".")]

    new_state_dict = OrderedDict()
    migration_log = []

    for old_key, value in state_dict.items():
        new_key = old_key

        # Step 1: Try prefix mappings (apply at most one).
        for old_prefix, new_prefix in prefix_map:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break  # Only one prefix mapping per key.

        # Step 2: Try substring mappings (apply at most one per mapping).
        for old_sub, new_sub in substring_map:
            if old_sub in new_key:
                new_key = new_key.replace(old_sub, new_sub, 1)
                # Do not break: multiple distinct substrings could theoretically
                # match, but in practice each key contains at most one match.

        new_state_dict[new_key] = value
        if new_key != old_key:
            migration_log.append((old_key, new_key))

    return new_state_dict, migration_log


def get_key_map(variant):
    """Return the combined key map for the given model variant.

    All mappings are combined since non-matching patterns are no-ops —
    applying SE-specific patterns to SS checkpoints (or vice versa) is safe.

    Args:
        variant (str): "SS".

    Returns:
        list of tuple: Combined (old_pattern, new_pattern) list.
    """
    # All patterns are safe to apply universally because:
    # - Prefix patterns only match their specific key prefixes.
    # - Substring patterns are unique enough to avoid cross-variant collisions.
    return PREFIX_KEY_MAP + SUBSTRING_KEY_MAP


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _extract_epoch(pth_path):
    """Extract epoch number from filename like 'epoch.NNNN.pth'.

    Returns -1 if the filename does not match the expected pattern.
    """
    match = re.search(r"epoch\.(\d+)\.pth$", pth_path.name)
    if match:
        return int(match.group(1))
    return -1


def _find_latest_per_dir(pth_files):
    """For each parent directory, return only the file with the highest epoch number.

    Groups .pth files by their parent directory path, sorts each group by epoch
    number extracted from 'epoch.NNNN.pth' filenames, and returns the highest
    epoch file per directory.

    Args:
        pth_files (list of Path): List of .pth file paths.

    Returns:
        list of Path: One file per parent directory (the one with highest epoch).
    """
    dir_groups = {}
    for f in pth_files:
        parent = f.parent
        dir_groups.setdefault(parent, []).append(f)

    latest_files = []
    for parent, files in dir_groups.items():
        # Sort by epoch number; fall back to filename string sort for non-standard names.
        files_with_epoch = [(f, _extract_epoch(f)) for f in files]
        files_with_epoch.sort(key=lambda x: (x[1], x[0].name))
        latest_files.append(files_with_epoch[-1][0])

    return latest_files


def _infer_variant_from_path(pth_path):
    """Infer model variant from the checkpoint file path.

    Looks for 'SR_CorrNet_SS' in the path string.
    Returns None if the variant cannot be determined.

    Args:
        pth_path (Path): Path to a .pth file.

    Returns:
        str or None: "SS", or None if not determinable.
    """
    path_str = str(pth_path)
    if "SR_CorrNet_SS" in path_str:
        return "SS"
    return None


def _migrate_file(pth_path, variant, backup=True, dry_run=False):
    """Load a checkpoint file, apply key migration, and optionally save it back.

    Args:
        pth_path (Path): Path to the .pth checkpoint file.
        variant (str): Model variant string ("SS"). Used for
            display only — the key map is always the full combined map.
        backup (bool): If True, copy the original file to <path>.bak before
            overwriting. Ignored when dry_run=True.
        dry_run (bool): If True, show what would change without modifying files.

    Returns:
        tuple: (n_migrated, n_total) — number of keys changed and total keys.
    """
    checkpoint = torch.load(pth_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        print(f"  [SKIP] No 'model_state_dict' key in {pth_path}")
        return 0, 0

    old_sd = checkpoint["model_state_dict"]
    key_map = get_key_map(variant)
    new_sd, migration_log = migrate_state_dict(old_sd, key_map)

    n_total = len(old_sd)
    n_migrated = len(migration_log)

    if dry_run:
        print(f"  [DRY-RUN] {pth_path}")
        print(f"    Keys: {n_total} total, {n_migrated} would be remapped")
        for old_k, new_k in migration_log:
            print(f"    {old_k!r:60s} -> {new_k!r}")
    else:
        if n_migrated == 0:
            print(f"  [SKIP] {pth_path.name} — 0 keys to remap (already migrated?)")
            return 0, n_total

        if backup:
            bak_path = pth_path.with_suffix(".pth.bak")
            shutil.copy2(pth_path, bak_path)
            print(f"  [BAK]  {bak_path.name}")

        checkpoint["model_state_dict"] = new_sd
        torch.save(checkpoint, pth_path)
        print(f"  [DONE] {pth_path.name} — {n_migrated}/{n_total} keys remapped")

    return n_migrated, n_total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Migrate SR_CorrNet checkpoint state_dict keys after layer renames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run on a single file
  python -m utils.migrate_checkpoint --path models/SR_CorrNet_SS/log/log_1ch_WSJ_fix_2spk/scratch_weights/epoch.0025.pth --variant SS --dry-run

  # Migrate all files in a directory
  python -m utils.migrate_checkpoint --dir models/SR_CorrNet_SS/log/log_1ch_WSJ_fix_2spk/scratch_weights/ --variant SS

  # Migrate only the latest checkpoint per directory
  python -m utils.migrate_checkpoint --all --latest-only --backup

  # Dry-run across all checkpoints, latest only
  python -m utils.migrate_checkpoint --all --latest-only --dry-run
        """,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--path", type=Path, metavar="PATH",
        help="Migrate a single .pth checkpoint file."
    )
    source_group.add_argument(
        "--dir", type=Path, metavar="DIR",
        help="Migrate all .pth files in a directory."
    )
    source_group.add_argument(
        "--all", action="store_true",
        help="Scan all models/SR_CorrNet_*/log/**/*.pth files."
    )

    parser.add_argument(
        "--variant", choices=["SS"], metavar="{SS}",
        help="Model variant (required for --path and --dir; inferred from path for --all)."
    )
    parser.add_argument(
        "--latest-only", action="store_true",
        help="When used with --dir or --all, process only the latest epoch file per directory."
    )
    parser.add_argument(
        "--backup", action="store_true", default=True,
        help="Create a .pth.bak backup before overwriting (default: True)."
    )
    parser.add_argument(
        "--no-backup", action="store_false", dest="backup",
        help="Skip creating .pth.bak backup files."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show changes without modifying any files."
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Determine the list of files to process.
    pth_files = []

    if args.path:
        if not args.path.is_file():
            parser.error(f"File not found: {args.path}")
        if args.variant is None:
            parser.error("--variant is required with --path")
        pth_files = [args.path]

    elif args.dir:
        if not args.dir.is_dir():
            parser.error(f"Directory not found: {args.dir}")
        if args.variant is None:
            parser.error("--variant is required with --dir")
        pth_files = sorted(args.dir.glob("*.pth"))
        if not pth_files:
            print(f"No .pth files found in {args.dir}")
            return
        if args.latest_only:
            pth_files = _find_latest_per_dir(pth_files)

    elif args.all:
        # Resolve project root relative to this script's location.
        project_root = Path(__file__).resolve().parent.parent
        pth_files = sorted(project_root.glob("models/SR_CorrNet_*/log/**/*.pth"))
        # Exclude .bak files that happen to match the glob (safety).
        pth_files = [f for f in pth_files if not f.name.endswith(".bak")]
        if not pth_files:
            print("No .pth files found under models/SR_CorrNet_*/log/")
            return
        if args.latest_only:
            pth_files = _find_latest_per_dir(pth_files)

    # Process each file.
    total_migrated = 0
    total_keys = 0
    skipped = 0

    for pth_path in pth_files:
        # Determine variant for this file.
        if args.all:
            variant = _infer_variant_from_path(pth_path)
            if variant is None:
                print(f"  [WARN] Cannot infer variant from path, skipping: {pth_path}")
                skipped += 1
                continue
        else:
            variant = args.variant

        n_migrated, n_total = _migrate_file(
            pth_path,
            variant=variant,
            backup=args.backup,
            dry_run=args.dry_run,
        )
        total_migrated += n_migrated
        total_keys += n_total

    # Summary.
    mode_label = "[DRY-RUN] " if args.dry_run else ""
    print(
        f"\n{mode_label}Summary: {len(pth_files)} file(s) processed, "
        f"{total_migrated} keys remapped out of {total_keys} total"
        + (f", {skipped} skipped" if skipped else "")
    )


if __name__ == "__main__":
    main()
