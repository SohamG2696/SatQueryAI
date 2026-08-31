"""
ml/inference/semantic_grounder.py
==================================
Category-Aware Semantic Grounding Module for SECOND Dataset.

Decodes SECOND semantic label maps (RGB format) and computes per-category 
change statistics using T1/T2 semantic labels and ChangeFormer binary change predictions.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
from PIL import Image

# 100% Empirically Verified RGB Palette mapping for SECOND / CDVQA Dataset
COLOR_TO_CLASS: Dict[Tuple[int, int, int], str] = {
    (0, 0, 255): "water",
    (0, 128, 0): "low_vegetation",
    (0, 255, 0): "trees",
    (128, 0, 0): "buildings",
    (128, 128, 128): "NVG_surface",
    (255, 0, 0): "playgrounds",
}

CLASS_TO_COLOR: Dict[str, Tuple[int, int, int]] = {v: k for k, v in COLOR_TO_CLASS.items()}

# Recognized class aliases
CLASS_ALIASES: Dict[str, str] = {
    "water": "water",
    "low_vegetation": "low_vegetation",
    "low vegetation": "low_vegetation",
    "vegetation": "low_vegetation",
    "trees": "trees",
    "tree": "trees",
    "buildings": "buildings",
    "building": "buildings",
    "nvg_surface": "NVG_surface",
    "nvg surface": "NVG_surface",
    "non-vegetated ground surface": "NVG_surface",
    "non vegetated ground surface": "NVG_surface",
    "non-vegetated": "NVG_surface",
    "ground": "NVG_surface",
    "playgrounds": "playgrounds",
    "playground": "playgrounds",
}

ALL_CLASSES = ["water", "low_vegetation", "trees", "buildings", "NVG_surface", "playgrounds"]


def normalize_category_name(category: str) -> Optional[str]:
    """Normalize a category query string into one of the 6 canonical class names."""
    if not category:
        return None
    key = category.strip().lower()
    return CLASS_ALIASES.get(key, None)


class SemanticGrounder:
    """
    Decodes SECOND RGB labels and computes category-aware change statistics.
    """

    def __init__(self, second_root: Optional[Union[str, Path]] = None):
        if second_root is not None:
            self.second_root = Path(second_root)
        else:
            # Check local SatQueryAI path first, then fallback to parent SIH2026 path
            this_file = Path(__file__).resolve()
            satquery_root = this_file.parents[2]
            sih2026_root = satquery_root.parent

            candidate_local = satquery_root / "datasets" / "SECOND" / "SECOND_train_set"
            candidate_parent = sih2026_root / "datasets" / "SECOND" / "SECOND_train_set"

            if candidate_local.exists():
                self.second_root = candidate_local
            elif candidate_parent.exists():
                self.second_root = candidate_parent
            else:
                self.second_root = candidate_local

        self.label1_dir = self.second_root / "label1"
        self.label2_dir = self.second_root / "label2"

    def load_label(self, source: Union[str, Path, Image.Image, np.ndarray], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load an RGB semantic label array [H, W, 3] uint8.
        Optional target_size (H, W) resizes with Nearest Neighbor.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Label file not found: {path}")
            pil_img = Image.open(path).convert("RGB")
        elif isinstance(source, Image.Image):
            pil_img = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[2] == 3:
                pil_img = Image.fromarray(source.astype(np.uint8))
            else:
                raise ValueError(f"Expected array shape (H, W, 3), got {source.shape}")
        else:
            raise TypeError(f"Unsupported label source type: {type(source)}")

        if target_size is not None and pil_img.size[::-1] != target_size:
            # pil_img.size is (W, H); target_size is (H, W)
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.NEAREST)

        return np.array(pil_img, dtype=np.uint8)

    def find_label_paths(self, image_ref: Union[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Given an image filename or path, locate corresponding label1 and label2 files.
        """
        filename = Path(image_ref).name
        l1 = self.label1_dir / filename
        l2 = self.label2_dir / filename
        return (l1 if l1.exists() else None, l2 if l2.exists() else None)

    def decode_class_mask(self, label_rgb: np.ndarray, category: str) -> np.ndarray:
        """
        Get boolean mask (H, W) where label matches the given category.
        """
        norm_cat = normalize_category_name(category)
        if norm_cat is None or norm_cat not in CLASS_TO_COLOR:
            return np.zeros(label_rgb.shape[:2], dtype=bool)

        target_rgb = CLASS_TO_COLOR[norm_cat]
        return np.all(label_rgb == target_rgb, axis=2)

    def compute_category_change(
        self,
        label_t1: np.ndarray,
        label_t2: np.ndarray,
        change_mask: np.ndarray,
        category: str,
    ) -> Dict[str, Any]:
        """
        Compute category-aware change statistics for a specific category.

        Parameters
        ----------
        label_t1 : [H, W, 3] uint8 RGB array
        label_t2 : [H, W, 3] uint8 RGB array
        change_mask : [H, W] binary change mask (0/1 or 0/255)
        category : str

        Returns
        -------
        dict with category, t1_pixels, t2_pixels, changed_pixels, change_ratio, change_present, area_delta
        """
        norm_cat = normalize_category_name(category)
        if norm_cat is None:
            return {
                "category": category,
                "canonical_category": "unknown",
                "t1_pixels": 0,
                "t2_pixels": 0,
                "changed_pixels": 0,
                "change_ratio": 0.0,
                "change_present": False,
                "area_delta": 0,
            }

        # Normalize change mask to binary 0/1
        binary_change = (change_mask > 0).astype(bool)

        # Get class masks
        m_t1 = self.decode_class_mask(label_t1, norm_cat)
        m_t2 = self.decode_class_mask(label_t2, norm_cat)
        m_union = m_t1 | m_t2

        t1_pixels = int(m_t1.sum())
        t2_pixels = int(m_t2.sum())
        area_delta = t2_pixels - t1_pixels

        # Changed pixels within this category's region (union of T1 and T2)
        changed_pixels = int((binary_change & m_union).sum())
        total_cat_pixels = int(m_union.sum())

        if total_cat_pixels > 0:
            change_ratio = float(changed_pixels / total_cat_pixels)
        else:
            change_ratio = 0.0

        # Change present if changed pixels > 10 and ratio > 0.01
        change_present = (changed_pixels >= 10) and (change_ratio > 0.01)

        return {
            "category": category,
            "canonical_category": norm_cat,
            "t1_pixels": t1_pixels,
            "t2_pixels": t2_pixels,
            "changed_pixels": changed_pixels,
            "change_ratio": round(change_ratio, 4),
            "change_present": change_present,
            "area_delta": area_delta,
        }

    def compute_all_category_changes(
        self,
        label_t1: np.ndarray,
        label_t2: np.ndarray,
        change_mask: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute category-aware change statistics for all 6 classes.
        """
        results = {}
        for cat in ALL_CLASSES:
            results[cat] = self.compute_category_change(label_t1, label_t2, change_mask, cat)
        return results

    def compute_transitions(
        self,
        label_t1: np.ndarray,
        label_t2: np.ndarray,
        change_mask: np.ndarray,
        from_category: Optional[str] = None,
    ) -> List[Tuple[str, str, int]]:
        """
        Calculate semantic transitions (T1 class -> T2 class) in changed regions.

        Returns list of tuples: [(from_class, to_class, pixel_count), ...]
        sorted by pixel count descending.
        """
        binary_change = (change_mask > 0).astype(bool)
        if not np.any(binary_change):
            return []

        filter_from = normalize_category_name(from_category) if from_category else None

        transitions: Dict[Tuple[str, str], int] = {}

        # Loop over classes to build vectorised transition matrix
        for rgb_1, cat_1 in COLOR_TO_CLASS.items():
            if filter_from and cat_1 != filter_from:
                continue
            m1 = np.all(label_t1 == rgb_1, axis=2) & binary_change
            if not np.any(m1):
                continue
            for rgb_2, cat_2 in COLOR_TO_CLASS.items():
                if cat_1 == cat_2:
                    continue  # Only count actual class transitions
                m2 = np.all(label_t2 == rgb_2, axis=2)
                count = int((m1 & m2).sum())
                if count > 0:
                    transitions[(cat_1, cat_2)] = count

        sorted_trans = sorted(
            [(k[0], k[1], v) for k, v in transitions.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        return sorted_trans


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMANTIC GROUNDER — SELF TEST")
    print("=" * 70)

    grounder = SemanticGrounder()

    l1_path, l2_path = grounder.find_label_paths("00003.png")
    if l1_path and l2_path:
        print(f"Found labels for 00003.png:")
        print(f"  T1: {l1_path.name}")
        print(f"  T2: {l2_path.name}")

        l1 = grounder.load_label(l1_path)
        l2 = grounder.load_label(l2_path)
        mask = (np.any(l1 != l2, axis=2)).astype(np.uint8)

        all_stats = grounder.compute_all_category_changes(l1, l2, mask)
        for cat, stats in all_stats.items():
            print(f"  {cat:15s}: changed={stats['changed_pixels']:5d}, ratio={stats['change_ratio']*100:6.2f}%, delta={stats['area_delta']:+6d}")

        transitions = grounder.compute_transitions(l1, l2, mask)
        print(f"\nTop Transitions for 00003.png:")
        for f_cat, t_cat, count in transitions[:5]:
            print(f"  {f_cat} -> {t_cat}: {count} pixels")

    print("\nSelf-test complete.")
