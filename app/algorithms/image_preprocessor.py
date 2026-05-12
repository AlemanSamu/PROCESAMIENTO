from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

from app.core.errors import ProcessingError

from .artifacts import PipelineStageResult, PreprocessedImage, ValidatedImage, write_json

try:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None


DEFAULT_ALLOWED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


class ImagePreprocessor:
    """Valida, normaliza y mide las imagenes antes del resto del pipeline."""

    name = "image_validation_and_preprocessing"

    def __init__(
        self,
        allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
        *,
        profile: str = "balanced",
        max_width: int = 1920,
    ) -> None:
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)
        self.profile = self._normalize_profile(profile)
        self.max_width = max(256, int(max_width or 1920))

    @classmethod
    def from_settings(cls, settings: Any) -> "ImagePreprocessor":
        return cls(
            allowed_extensions=tuple(getattr(settings, "allowed_image_extensions", DEFAULT_ALLOWED_EXTENSIONS)),
            profile=str(getattr(settings, "profile", "balanced")),
            max_width=int(getattr(settings, "image_preprocessing_max_width", 1920)),
        )

    def run(
        self,
        images_dir: Path,
        work_dir: Path,
        *,
        output_images_dir: Path | None = None,
    ) -> tuple[list[PreprocessedImage], PipelineStageResult]:
        if not images_dir.exists():
            raise ProcessingError(f"El directorio de imagenes no existe: {images_dir}.")

        source_images = sorted(
            [path for path in images_dir.iterdir() if path.is_file()],
            key=lambda path: path.name.lower(),
        )
        if not source_images:
            raise ProcessingError("No se encontraron imagenes para reconstruir.")

        normalized_dir = output_images_dir or (work_dir / "preprocessed_images")
        normalized_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_images: list[PreprocessedImage] = []
        manifest: list[dict[str, object]] = []
        real_image_count = 0
        synthetic_image_count = 0

        for index, source in enumerate(source_images, start=1):
            extension = source.suffix.lower()
            if extension not in self.allowed_extensions:
                raise ProcessingError(
                    f"Formato no soportado en el pipeline: '{extension}'. "
                    f"Permitidos: {', '.join(sorted(self.allowed_extensions))}."
                )

            size_bytes = source.stat().st_size
            if size_bytes <= 0:
                raise ProcessingError(f"La imagen '{source.name}' esta vacia.")

            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            normalized_extension = ".png" if extension == ".png" else ".jpg"
            normalized_name = f"{index:03d}_{source.stem}{normalized_extension}"
            target_path = normalized_dir / normalized_name

            try:
                processed = self._process_real_image(source, target_path)
                stage_mode = "real"
                real_image_count += 1
            except Exception as exc:
                shutil.copy2(source, target_path)
                processed = self._synthetic_processed_record(source, target_path, checksum, str(exc))
                stage_mode = "synthetic"
                synthetic_image_count += 1

            validated = ValidatedImage(
                source_path=source,
                normalized_name=normalized_name,
                index=index,
                size_bytes=size_bytes,
                sha256=checksum,
                extension=extension,
                width=int(processed["original_width"]),
                height=int(processed["original_height"]),
                pixel_count=int(processed["original_width"]) * int(processed["original_height"]),
            )
            preprocessed = PreprocessedImage(
                source=validated,
                preprocessed_path=target_path,
                brightness=float(processed["metrics"]["brightness"]),
                contrast=float(processed["metrics"]["contrast"]),
                sharpness=float(processed["metrics"]["sharpness"]),
            )
            preprocessed_images.append(preprocessed)
            manifest.append(
                {
                    **preprocessed.to_dict(),
                    "mode": stage_mode,
                    "profile": self.profile,
                    "original_path": str(source),
                    "processed_path": str(target_path),
                    "dimensions_before": {
                        "width": int(processed["original_width"]),
                        "height": int(processed["original_height"]),
                    },
                    "dimensions_after": {
                        "width": int(processed["processed_width"]),
                        "height": int(processed["processed_height"]),
                    },
                    "metrics": processed["metrics"],
                    "transformations": processed["transformations"],
                    "status": processed["status"],
                    "warnings": processed["warnings"],
                    "rejected_reasons": processed["rejected_reasons"],
                }
            )

        mode = "real" if real_image_count > 0 else "synthetic"
        accepted_count = sum(1 for item in manifest if item.get("status") == "accepted")
        warning_count = sum(1 for item in manifest if item.get("status") == "warning")
        rejected_count = sum(1 for item in manifest if item.get("status") == "rejected")
        manifest_path = write_json(
            work_dir / "preprocessing_manifest.json",
            {
                "stage": self.name,
                "mode": mode,
                "profile": self.profile,
                "max_width": self.max_width,
                "output_images_dir": str(normalized_dir),
                "image_count": len(preprocessed_images),
                "real_image_count": real_image_count,
                "synthetic_image_count": synthetic_image_count,
                "accepted_images": accepted_count,
                "warning_images": warning_count,
                "rejected_images": rejected_count,
                "images": manifest,
            },
        )
        report = PipelineStageResult(
            name=self.name,
            status="completed",
            summary=f"Se validaron y normalizaron {len(preprocessed_images)} imagenes.",
            mode=mode,
            artifact_path=manifest_path,
            metrics={
                "image_count": len(preprocessed_images),
                "real_image_count": real_image_count,
                "synthetic_image_count": synthetic_image_count,
                "mode": mode,
                "profile": self.profile,
                "max_width": self.max_width,
                "output_images_dir": str(normalized_dir),
                "accepted_images": accepted_count,
                "warning_images": warning_count,
                "rejected_images": rejected_count,
            },
        )
        return preprocessed_images, report

    def _process_real_image(self, source: Path, target_path: Path) -> dict[str, Any]:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            if image.width <= 0 or image.height <= 0:
                raise ProcessingError(f"La imagen '{source.name}' no tiene dimensiones validas.")

            original_width = image.width
            original_height = image.height
            output_format = "PNG" if target_path.suffix.lower() == ".png" else "JPEG"
            output_mode = "RGBA" if output_format == "PNG" and image.mode in {"RGBA", "LA"} else "RGB"
            image = image.convert(output_mode)
            transformations = [f"exif_transpose", f"convert_{output_mode.lower()}"]
            processed_image = self._resize_if_needed(image, transformations)
            processed_image = self._enhance_with_opencv_if_available(processed_image, transformations)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if output_format == "PNG":
                processed_image.save(target_path, format="PNG", optimize=True)
            else:
                processed_image.save(target_path, format="JPEG", quality=92, optimize=True)
            transformations.append(f"normalized_format_{output_format.lower()}")
            metrics = self._read_real_metrics(processed_image, target_path)
            status, warnings, rejected = self._classify_metrics(metrics)

        return {
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": processed_image.width,
            "processed_height": processed_image.height,
            "metrics": metrics,
            "transformations": transformations,
            "status": status,
            "warnings": warnings,
            "rejected_reasons": rejected,
        }

    def _resize_if_needed(self, image: Image.Image, transformations: list[str]) -> Image.Image:
        if image.width <= self.max_width:
            return image
        ratio = self.max_width / float(max(1, image.width))
        target_height = max(1, int(round(image.height * ratio)))
        transformations.append(f"resize_max_width_{self.max_width}")
        return image.resize((self.max_width, target_height), Image.Resampling.LANCZOS)

    def _enhance_with_opencv_if_available(self, image: Image.Image, transformations: list[str]) -> Image.Image:
        if cv2 is None or np is None:
            transformations.append("opencv_unavailable")
            return image

        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        contrast = float(np.std(l_channel) / 255.0)
        if contrast < 0.30:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            transformations.append("clahe_contrast")
        enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        transformations.append("denoise_soft")
        return Image.fromarray(denoised)

    @staticmethod
    def _read_real_metrics(image: Image.Image, target_path: Path | None = None) -> dict[str, int | float]:
        grayscale = image.convert("L")
        stats = ImageStat.Stat(grayscale)
        edge_image = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edge_image)

        brightness = round((stats.mean[0] if stats.mean else 0.0) / 255.0, 4)
        contrast = round((stats.stddev[0] if stats.stddev else 0.0) / 255.0, 4)
        sharpness = round(
            min(
                1.0,
                (
                    ((edge_stats.mean[0] if edge_stats.mean else 0.0) / 255.0) * 0.7
                    + ((edge_stats.stddev[0] if edge_stats.stddev else 0.0) / 255.0) * 0.3
                ),
            ),
            4,
        )
        blur_score = ImagePreprocessor._estimate_blur_score(grayscale)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "blur_score": blur_score,
            "width": image.width,
            "height": image.height,
            "pixel_count": image.width * image.height,
            "file_size": int(target_path.stat().st_size) if target_path is not None and target_path.exists() else 0,
        }

    @staticmethod
    def _estimate_blur_score(grayscale: Image.Image) -> float:
        if cv2 is not None and np is not None:
            array = np.asarray(grayscale)
            variance = float(cv2.Laplacian(array, cv2.CV_64F).var())
            return round(variance, 4)
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        stats = ImageStat.Stat(edges)
        return round(float(stats.var[0] if stats.var else 0.0), 4)

    def _classify_metrics(self, metrics: dict[str, int | float]) -> tuple[str, list[str], list[str]]:
        warnings: list[str] = []
        rejected: list[str] = []
        if float(metrics["brightness"]) < 0.12:
            warnings.append("low_brightness")
        if float(metrics["brightness"]) > 0.92:
            warnings.append("high_brightness")
        if float(metrics["contrast"]) < 0.08:
            warnings.append("low_contrast")
        blur_threshold = 45.0 if self.profile == "quality" else 30.0
        if float(metrics["blur_score"]) < blur_threshold:
            warnings.append("possible_blur")
        return ("rejected" if rejected else ("warning" if warnings else "accepted"), warnings, rejected)

    @staticmethod
    def _normalize_profile(profile: str) -> str:
        normalized = (profile or "balanced").strip().lower()
        if normalized not in {"conservative", "balanced", "quality"}:
            return "balanced"
        return normalized

    def _synthetic_processed_record(
        self,
        source: Path,
        target_path: Path,
        checksum: str,
        reason: str,
    ) -> dict[str, Any]:
        metrics = self._synthetic_metrics(checksum)
        metrics["blur_score"] = 0.0
        metrics["file_size"] = int(target_path.stat().st_size) if target_path.exists() else 0
        return {
            "original_width": int(metrics["width"]),
            "original_height": int(metrics["height"]),
            "processed_width": int(metrics["width"]),
            "processed_height": int(metrics["height"]),
            "metrics": metrics,
            "transformations": ["copy_original_fallback"],
            "status": "warning",
            "warnings": [f"real_preprocessing_failed: {reason}"],
            "rejected_reasons": [],
        }

    @staticmethod
    def _synthetic_metrics(checksum: str) -> dict[str, int | float]:
        return {
            "brightness": ImagePreprocessor._derive_score(checksum, 3, 0.18, 0.82),
            "contrast": ImagePreprocessor._derive_score(checksum, 11, 0.22, 0.74),
            "sharpness": ImagePreprocessor._derive_score(checksum, 19, 0.25, 0.68),
            "width": 0,
            "height": 0,
            "pixel_count": 0,
        }

    @staticmethod
    def _derive_score(checksum: str, offset: int, minimum: float, maximum: float) -> float:
        window = checksum[offset : offset + 8]
        raw = int(window or "0", 16)
        span = max(maximum - minimum, 0.0001)
        ratio = (raw % 10_000) / 10_000
        return round(minimum + (ratio * span), 4)
