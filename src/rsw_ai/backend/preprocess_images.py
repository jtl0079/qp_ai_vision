from pathlib import Path
import shutil

import cv2


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def preprocess_images(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for input_path in input_dir.rglob("*"):

        if not input_path.is_file():
            continue

        # Relative path from input_dir
        relative_path = input_path.relative_to(input_dir)

        # Corresponding output path
        output_path = output_dir / relative_path

        # Create parent directory
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================
        # Image
        # ====================================

        if input_path.suffix.lower() in IMAGE_EXTENSIONS:

            # Load image into memory
            image = cv2.imread(str(input_path))

            if image is None:
                continue

            # Contrast
            image = cv2.convertScaleAbs(
                image,
                alpha=1.5,
                beta=0,
            )

            
            # Gaussian Denoising
            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0,
            )
            

            # Export
            cv2.imwrite(
                str(output_path),
                image,
            )

            # Release image from memory
            del image

        # ====================================
        # Non-image
        # ====================================

        else:
            shutil.copy2(
                input_path,
                output_path,
            )