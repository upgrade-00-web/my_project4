"""Эксперимент: сравнение трёх наборов параметров."""
from pathlib import Path
import csv
from io_utils import generate_samples, list_images, load_image
from processing import run_pipeline
from metrics import calculate_metrics

def main():
    input_dir = Path("images")
    output_dir = Path("results/experiment")
    if not input_dir.exists() or not any(input_dir.iterdir()):
        generate_samples(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        {"threshold": 35, "kernel_size": 3, "min_area": 8, "blur_size": 3},
        {"threshold": 45, "kernel_size": 3, "min_area": 12, "blur_size": 3},
        {"threshold": 60, "kernel_size": 5, "min_area": 18, "blur_size": 5},
    ]
    rows = []
    for path in list_images(input_dir):
        image = load_image(path)
        for number, params in enumerate(settings, 1):
            result = run_pipeline(image, **params)
            mse, psnr, ssim = calculate_metrics(image, result.annotated)
            rows.append({
                "filename": path.name, "variant": number, **params,
                "scratch_count": len(result.scratches),
                "lengths_px": ";".join(f"{s.length:.1f}" for s in result.scratches),
                "mse": round(mse, 3), "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4)
            })
            if number == 2:
                import cv2
                cv2.imwrite(str(output_dir / f"{path.stem}_result.png"), result.annotated)
    with (output_dir / "experiment.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Эксперимент завершён. Таблица: results/experiment/experiment.csv")

if __name__ == "__main__":
    main()
