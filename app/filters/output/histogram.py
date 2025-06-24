import numpy as np
from filters.filter import Filter
from logging import Logger

class HistogramFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name=__name__, logger=logger)

    def apply(self, data: dict):
        features = []
        histograms = []
        for deep_vec, morph_grad, magnitudes, directions in zip(
            data["fingerprint"], data['gradients'], data["magnitudes"], data["directions"]
        ):
            # Morphological gradient features
            hist_grad, _ = np.histogram(morph_grad, bins=32, range=(0, 255), density=True)
            mean_grad = morph_grad.mean()
            std_grad = morph_grad.std()

            # Magnitude features
            hist_mag, _ = np.histogram(magnitudes, bins=32, range=(0, 255), density=True)
            mean_mag = magnitudes.mean()
            std_mag = magnitudes.std()

            # Direction features (assuming directions in degrees 0-360)
            hist_dir, _ = np.histogram(directions, bins=32, range=(0, 360), density=True)
            mean_dir = directions.mean()
            std_dir = directions.std()

            # Combine all handcrafted features
            handcrafted = np.concatenate([
                hist_grad, [mean_grad, std_grad],
                hist_mag, [mean_mag, std_mag],
                hist_dir, [mean_dir, std_dir]
            ])
            # Combine with deep features
            combined = np.concatenate([deep_vec, handcrafted])
            self.logger.info(f"Combined feature vector length: {len(combined)} {combined}")
            features.append(combined)

        data["final_features"] = features
        return data