import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage, ImageOps
from pathlib import Path
import math
import cv2 as cv
import json

def prepareImage(image, scale=1, offset=0):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.unsignedinteger):
        image = image.astype(np.float64) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float64)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    return np.clip(image * scale + offset, 0, 1)


def save(image, path, scale=1, offset=0):
    image = prepareImage(image, scale, offset)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image, cmap="gray", vmin=0, vmax=1)

def get(path):
    with PILImage.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64) / 255.0

def gaussianBlur(I, stdev):
    return cv.GaussianBlur(
        I,
        (0, 0),
        sigmaX=stdev,
        sigmaY=stdev,
    )

def bilateral(I, sigma_s, sigma_r):
    """
    sigma_s: spatial Gaussian standard deviation, in pixels
    sigma_r: intensity Gaussian standard deviation
    """
    #* standard bilateral filtering is joint filtering with the image as its guide.
    return jointBilateral(I, I, sigma_s, sigma_r)

def jointBilateral(A, F, sigma_s, sigma_r):
    """
    A : ambient image
    F : flash image
    """
    A = np.asarray(A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)

    margin = min(0.01, sigma_r / 2)
    minimum = np.min(F) - margin
    maximum = np.max(F) + margin

    num_segments = math.ceil((maximum - minimum) / sigma_r)
    step = (maximum - minimum) / num_segments

    output = np.zeros(A.shape)
    for j in range (num_segments+1):
        reference = minimum + j * step

        #* use flash pixels for intensity similarity
        intensity_weights = np.exp(
            -(F - reference)**2 / (2 * sigma_r**2)
        )

        #* use ambient pixels to calculate the weighted image
        weighted_image = intensity_weights * A
        numerator = gaussianBlur(weighted_image,sigma_s)
        denominator = gaussianBlur(intensity_weights,sigma_s)
        filtered_at_reference = numerator / np.maximum(denominator, 1e-12)

        #* use interpolation based on flash
        interpolation_weights = np.maximum(0,1 - np.abs(F - reference) / step)
        output = output + (interpolation_weights * filtered_at_reference)
    
    return output

def detailTransfer(A, F, F_Base, A_NR, sigma_s, sigma_r, epsilon):
    return A_NR * (F + epsilon) / (F_Base + epsilon)

def shadowSpecularityMask(A_Base, A_Detail, M):
    if M.ndim == 2 and A_Base.ndim == 3:
        M = M[..., None]  # Share one luminance mask across all RGB channels.
    return (1 - M) * A_Detail + M * A_Base

def getISO(path):
    """Read ISO from the EXIF subdirectory used by the supplied lamp TIFFs."""
    with PILImage.open(path) as image:
        exif = image.getexif()
        values = dict(exif)
        if 34665 in exif:
            values.update(exif.get_ifd(34665))
        value = values.get(34855, values.get(34867))
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        return float(value) if value is not None else None


def makeShadowSpecularityMask(A, F, isoAmbient, isoFlash,
                              shadowThreshold=0.02, specularityThreshold=0.95,
                              morphologySize=3, featherSigma=2.0):
    """Return shadow, specularity, and feathered union masks (H x W).

    Uses linear luminance and equal exposure times, as in the assignment.
    Thresholds are starting values to tune; they are not selected optima.
    Reference: Petschnigg et al., section 4.3:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/Petschnigg-SG04.pdf
    """

    def linearLuminance(image):
        linear = np.where(image <= 0.0404482, image / 12.92,
                          ((image + 0.055) / 1.055) ** 2.4)
        return linear @ np.array([0.2126, 0.7152, 0.0722])

    ambient = linearLuminance(A) * (isoFlash / isoAmbient)
    flash = linearLuminance(F)
    shadow = (flash - ambient <= shadowThreshold).astype(np.uint8)
    specularity = (flash >= specularityThreshold).astype(np.uint8)

    def clean(mask):
        if morphologySize > 1:
            kernel = np.ones((morphologySize, morphologySize), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            mask = cv.dilate(mask, kernel)
        return mask.astype(np.float64)

    shadow = clean(shadow)
    specularity = clean(specularity)
    mask = np.maximum(shadow, specularity)
    if featherSigma > 0:
        mask = gaussianBlur(mask, featherSigma)
    return shadow, specularity, np.clip(mask, 0, 1)

def show(image, scale=1, offset=0):
    """Show an image or mask with a fixed [0, 1] intensity range.

    signed differences/derivatives, use offset=0.5 => zero is grey
    scale controls how much the differences are amplified
    """
    image = prepareImage(image, scale, offset)
    plt.figure()
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.show()

def showPair(image1, image2, titles=None, scale=1, offset=0):
    """Show two images side by side with the same intensity scale."""
    if titles is not None and len(titles) != 2:
        raise ValueError("Provide two titles, one for each image.")
    images = [prepareImage(im, scale, offset) for im in (image1, image2)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    fig.tight_layout()
    plt.show()


class Image:
    """One aligned flash/ambient pair and cached parameter experiments.

    Each sigma pair is used for all filters in that experiment. Detail and
    final outputs cover every (sigma_s, sigma_r, epsilon) combination.
    Caches last for this object's lifetime; construct a new object for new
    input images. Keep A and F unchanged after construction.
    """

    def __init__(self, pathBase, pathType, sigmas, epsilons, *,
                 outputDir=None, iso=None, maskOptions=None):
        self.pathType = pathType.lstrip(".")
        base = Path(pathBase).expanduser()
        # Accept the requested '/data/lamp/lamp' shorthand as project-relative.
        if base.is_absolute() and len(base.parts) > 1 and base.parts[1] == "data":
            base = Path(*base.parts[1:])
        if not base.is_absolute():
            script = Path(__file__).resolve().parent
            roots = (script, script.parent, Path.cwd(), Path.cwd().parent)
            candidates = [root / base for root in roots]
        else:
            candidates = [base]
        for candidate in candidates:
            paths = [Path(f"{candidate}_{kind}.{self.pathType}")
                     for kind in ("ambient", "flash")]
            if all(path.is_file() for path in paths):
                base = candidate.resolve()
                break
        else:
            raise FileNotFoundError(
                f"Could not find {base}_ambient.{self.pathType} and "
                f"{base}_flash.{self.pathType}. Put this file in the assignment's "
                "src folder, run from the assignment folder, or use an absolute prefix."
            )

        self.pathBase = str(base)
        self.name = base.name
        self.ambientPath = Path(f"{base}_ambient.{self.pathType}")
        self.flashPath = Path(f"{base}_flash.{self.pathType}")
        self.A = get(self.ambientPath)
        self.F = get(self.flashPath)
        if self.A.shape != self.F.shape:
            raise ValueError("Flash and ambient image dimensions must match")

        self.sigmas = list(dict.fromkeys(tuple(map(float, pair)) for pair in sigmas))
        self.epsilons = list(dict.fromkeys(float(e) for e in epsilons))

        root = next((p.parent for p in base.parents if p.name == "data"), base.parent)
        self.outputDir = Path(outputDir).expanduser() if outputDir is not None else root / "results"
        if not self.outputDir.is_absolute():
            self.outputDir = root / self.outputDir
        self.isoAmbient, self.isoFlash = (
            (getISO(self.ambientPath), getISO(self.flashPath)) if iso is None
            else tuple(float(v) for v in iso)
        )
        self.maskOptions = dict(shadowThreshold=0.02, specularityThreshold=0.95,
                                morphologySize=3, featherSigma=2.0)
        if maskOptions is not None:
            unknown = set(maskOptions) - set(self.maskOptions)
            if unknown:
                raise ValueError(f"Unknown mask options: {sorted(unknown)}")
            self.maskOptions.update(maskOptions)

        self.A_Base = {}
        self.F_Base = {}
        self.A_NR = {}
        self.A_Detail = {}
        self.A_Final = {}
        self.M = self.M_Shadow = self.M_Specularity = None
        self._maskKey = None
        self.saved = {}

    @staticmethod
    def _parameterName(value, compact=False):
        text = np.format_float_positional(float(value), trim="-")
        if compact and text.startswith("0."):
            return text[2:]  # 0.05 -> '05'; 0.0001 -> '0001'.
        # Distinguish integer range sigmas/epsilons from compact fractions.
        return text.replace(".", "p") + ("p0" if compact and "." not in text else "")

    def _suffix(self, key):
        suffix = (f"{self._parameterName(key[0])}-"
                  f"{self._parameterName(key[1], compact=True)}")
        if len(key) == 3:
            suffix += f"-eps-{self._parameterName(key[2], compact=True)}"
        return suffix

    def _saveImage(self, image, label, **record):
        path = self.outputDir / f"{self.name}-{label}.png"
        save(image, path)
        self.saved[path.name] = record
        return path

    def _saveResults(self, method, results):
        paths = []
        for key, result in results.items():
            paths.append(self._saveImage(
                result, f"{method}-{self._suffix(key)}", method=method,
                sigma_s=key[0], sigma_r=key[1],
                epsilon=key[2] if len(key) == 3 else None,
            ))
        self.saveParameters()
        return paths

    #* Step 1: basic bilateral filtering of the ambient image.
    def calculateBilateral(self):
        for key in self.sigmas:
            if key not in self.A_Base:
                print(f"{self.name}: bilateral, sigma_s={key[0]}, sigma_r={key[1]}", flush=True)
                self.A_Base[key] = bilateral(self.A, *key)
        return self.A_Base

    def saveBilateral(self):
        return self._saveResults("bilateral", self.calculateBilateral())

    #* Step 2: denoise the ambient image using the flash as the guide.
    def calculateJointBilateral(self):
        for key in self.sigmas:
            if key not in self.A_NR:
                print(f"{self.name}: joint bilateral, sigma_s={key[0]}, sigma_r={key[1]}", flush=True)
                self.A_NR[key] = jointBilateral(self.A, self.F, *key)
        return self.A_NR

    def saveJointBilateral(self):
        return self._saveResults("joint-bilateral", self.calculateJointBilateral())

    #* Step 3: reuse the joint result and one filtered flash per sigma pair.
    def calculateDetailTransfer(self):
        self.calculateJointBilateral()
        for sigma_s, sigma_r in self.sigmas:
            pair = (sigma_s, sigma_r)
            if pair not in self.F_Base:
                print(f"{self.name}: flash bilateral, sigma_s={sigma_s}, sigma_r={sigma_r}", flush=True)
                self.F_Base[pair] = bilateral(self.F, sigma_s, sigma_r)
            for epsilon in self.epsilons:
                key = (sigma_s, sigma_r, epsilon)
                if key not in self.A_Detail:
                    self.A_Detail[key] = detailTransfer(
                        self.A, self.F, self.F_Base[pair], self.A_NR[pair],
                        sigma_s, sigma_r, epsilon,
                    )
        return self.A_Detail

    def saveDetailTransfer(self):
        self.calculateDetailTransfer()
        self._saveResults("flash-base", self.F_Base)
        return self._saveResults("detail-transfer", self.A_Detail)

    #* Step 4: detect unreliable flash regions and blend with the ambient base.
    def calculateMask(self):
        if self.isoAmbient is None or self.isoFlash is None or min(self.isoAmbient, self.isoFlash) <= 0:
            raise ValueError("Valid ISO values are required for the shadow mask; pass iso=(ambient, flash)")
        key = (self.isoAmbient, self.isoFlash, tuple(sorted(self.maskOptions.items())))
        if self.M is None or key != self._maskKey:
            self.M_Shadow, self.M_Specularity, self.M = makeShadowSpecularityMask(
                self.A, self.F, self.isoAmbient, self.isoFlash, **self.maskOptions,
            )
            self._maskKey = key
            self.A_Final.clear()  # A changed mask invalidates only final blends.
        return self.M

    def calculateShadowSpecularityMask(self):
        self.calculateMask()
        self.calculateBilateral()
        self.calculateDetailTransfer()
        for key, detail in self.A_Detail.items():
            if key not in self.A_Final:
                self.A_Final[key] = shadowSpecularityMask(self.A_Base[key[:2]], detail, self.M)
        return self.A_Final

    def saveShadowSpecularityMask(self):
        self.calculateShadowSpecularityMask()
        for label, mask in (("shadow-mask", self.M_Shadow),
                            ("specularity-mask", self.M_Specularity), ("mask", self.M)):
            self._saveImage(mask, label, method=label)
        return self._saveResults("final", self.A_Final)

    def saveInputs(self):
        paths = [self._saveImage(self.A, "ambient", method="input"),
                 self._saveImage(self.F, "flash", method="input")]
        self.saveParameters()
        return paths

    def saveDifferences(self, scale=5):
        """Save signed RGB differences: zero is gray, display = 0.5 + scale * diff.

        Saves adjacent processing steps and each setting versus the first
        setting of the same method. PNGs clip to [0, 1]; the JSON records the
        display scale and numerical differences before display clipping.
        These differences measure changes, not accuracy against ground truth.
        """
        self.calculateShadowSpecularityMask()
        paths = []

        def write(left, right, label):
            difference = left - right
            paths.append(self._saveImage(
                prepareImage(difference, scale=scale, offset=0.5), f"diff-{label}",
                method="difference", scale=scale, offset=0.5,
                mean_absolute_difference=float(np.mean(np.abs(difference))),
                max_absolute_difference=float(np.max(np.abs(difference))),
            ))

        for key in self.sigmas:
            write(self.A_NR[key], self.A_Base[key], f"joint-minus-bilateral-{self._suffix(key)}")
        for key in self.A_Detail:
            write(self.A_Detail[key], self.A_NR[key[:2]], f"detail-minus-joint-{self._suffix(key)}")
            write(self.A_Final[key], self.A_Detail[key], f"final-minus-detail-{self._suffix(key)}")
        for method, results in (("bilateral", self.A_Base), ("joint-bilateral", self.A_NR),
                                ("detail-transfer", self.A_Detail), ("final", self.A_Final)):
            first = next(iter(results))
            for key in results:
                if key != first:
                    write(results[key], results[first],
                          f"{method}-{self._suffix(key)}-minus-{self._suffix(first)}")
        self.saveParameters()
        return paths

    def saveParameters(self):
        """Record exact parameters and filename meanings beside the PNG files."""
        self.outputDir.mkdir(parents=True, exist_ok=True)
        path = self.outputDir / f"{self.name}-parameters.json"
        metadata = {
            "ambient": str(self.ambientPath), "flash": str(self.flashPath),
            "sigmas": self.sigmas, "epsilons": self.epsilons,
            "iso_ambient": self.isoAmbient, "iso_flash": self.isoFlash,
            "mask_options": self.maskOptions,
            "mask_exposure_assumption": "Equal exposure times; linear ambient scaled by ISO_flash / ISO_ambient.",
            "mean_mask_weight": None if self.M is None else float(np.mean(self.M)),
            "image_export": "RGB floats clipped to [0, 1], then saved as PNG; no min/max rescaling.",
            "files": self.saved,
        }
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        return path

    def saveAll(self):
        self.saveInputs()
        self.saveBilateral()
        self.saveJointBilateral()
        self.saveDetailTransfer()
        self.saveShadowSpecularityMask()
        self.saveDifferences()
        print(f"Saved {len(self.saved)} images and parameters to {self.outputDir}", flush=True)
        return self.outputDir
