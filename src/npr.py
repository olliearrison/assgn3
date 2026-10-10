"""Part 5: compact GradientShop-inspired still-image NPR (paper sections 3.2/6.2)."""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from gradient_domain import gradient, divergence, boundaryMask


def longEdges(I, edge_scale=90):
    """Approximate edge length with Canny contour arc length; share it across RGB."""
    gray = cv.GaussianBlur(I @ np.array([.2126, .7152, .0722]), (0, 0), 2.0)
    lo, hi = np.percentile(gray, [1, 99])
    detection = np.uint8(np.clip((gray-lo)/max(hi-lo, 1e-8), 0, 1)*255)
    edges = cv.Canny(detection, 25, 65, L2gradient=True)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    lengths = np.zeros(gray.shape, np.float32)
    for contour in sorted(contours, key=lambda c: cv.arcLength(c, False)):
        cv.drawContours(lengths, [contour], -1, float(cv.arcLength(contour, False)), 1)
    saliency = 1 - np.exp(-lengths**2 / (2*edge_scale**2))
    support = cv.GaussianBlur(cv.dilate(saliency, np.ones((5, 5), np.uint8)), (0, 0), 1)
    # Squared normal components implement the paper's cos^2(theta), sin^2(theta).
    normal = np.stack((cv.Sobel(gray, cv.CV_64F, 1, 0), cv.Sobel(gray, cv.CV_64F, 0, 1)), -1)
    orientation = normal**2 / np.maximum(np.sum(normal**2, axis=-1, keepdims=True), 1e-12)
    return support, orientation, saliency


def integrate(I, G, weights, data_weight=.019, tolerance=1e-6, max_iterations=2000):
    """CG for lambda*(f-I)^2 + sum(weights*(grad(f)-G)^2), fixed input boundary."""
    B = boundaryMask(I.shape)
    channels, diagnostics = [], []
    for c in range(I.shape[-1]):
        original = np.ascontiguousarray(I[..., c])
        w = np.ascontiguousarray(weights[..., c, :])
        def operator(x):
            return data_weight*x - divergence(w*gradient(x))
        rhs = data_weight*original - divergence(w*G[..., c, :])
        x = original.copy()
        r = B*(rhs-operator(x))
        d = r.copy()
        delta = float(np.sum(r*r))
        n = 0
        while np.sqrt(delta) > tolerance and n < max_iterations:
            q = B*operator(d)
            denominator = float(np.sum(d*q))
            if not np.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError('NPR CG operator is not positive definite')
            alpha = delta / denominator
            x += alpha*d
            r -= alpha*q
            next_delta = float(np.sum(r*r))
            d = r + (next_delta/delta)*d
            delta = next_delta
            n += 1
        residual = float(np.linalg.norm(B*(rhs-operator(x))))
        if not np.isfinite(x).all() or residual > tolerance*1.001:
            raise RuntimeError(f'NPR solve did not converge: residual={residual}')
        channels.append(x)
        diagnostics.append(dict(iterations=n, residual=residual,
                                boundary_max_error=float(np.max(np.abs((1-B)*(x-original))))))
    return np.stack(channels, -1), diagnostics


def stylize(I, edge_scale=90, boost=2.5, data_weight=.019, outline=.25):
    I = np.asarray(I, dtype=np.float64)
    if I.ndim != 3 or I.shape[-1] != 3 or min(I.shape[:2]) < 3 or not np.isfinite(I).all():
        raise ValueError('Expected a finite H x W x 3 RGB image, at least 3 x 3')
    if edge_scale <= 0 or boost < 1 or data_weight <= 0 or not 0 <= outline <= 1:
        raise ValueError('Require edge_scale>0, boost>=1, data_weight>0, 0<=outline<=1')
    support, orientation, edges = longEdges(I, edge_scale)
    source = gradient(I)
    target = source * (boost*support[..., None]*orientation)[..., None, :]
    # GradientShop's robust gradient weights, b=9; retain the input data term.
    weights = (1 + np.abs(source-target))**-9
    reconstructed, info = integrate(I, target, weights, data_weight)
    ink = cv.GaussianBlur(edges, (0, 0), .6)*boundaryMask(I.shape)
    result = reconstructed*(1-outline*ink[..., None])
    return result, reconstructed, support, info


def save(array, path):
    Image.fromarray(np.uint8(np.clip(array, 0, 1)*255 + .5)).save(path)


def comparison(panels, path, columns=2):
    width, height = 720, 520
    sheet = Image.new('RGB', (columns*width, ((len(panels)+columns-1)//columns)*height), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 27)
    except OSError:
        font = ImageFont.load_default()
    for n, (label, array) in enumerate(panels):
        x, y = (n % columns)*width, (n // columns)*height
        draw.text((x+15, y+10), label, fill='black', font=font)
        im = Image.fromarray(np.uint8(np.clip(array, 0, 1)*255+.5))
        im.thumbnail((width-30, height-60), Image.Resampling.LANCZOS)
        sheet.paste(im, (x+(width-im.width)//2, y+52+(height-60-im.height)//2))
    sheet.save(path)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', default=[root/'data/mine'/f'{s}_flash.JPG' for s in ('potion', 'lemon', 'cat')])
    parser.add_argument('--output-dir', type=Path, default=root/'results/part5')
    parser.add_argument('--max-side', type=int, default=960)
    parser.add_argument('--edge-scale', type=float, default=90)
    parser.add_argument('--boost', type=float, default=2.5)
    parser.add_argument('--data-weight', type=float, default=.019)
    parser.add_argument('--outline', type=float, default=.25)
    args = parser.parse_args()
    if args.max_side < 16:
        parser.error('--max-side must be at least 16')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    panels, records = [], []
    for path in args.inputs:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            original_size = image.size
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            I = np.asarray(image, dtype=np.float64)/255
        print(f'Stylizing {path.name} at {I.shape[1]} x {I.shape[0]}', flush=True)
        result, reconstructed, support, info = stylize(I, args.edge_scale, args.boost, args.data_weight, args.outline)
        folder = out/path.stem
        folder.mkdir(parents=True, exist_ok=True)
        for label, array in [('input', I), ('stylized', result), ('reconstructed', reconstructed), ('saliency', support)]:
            save(array, folder/f'{label}.png')
        pair = [(f'{path.stem}: input', I), (f'{path.stem}: NPR', result)]
        comparison(pair, folder/'comparison.png')
        panels.extend(pair)
        records.append(dict(input=str(path.resolve()), original_size=list(original_size), output_shape=list(I.shape), solver=info,
                            fraction_clipped=float(np.mean((result<0)|(result>1))), mean_absolute_change=float(np.mean(abs(result-I)))))
    comparison(panels, out/'comparison.png')
    metadata = dict(method='GradientShop 6.2 with Canny contour-length approximation; weighted screened-Poisson CG',
                    reference='https://grail.cs.washington.edu/projects/gradientshop/demos/gs_paper_TOG_2009.pdf',
                    parameters=dict(edge_scale=args.edge_scale, boost=args.boost, data_weight=args.data_weight, outline=args.outline,
                                    robust_exponent=9, canny_thresholds=[25,65], guide_blur_sigma=2.0, max_side=args.max_side,
                                    cg_tolerance=1e-6, cg_max_iterations=2000), runs=records)
    (out/'parameters.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {len(records)} examples to {out}', flush=True)


if __name__ == '__main__':
    main()
