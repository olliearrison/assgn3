"""Part 2/3 experiments and image/JSON exports."""
import json
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image as PILImage, ImageDraw
from image import Image, get, save
from gradient_domain import (gradient, divergence, laplacian, boundaryMask,
    gradientFieldIntegration, fusePair, LAPLACIAN_BACKEND, validateDivGradPrincipals)


def writeJSON(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resize(I, max_size):
    if not max_size or max(I.shape[:2]) <= max_size:
        return I.copy()
    scale = max_size / max(I.shape[:2])
    return cv.resize(I, (round(I.shape[1]*scale), round(I.shape[0]*scale)), interpolation=cv.INTER_AREA)


def contactSheet(panels, path, columns=3, width=420):
    rows = (len(panels)+columns-1)//columns
    height = 320
    sheet = PILImage.new("RGB", (columns*width, rows*height), "#f4f4f2")
    draw = ImageDraw.Draw(sheet)
    for i, (title, array) in enumerate(panels):
        a = np.asarray(array)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        im = PILImage.fromarray(np.uint8(np.clip(a, 0, 1)*255 + 0.5))
        im.thumbnail((width-16, height-42))
        x, y = (i % columns)*width, (i//columns)*height
        draw.text((x+8, y+7), title, fill="#172536")
        sheet.paste(im, (x+(width-im.width)//2, y+32+(height-42-im.height)//2))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def alignPair(A, F, out):
    """Estimate a global affine registration; preserve originals separately.

    ECC uses CLAHE grayscale previews for lighting robustness. Warp maps ambient
    coordinates into the flash image. Crop a common valid rectangle after warp.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if A.shape != F.shape:
        raise ValueError("Input pair dimensions do not match")
    a, f = resize(A, 640), resize(F, 640)
    def guide(I):
        gray = cv.cvtColor(np.uint8(np.clip(I,0,1)*255), cv.COLOR_RGB2GRAY)
        return cv.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(gray).astype(np.float32)/255
    ga, gf = guide(a), guide(f)
    warp = np.eye(2,3,dtype=np.float32)
    try:
        score, warp = cv.findTransformECC(ga, gf, warp, cv.MOTION_AFFINE,
                     (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 400, 1e-7), None, 5)
    except cv.error as exc:
        raise RuntimeError("Registration did not converge; inspect the pair before processing") from exc
    if not np.isfinite(warp).all() or np.linalg.det(warp[:,:2]) <= 0:
        raise RuntimeError("Registration returned an invalid transform")
    # Account for x/y resize rounding separately using homogeneous coordinates.
    S = np.diag([a.shape[1]/A.shape[1], a.shape[0]/A.shape[0], 1.])
    W = np.eye(3); W[:2] = warp
    W = np.linalg.inv(S) @ W @ S
    shape = (A.shape[1], A.shape[0])
    aligned = cv.warpAffine(F, W[:2], shape, flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                            borderMode=cv.BORDER_CONSTANT)
    valid = cv.warpAffine(np.ones(A.shape[:2], np.float32), W[:2], shape,
                         flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_CONSTANT) > 0.999
    low, high = 0, min(A.shape[:2])//4
    if not valid[high:-high, high:-high].all():
        raise RuntimeError("Registration would discard too much of the image")
    while low < high:
        mid = (low+high)//2
        region = valid if mid == 0 else valid[mid:-mid, mid:-mid]
        if region.all(): high = mid
        else: low = mid+1
    margin = low
    crop = (slice(margin,-margin), slice(margin,-margin)) if margin else (slice(None), slice(None))
    ac, fc = A[crop].copy(), aligned[crop].copy()
    warped_guide = cv.warpAffine(gf, warp, (ga.shape[1],ga.shape[0]),
                                flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    def corr(x,y): return float(np.corrcoef(x.ravel(),y.ravel())[0,1])
    info = dict(method="affine ECC on CLAHE grayscale previews", ecc_score=float(score),
                transform_ambient_to_flash=W.tolist(), crop_border_pixels=margin,
                original_shape=list(A.shape), aligned_shape=list(ac.shape),
                guide_correlation_before=corr(ga[20:-20,20:-20],gf[20:-20,20:-20]), guide_correlation_after=corr(ga[20:-20,20:-20],warped_guide[20:-20,20:-20]))
    save(ac,out/"ambient-aligned.png"); save(fc,out/"flash-aligned.png")
    # Red/cyan overlays emphasize registration of edges without hiding originals.
    def overlay(x,y):
        x,y=resize(x,640),resize(y,640)
        gx,gy=guide(x),guide(y)
        return np.stack([gx,gy,gy],axis=-1)
    contactSheet([("Before: red ambient / cyan flash",overlay(A,F)),
                  ("After: red ambient / cyan flash",overlay(ac,fc))], out/"alignment-comparison.png",columns=2)
    writeJSON(out/"alignment.json",info)
    return ac,fc,info


def loadPair(data, name, extension, out=None, align=False):
    A,F = [get(Path(data)/f"{name}_{kind}.{extension}") for kind in ("ambient","flash")]
    if A.shape != F.shape:
        raise ValueError("Pair dimensions differ")
    info = dict(method="provided aligned pair",shape=list(A.shape))
    if align:
        return alignPair(A,F,out)
    return A,F,info


def solveRecord(A,F,out,sigma,tau,boundary="average",initialization="average",epsilon=1e-5,N=4000):
    result, record, fields = fusePair(A,F,sigma,tau,boundary,initialization,epsilon,N)
    record["laplacian_backend"] = LAPLACIAN_BACKEND
    record["fraction_below_zero"] = float(np.mean(result < 0))
    record["fraction_above_one"] = float(np.mean(result > 1))
    if not all(c["converged"] for c in record["solver"]):
        raise RuntimeError(f"Fusion did not converge: {out}")
    save(result,Path(str(out)+".png"))
    writeJSON(Path(str(out)+".json"),record)
    return result,record,fields


def reconstruction(I,out,preview_size=320,full=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    work = I if full else resize(I,preview_size)
    B=boundaryMask(work.shape);D=divergence(gradient(work))
    settings = [(1e-6,8000)] if full else [(1e-6,25),(1e-6,250),(1e-3,4000),(1e-5,4000),(1e-7,4000)]
    records=[];panels=[("Original",work)]
    for eps,N in settings:
        label=f"eps-{eps:g}-N-{N}"
        print(f"Reconstruction {label} {work.shape}",flush=True)
        restored,info=gradientFieldIntegration(D,np.zeros_like(work),B,work,eps,N,True)
        error=restored-work
        record=dict(epsilon=eps,N=N,shape=list(work.shape),rmse=float(np.sqrt(np.mean(error**2))),
                    max_absolute_error=float(np.max(np.abs(error))),solver=info,
                    interior_operator_identity=validateDivGradPrincipals(work),laplacian_backend=LAPLACIAN_BACKEND)
        records.append(record)
        save(restored,out/f"reconstructed-{label}.png")
        save(error,out/f"error-{label}.png",scale=20,offset=.5)
        writeJSON(out/f"{label}.json",record)
        panels.append((f"{label}; RMSE {record['rmse']:.2e}",restored))
        if full:
            if not all(c["converged"] for c in info): raise RuntimeError("Full reconstruction failed to converge")
            np.savez_compressed(out/"reconstruction-raw.npz",original=work,reconstructed=restored,error=error)
            panels.append(("Error: gray = 0; gain 20",.5+20*error))
    contactSheet(panels,out/"comparison.png")
    writeJSON(out/"summary.json",records)
    return records


def fusionPreviews(A,F,out,preview_size=320):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    A,F=resize(A,preview_size),resize(F,preview_size)
    records=[];grid=[];baseline=None
    for sigma in [20,40,80]:
        for tau in [.8,.9,.95]:
            label=f"sigma-{sigma}-tau-{tau:g}"
            print(f"Fusion preview {out.name}: {label}",flush=True)
            result,record,_=solveRecord(A,F,out/label,sigma,tau)
            grid.append((label,result));records.append(record)
            if sigma==40 and tau==.9: baseline=result
    contactSheet(grid,out/"parameter-comparison.png")
    boundary_panels=[]
    for boundary in ["ambient","flash","average"]:
        result,record,_=solveRecord(A,F,out/f"boundary-{boundary}",40,.9,boundary=boundary)
        boundary_panels.append((f"Boundary: {boundary}",result));records.append(record)
    contactSheet(boundary_panels,out/"boundary-comparison.png")
    init_panels=[]
    for init in ["zero","ambient","flash","average"]:
        result,record,_=solveRecord(A,F,out/f"init-{init}",40,.9,initialization=init)
        record["max_difference_from_average_init"]=float(np.max(np.abs(result-baseline)))
        writeJSON(out/f"init-{init}.json",record)
        init_panels.append((f"Init {init}; iterations " + "/".join(str(c["iterations"]) for c in record["solver"]),result))
        records.append(record)
    contactSheet(init_panels,out/"initialization-comparison.png",columns=2)
    writeJSON(out/"summary.json",records)


def finalFusion(A,F,out,selection):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    print(f"Full fusion {out}: {A.shape}",flush=True)
    result,record,(G,M,ws)=solveRecord(A,F,out/"fused",**selection,epsilon=1e-6,N=10000)
    gA,gF=gradient(A),gradient(F)
    # One fixed gain across all fields and both axes. Exclude boundary padding.
    values=np.concatenate([np.abs(g[1:-1,1:-1]).ravel() for g in [gA,gF,G]])
    gain=.45/max(float(np.percentile(values,99)),1e-6)
    panels=[]
    for name,g in [("ambient",gA),("flash",gF),("fused",G)]:
        for axis in range(2):
            save(g[...,axis],out/f"gradient-{name}-{'x' if axis==0 else 'y'}.png",scale=gain,offset=.5)
            panels.append((f"{name}: {'x' if axis==0 else 'y'}",.5+gain*g[...,axis]))
    save(A,out/"ambient.png");save(F,out/"flash.png")
    save(M,out/"coherency.png");save(ws,out/"saturation.png")
    save(result-A,out/"fused-minus-ambient.png",scale=2,offset=.5)
    save(result-F,out/"fused-minus-flash.png",scale=2,offset=.5)
    contactSheet(panels,out/"gradient-comparison.png")
    contactSheet([("Ambient",A),("Flash",F),("Gradient fusion",result)],out/"comparison.png")
    record.update(gradient_display_gain=gain,gradient_display_offset=.5,
                  saturation_normalization="(tanh(sigma*(F-tau_s))+1)/2",
                  undefined_coherency="0 where product of gradient norms <= 1e-12")
    writeJSON(out/"fused.json",record)
    np.savez_compressed(out/"fields-raw.npz",gradient_ambient=gA.astype(np.float32),
                        gradient_flash=gF.astype(np.float32),gradient_fused=G.astype(np.float32),
                        coherency=M.astype(np.float32),saturation=ws.astype(np.float32),
                        fused=result,divergence=divergence(G))
    return record


def bilateralExperiments(data,out,selection,mask_options,name="potion"):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pair=Image(str(Path(data)/f"mine/{name}"),"JPG",[(4,.1),(8,.1),(8,.2)], [.0001,.001],
               outputDir=out,maskOptions=mask_options)
    pair.A,pair.F,alignment=alignPair(pair.A,pair.F,out/"alignment")
    pair.saveAll()
    key=tuple(selection)
    threshold_panels=[]; threshold_records=[]
    selected_threshold=pair.maskOptions["shadowThreshold"]
    for threshold in [0.,.003,.005,.02]:
        pair.maskOptions["shadowThreshold"]=threshold
        final=pair.calculateShadowSpecularityMask()[key]
        save(final,out/f"mask-threshold-{threshold:g}-final.png")
        save(pair.M,out/f"mask-threshold-{threshold:g}-mask.png")
        threshold_panels.extend([(f"Threshold {threshold:g}: final",final.copy()),
                                 (f"Threshold {threshold:g}: mask",pair.M.copy())])
        threshold_records.append(dict(shadowThreshold=threshold,mean_mask_weight=float(pair.M.mean())))
    pair.maskOptions["shadowThreshold"]=selected_threshold
    pair.calculateShadowSpecularityMask()
    pair.saveParameters()
    contactSheet(threshold_panels,out/"mask-threshold-comparison.png",columns=2)
    writeJSON(out/"mask-thresholds.json",threshold_records)
    save(pair.A_Final[key],out/"selected-denoised.png")
    np.savez_compressed(out/"selected-denoised-raw.npz",image=pair.A_Final[key])
    panels=[("Ambient (aligned crop)",pair.A),("Flash (aligned crop)",pair.F)]
    for params,result in pair.A_Final.items():
        panels.append((f"Final: s={params[0]:g}, r={params[1]:g}, eps={params[2]:g}",result))
    contactSheet(panels,out/"parameter-comparison.png",columns=2,width=560)
    stages=[("Ambient",pair.A),("Bilateral",pair.A_Base[key[:2]]),
            ("Joint bilateral",pair.A_NR[key[:2]]),("Detail transfer",pair.A_Detail[key]),
            ("Masked final",pair.A_Final[key]),("Mask: white uses ambient base",pair.M)]
    contactSheet(stages,out/"stages-comparison.png")
    writeJSON(out/"selection.json",dict(sigma_s=key[0],sigma_r=key[1],epsilon=key[2],
              alignment=alignment,shape=list(pair.A.shape),iso_ambient=pair.isoAmbient,iso_flash=pair.isoFlash,
              mask_options=pair.maskOptions,mean_mask_weight=float(pair.M.mean())))


def part2(data,results,stage,config,preview_size):
    A,F,_=loadPair(Path(data)/"museum","museum","png")
    root=Path(results)/"part2"
    if stage in ("preview","all"):
        reconstruction(A,root/"reconstruction-preview",preview_size)
        fusionPreviews(A,F,root/"fusion-preview",preview_size)
    if stage in ("final","all"):
        reconstruction(A,root/"reconstruction-full",full=True)
        finalFusion(A,F,root/"fusion-full",config["museum"])


def part3(data,results,stage,config,preview_size,scenes=("potion","lemon","cat")):
    root=Path(results)/"part3"
    if "potion" in scenes:
        bilateralExperiments(data,root/"potion-bilateral",config["potion"],config["potion_mask"])
    if "cat" in scenes:
        bilateralExperiments(data,root/"cat-bilateral",config["cat_bilateral"],config["cat_mask"],name="cat")
    for name in ("lemon","cat"):
        if name not in scenes:
            continue
        A,F,_=loadPair(Path(data)/"mine",name,"JPG",root/f"{name}-alignment",align=True)
        if stage in ("preview","all"):
            fusionPreviews(A,F,root/f"{name}-fusion-preview",preview_size)
        if stage in ("final","all"):
            finalFusion(A,F,root/f"{name}-fusion-full",config[name])
