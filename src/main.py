"""Assignment 3 entry point. Defaults to the remaining Parts 2 and 3."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
from image import Image
from gradient_domain import (gradient, laplacian, divergence,
    validateDivGradPrincipals, gradientFieldIntegration, fusedGradient, LAPLACIAN_BACKEND)
from experiments import part2, part3

ROOT = Path(__file__).resolve().parents[1]


def part1(data, results):
    # Preserve the original Part 1 entry point; it runs only when requested.
    lamp = Image(str(Path(data)/"lamp/lamp"), "tif", [(4,.10)], [.0001,.001],
                 outputDir=Path(results)/"part1")
    lamp.saveAll()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts",type=int,nargs="+",choices=[1,2,3],default=[2,3])
    parser.add_argument("--stage",choices=["preview","final","all"],default="all")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--config",type=Path,default=ROOT/"experiments.json")
    parser.add_argument("--preview-size",type=int,default=320)
    parser.add_argument("--personal-scenes",nargs="+",choices=["potion","lemon","cat"],
                        default=["potion","lemon","cat"])
    args=parser.parse_args()
    if args.preview_size < 16: parser.error("--preview-size must be at least 16")
    args.data_dir=args.data_dir.resolve()
    args.output_dir=args.output_dir.resolve()
    config=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print("Laplacian:",LAPLACIAN_BACKEND,flush=True)
    for part in args.parts:
        if part==1: part1(args.data_dir,args.output_dir)
        elif part==2:
            part2(args.data_dir,args.output_dir,args.stage,config,args.preview_size)
        else:
            part3(args.data_dir,args.output_dir,args.stage,config,args.preview_size,args.personal_scenes)
    print(f"Completed. Results: {args.output_dir.resolve()}",flush=True)


if __name__=="__main__":
    main()
