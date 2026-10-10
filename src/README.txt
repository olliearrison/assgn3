Assignment 3: Parts 2 and 3

Run from the assignment directory:

python3 src/main.py

Code layout:
src/gradient_domain.py: reusable operators, Algorithm 1 CG, fusion equations.
src/experiments.py: registration, comparisons, exports, Part 2 and Part 3 runs.
src/image.py: existing bilateral pipeline, image loading and exports.
src/main.py: command-line entry point; importing it no longer launches a run.
src/test_gradient_domain.py: independent numerical validation.