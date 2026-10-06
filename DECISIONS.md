# DECISIONS

Decision and tradeoff report for the SplitSnap preliminary experiments. Entries were written at the moment each decision was made, and the **Evidence** lines were filled in once the corresponding run finished. Numbers quoted here come from files in `results/`.

---

## D01. Separate git repository inside the project folder
- Decision: run `git init` inside `SplitSnap/` instead of committing to the enclosing `/Users/tominekan/Code` repo.
- Why: the enclosing repo has no commits and hundreds of unrelated staged files. The task says to commit after each task, and also never to touch anything outside the project. A nested repo satisfies both.
- Tradeoffs: the parent repo will see `CV/SplitSnap` as an embedded repo if you later `git add` it there. You may want to push this repo on its own, or remove `SplitSnap/.git` and commit from the parent.
- Alternatives considered: (a) commit in the parent repo, which would sweep in unrelated staged files; (b) skip commits, which breaks the task instructions.
- How to test it: n/a (housekeeping).
- Evidence: `git -C .. status` was not modified; `git log` in this folder shows one commit per task.

## D02. Install Tesseract locally inside the project (micromamba, conda-forge)
- Decision: Tesseract 5.5.3 from conda-forge, installed into `tools/tess/` with a project-local micromamba. `src/common.py` points pytesseract at `tools/tess/bin/tesseract` when it exists, else falls back to a `tesseract` on PATH.
- Why: the task assumed Tesseract was already installed, but it was not. `brew install` writes to `/opt/homebrew`, outside the project folder, which the constraints forbid.
- Tradeoffs: ~250 MB in `tools/` (gitignored). Teammates who already have a system Tesseract can ignore it. Version may differ from a teammate's Homebrew Tesseract, which can shift CER slightly.
- Alternatives considered: (a) Homebrew (simplest, but writes outside the folder); (b) building from source (slow, same issue for leptonica); (c) skipping OCR (impossible).
- How to test it: run `python src/evaluate.py` on a machine with system Tesseract and compare `results/summary.csv`.
- Evidence: `tools/tess/bin/tesseract --version` prints `tesseract 5.5.3`.
