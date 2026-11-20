#!/usr/bin/env python3
"""Set up and run BERNN + scANVI grouped three-fold CV from source data.

Source code and saved figure snapshots are never overwritten. Use --prepare
first. Each method/dataset/fold is trained freshly, with label-disjoint outer
test groups. scANVI sees unlabeled target expression (transductive); BERNN is
inductive. This distinction is recorded in the results.
"""
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parents[2]
SOURCE_URL="https://github.com/spell00/task_batch_integration_clean.git"
SOURCE_REV="9cc7960d89d409d76715449e8f8a4f4860a5dfe7"
SETS=("dkd","gtex_v9","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens")
CODE=HERE/"training/experiment_code"
MANIFESTS=HERE/"training/cv/manifests"
EXP_REL=Path("experiments/single_cell_paper_20261019")

def run(args,env=None,stdout=None):
    print("$"," ".join(map(str,args)),flush=True)
    result=subprocess.run(list(map(str,args)),env=env,stdout=stdout,
                          stderr=subprocess.STDOUT if stdout is not None else None)
    if result.returncode:raise RuntimeError(f"Command failed (exit {result.returncode})")

def prepare(workspace,source_data):
    assert not workspace.is_symlink()
    if not (workspace/".git").exists():
        if workspace.exists() and any(workspace.iterdir()):
            raise RuntimeError("Refusing nonempty non-Git destination: "+str(workspace))
        run(["git","clone",SOURCE_URL,workspace])
    rev=subprocess.check_output(["git","-C",str(workspace),"rev-parse","HEAD"],text=True).strip()
    if rev!=SOURCE_REV:run(["git","-C",workspace,"checkout",SOURCE_REV])
    exp=workspace/EXP_REL; folder=exp/"code";folder.mkdir(parents=True,exist_ok=True)
    for src in CODE.rglob("*"):
        if src.is_file():
            dest=folder/src.relative_to(CODE);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)
    # Host-side wrapper paths must point to this isolated workspace, not vm2's
    # original source checkout. The scientific method code is unchanged.
    f=folder/"comparators.py";s=f.read_text()
    old='ROOT=Path("/home/simonp/task_batch_integration_bernn");EXP=ROOT/"experiments/single_cell_paper_20261019"'
    new='ROOT=Path(os.environ["BERNN_CV_WORKSPACE"]);EXP=ROOT/"experiments/single_cell_paper_20261019"'
    if old not in s and new not in s:raise RuntimeError("Unrecognized comparator source")
    f.write_text(s.replace(old,new))
    # The dataset symlink resolves outside the repository bind mount. Expose
    # the source data inside every comparator container as read-only.
    patched=f.read_text()
    before='cmd+=["-e","PYTHONUNBUFFERED=1","-v",str(ROOT)+":/repo","-w",'
    after='cmd+=["-e","PYTHONUNBUFFERED=1","-v",str(ROOT)+":/repo","-v",os.environ["BERNN_CV_DATA_ROOT"]+":/repo/resources_full/task_batch_integration/datasets/cellxgene_census:ro","-w",'
    if before not in patched:raise RuntimeError("Comparator Docker mount contract changed")
    f.write_text(patched.replace(before,after))
    manifests=exp/"cv_manifests";manifests.mkdir(exist_ok=True)
    for item in MANIFESTS.glob("*.json"):shutil.copy2(item,manifests/item.name)
    # Files stay outside the publication Git clone (symlinked, not copied).
    target=workspace/"resources_full/task_batch_integration/datasets/cellxgene_census"
    if not source_data.is_dir():raise FileNotFoundError(source_data)
    if target.exists() or target.is_symlink():
        if target.resolve()!=source_data.resolve():
            raise RuntimeError("Dataset path points to another source: "+str(target))
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(source_data.resolve(),target_is_directory=True)
    for ds in SETS:
        inp=target/ds/"log_cp10k/dataset.h5ad"
        sol=target/ds/"log_cp10k/solution.h5ad"
        if not inp.is_file() or not sol.is_file():raise FileNotFoundError((inp,sol))
    shutil.copy2(HERE/"training/cv/fixed_hpo_config.json",exp/"cv_hpo_config.json")
    provenance={"source_commit":SOURCE_REV,"code_snapshot":"Oct 19 research experimental scripts",
      "outer_cv":"Three disjoint batch folds, complete test coverage; nested separate validation batches",
      "methods":{"bernn":"Inductive: train-batch expression and labels; validation for selection",
                 "scanvi":"Transductive: full dataset expression, labels masked outside training batches"},
      "source_dataset_dir":str(source_data.resolve())}
    (exp/"cv_provenance.json").write_text(json.dumps(provenance,indent=2))
    print("Prepared",workspace,"with frozen code, manifests and read-only dataset links")

def fit(workspace,ds,fold,method,bernn_image,scanvi_image,data_dir):
    exp=workspace/EXP_REL
    if not (exp/"cv_manifests"/f"{ds}_fold{fold}.json").exists():
        raise RuntimeError("Run --prepare first")
    seed=42+fold
    b=exp/"bernn"/str(seed)/ds
    if method=="bernn":
        if (b/"scores.json").exists():
            print("Already trained:",b);return
        name=f"bernn-cv-{ds}-{fold}".replace("_","-")
        logs=exp/"cv_logs";logs.mkdir(exist_ok=True)
        cmd=["docker","run","--rm","--gpus","all","--ipc=host",
             "--name",name,"-e","PYTHONUNBUFFERED=1",
             "-e","CUDA_VISIBLE_DEVICES=0",
             "-v",f"{workspace}:/repo",
             "-v",f"{data_dir}:/repo/resources_full/task_batch_integration/datasets/cellxgene_census:ro",
             "-w",f"/repo/{EXP_REL}",
             "--entrypoint","python",bernn_image,
             "code/run_dataset_cv3_transductive.py","--dataset",ds,"--fold",fold,
             "--seed",seed,"--out",f"/repo/{EXP_REL}/bernn/{seed}/{ds}",
             "--hpo-config",f"/repo/{EXP_REL}/cv_hpo_config.json",
             "--warmup",50,"--epochs",200]
        # Not a timeout; long experiments should be allowed to complete.
        with (logs/f"bernn_{ds}_fold{fold}.log").open("w") as f:run(cmd,stdout=f)
        if not (b/"scores.json").exists():raise RuntimeError("BERNN score missing after training")
        return
    if method=="scanvi":
        if not (b/"scores.json").exists():
            raise RuntimeError("Fit BERNN split first so scANVI uses identical train/valid/test batches")
        dst=exp/"comparators"/str(seed)/"scanvi"/ds
        if (dst/"scores.json").exists():
            print("Already trained:",dst);return
        if not scanvi_image:raise RuntimeError("--scanvi-image is required; pin a tested image")
        env=os.environ.copy()
        env.update(BERNN_CV_WORKSPACE=str(workspace),
                   SC_PAPER_SCVI_IMAGE=scanvi_image,SC_PAPER_GPU_DEVICE="0",
                   BERNN_CV_DATA_ROOT=str(data_dir))
        subprocess.check_call([sys.executable,"-c",
           "import comparators;comparators.fit('scanvi',"+repr(ds)+",True,seed="+str(seed)+")"],
           cwd=exp/"code",env=env)
        if not (dst/"scores.json").exists():raise RuntimeError("scANVI score absent; inspect fit.log")
        return
    raise ValueError(method)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--run",action="store_true")
    p.add_argument("--workspace",type=Path,default=Path.home()/"bernn_single_cell_cv_workspace")
    p.add_argument("--data-dir",type=Path,default=Path.home()/"task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census")
    p.add_argument("--method",choices=("bernn","scanvi","both"),default="both")
    p.add_argument("--dataset",choices=(*SETS,"all"),default="dkd")
    p.add_argument("--fold",choices=("0","1","2","all"),default="0")
    p.add_argument("--bernn-image",default="bernn-openproblems-vm2:cu121-fast-644b5a9")
    p.add_argument("--scanvi-image",default=None)
    a=p.parse_args()
    if a.prepare:prepare(a.workspace.resolve(),a.data_dir.resolve())
    sets=SETS if a.dataset=="all" else (a.dataset,)
    folds=range(3) if a.fold=="all" else [int(a.fold)]
    print("Plan:",len(sets)*len(folds)*({"both":2}.get(a.method,1)),
          "method/fold runs on",len(sets),"datasets, three outer folds available")
    if a.run:
        for ds in sets:
            for fold in folds:
                for method in (("bernn","scanvi") if a.method=="both" else (a.method,)):
                    fit(a.workspace.resolve(),ds,fold,method,a.bernn_image,a.scanvi_image,a.data_dir.resolve())
    elif not a.prepare:
        print("Dry run only: pass --prepare, then --run to start training.")

if __name__=="__main__":main()
