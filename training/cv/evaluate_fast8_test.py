#!/usr/bin/env python3
"""Compute CV3 Fast-8 on each TEST subset, from test-only graphs and clusters.

Scientific contract:
- input representations were fitted transductively on all expression without test labels;
- only test rows are passed to integration metric components;
- test neighbors, clusters and isolated-label eligibility are recalculated afresh;
- missing/unavailable metrics remain N/A (not zero);
- raw metric means are descriptive, not OpenProblems normalized/composite scores.
"""
import argparse, json, os, subprocess, time, traceback
from pathlib import Path

SETS=("dkd","gtex_v9","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens")
PREFIX="ghcr.io/openproblems-bio/task_batch_integration"
EXP="experiments/single_cell_paper_20261019"
RES=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
METRICS=("asw_batch_sampled","ilisi","cell_cycle_conservation","ari","nmi",
         "graph_connectivity","isolated_label_f1","clisi")

def atomic(path,obj):
    q=path.with_suffix(path.suffix+".tmp");q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(path)

def container(repo,data_dir,image,name,entry,args,log,cpus=2,timeout=14400):
    cmd=["docker","run","--rm","--cpus",str(cpus),"--memory=135g",
        "--ipc=host","-e","OMP_NUM_THREADS=2","-e","OPENBLAS_NUM_THREADS=2",
        "-e","MKL_NUM_THREADS=2","-e","NUMBA_NUM_THREADS=2",
        "-e","JAX_PLATFORMS=cpu",
        "-v",str(repo)+":/repo",
        "-v",str(data_dir)+":/repo/resources_full/task_batch_integration/datasets/cellxgene_census:ro",
        "-w","/repo/"+EXP,"--entrypoint","python",image,entry]+[str(x) for x in args]
    with log.open("a") as fh:
        fh.write("EXEC "+' '.join(map(str,cmd))+"\n");fh.flush()
        result=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f"exit {result.returncode}: {name}: see {log}")

def pth(path,repo):return "/repo/"+str(Path(path).relative_to(repo))

def component(repo,data_dir,dir_,kind,name,par):
    out=repo/par["output"].removeprefix("/repo/")
    if out.exists() and out.stat().st_size>0:return out
    image=PREFIX+"/"+kind+"/"+name+":test"
    source=f"/repo/src/{kind}/{name}/script.py"
    container(repo,data_dir,image,name,"code/component.py",[
       "--source",source,"--par",json.dumps(par,sort_keys=True)],
       dir_/(name+".log"),cpus=2)
    if not out.exists() or not out.stat().st_size:raise RuntimeError(f"No output: {name}")
    return out

def target_dir(repo,base,scope):
    return repo/EXP/"cv3_fast8_runs"/base.relative_to(repo/EXP)/scope

def subset(repo,data_dir,base,scope):
    dst=target_dir(repo,base,scope);dst.mkdir(parents=True,exist_ok=True)
    if (dst/"subset_ready.json").exists():return dst
    container(repo,data_dir,"bernn-openproblems-vm2:cu121-fast-644b5a9",
        "prepare","/repo/training/cv/prepare_fast8_subset.py",
        ["--base",pth(base,repo),"--data-dir",
         "/repo/resources_full/task_batch_integration/datasets/cellxgene_census","--scope",scope,"--destination",pth(dst,repo)],
        dst/"prepare.log",cpus=2)
    assert (dst/"subset_ready.json").exists()
    return dst

def sample_asw(repo,data_dir,dst):
    out=dst/"asw_batch_sampled.json"
    if out.exists():return
    container(repo,data_dir,"bernn-openproblems-vm2:cu121-fast-644b5a9",
        "asw_batch_sampled","/repo/training/cv/fast8_sampled_asw.py",
        ["--subset",pth(dst,repo)],dst/"asw_batch_sampled.log",cpus=2,timeout=3600)

def get_values(repo,data_dir,dst):
    out=dst/"metrics_raw.json"
    if out.exists():return json.loads(out.read_text())
    container(repo,data_dir,"bernn-openproblems-vm2:cu121-fast-644b5a9",
        "collect_metrics","/repo/training/cv/fast8_collect.py",
        ["--subset",pth(dst,repo)],dst/"collect.log",cpus=2)
    return json.loads(out.read_text())

def evaluate(repo,data_dir,base,scope):
    dst=subset(repo,data_dir,base,scope)
    if (dst/"metrics_raw.json").exists():
        print("ALREADY_DONE",base,scope,flush=True);return
    info=json.loads((dst/"subset_ready.json").read_text())
    # Only one batch means no cross-batch mixing can be evaluated.
    one_batch=info["n_batches"]<2
    inp=dst/"dataset_only.h5ad"
    sol=dst/"solution_only.h5ad"
    emb=dst/"integrated_only.h5ad"
    proc=dst/"processed.h5ad"
    if not proc.exists():
        result=component(repo,data_dir,dst,"data_processors","transform",{
           "input_integrated":pth(emb,repo),"input_dataset":pth(inp,repo),
           "expected_method_types":["embedding"],"output":pth(proc,repo)})
        assert result==proc
    clusters=[]
    for r in RES:
        f=dst/f"clustering_{r}.h5ad"
        if not f.exists():
            result=component(repo,data_dir,dst,"data_processors","precompute_clustering_run",{
              "input":pth(proc,repo),"resolution":r,"output":pth(f,repo)})
            assert result==f
        clusters.append(pth(f,repo))
    merged=dst/"processed_clustered.h5ad"
    if not merged.exists():
        result=component(repo,data_dir,dst,"data_processors","precompute_clustering_merge",{
             "input":pth(proc,repo),"clusterings":clusters,"output":pth(merged,repo)})
        assert result==merged
    for method in ("clustering_overlap","graph_connectivity","cell_cycle_conservation","lisi","isolated_label_f1"):
        target=dst/(method+".h5ad")
        if target.exists():continue
        if method=="lisi" and one_batch:
            (dst/"ilisi.NA").write_text("One batch: normalized iLISI denominator zero; cLISI may remain defined")
        par={"input_integrated":pth(merged,repo),"input_solution":pth(sol,repo),
             "output":pth(target,repo),"resolutions":RES}
        try:
            component(repo,data_dir,dst,"metrics",method,par)
        except Exception as exc:
            (dst/(method+".error")).write_text(repr(exc))
            print("METRIC_FAILURE",method,base.name,scope,str(exc),flush=True)
    sample_asw(repo,data_dir,dst)
    metrics=get_values(repo,data_dir,dst)
    print("FAST8_SUBSET_RESULT",base,scope,json.dumps(metrics),flush=True)

def paths(repo):
    exp=repo/EXP
    for ds in SETS:
        for fold in range(3):
            seed=42+fold
            for m in ("bernn","scanvi"):
                base=(exp/"bernn"/str(seed)/ds) if m=="bernn" else (exp/"comparators"/str(seed)/"scanvi"/ds)
                yield base,ds,fold,m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--scope",choices=("test","valid"),default="test")
    p.add_argument("--once",choices=("dkd-0-bernn","dkd-0-scanvi"),default=None)
    p.add_argument("--poll",action="store_true")
    a=p.parse_args();repo=a.repo.resolve();data_dir=a.data_dir.resolve()
    while True:
        count_completed=0
        count_failed=0
        for base,ds,fold,m in paths(repo):
            if a.once and f"{ds}-{fold}-{m}"!=a.once:continue
            if not (base/"scores.json").exists() or not (base/"integrated.h5ad").exists():continue
            if (target_dir(repo,base,a.scope)/"metrics_raw.json").exists():
                count_completed+=1
                continue
            status=target_dir(repo,base,a.scope)/"worker_status.json";status.parent.mkdir(parents=True,exist_ok=True)
            if a.poll and status.exists() and json.loads(status.read_text()).get("stage")=="failed":
                count_failed+=1
                continue  # Requires investigation, not repeated blind retries.
            atomic(status,{"stage":"running","start":time.time(),"scope":a.scope})
            try:
                evaluate(repo,data_dir,base,a.scope)
                atomic(status,{"stage":"completed","end":time.time(),"scope":a.scope})
                count_completed+=1
            except Exception as e:
                atomic(status,{"stage":"failed","end":time.time(),"reason":str(e),
                    "traceback":traceback.format_exc()[-2500:]})
                count_failed+=1
                print("EVALUATION_FAILED",base,repr(e),flush=True)
        atomic(repo/EXP/"cv3_fast8_runs"/"queue_status.json",{
            "updated":time.time(),"scope":a.scope,"completed":count_completed,
            "failed":count_failed,"total":30})
        print("CV3_FAST8_PROGRESS",count_completed,"/30 completed",count_failed,"failed",flush=True)
        if not a.poll or count_completed+count_failed>=30:break
        time.sleep(120)

if __name__=="__main__":main()
