#!/usr/bin/env python3
import argparse,json,subprocess,time,traceback,hashlib
from pathlib import Path
ROOT=Path("/home/simonp/task_batch_integration_bernn");EXP=ROOT/"experiments/single_cell_paper_20261019"
PREFIX="ghcr.io/openproblems-bio/task_batch_integration/"
def run_component(kind,name,par,outdir,timeout=172800):
 image=PREFIX+kind+"/"+name+":test"
 cmd=["docker","run","--rm","--cpus=5","--memory=300g","-e","OMP_NUM_THREADS=5","-e","OPENBLAS_NUM_THREADS=5",
      "-e","NUMBA_NUM_THREADS=5","-e","JAX_PLATFORMS=cpu","-v",str(ROOT)+":/repo",
      "-w","/repo/experiments/single_cell_paper_20261019","--entrypoint","python",image,
      "code/component.py","--source",f"/repo/src/{kind}/{name}/script.py","--par",json.dumps(par)]
 start=time.time()
 with (outdir/(name+".log")).open("a") as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 record={"component":name,"image":image,"exit":code,"seconds":time.time()-start,"scope":"all original cells; fixed trained embedding; metric evaluation only"}
 (outdir/(name+".status.json")).write_text(json.dumps(record,indent=2))
 if code:raise RuntimeError(str(record))
def cp(p):return "/repo/"+str(Path(p).relative_to(ROOT))
def process(base,ds,slow=False):
 met=base/"metrics";met.mkdir(exist_ok=True)
 inp=ROOT/"resources_full/task_batch_integration/datasets/cellxgene_census"/ds/"log_cp10k/dataset.h5ad"
 sol=inp.parent/"solution.h5ad";integrated=base/"integrated.h5ad";proc=met/"processed.h5ad"
 if slow:
  if not (met/"bras.h5ad").exists():run_component("metrics","bras",{"input_integrated":cp(integrated),"input_solution":cp(sol),"output":cp(met/"bras.h5ad")},met)
  return
 if not proc.exists():
  run_component("data_processors","transform",{"input_integrated":cp(integrated),"input_dataset":cp(inp),"expected_method_types":["embedding"],"output":cp(proc)},met)
 cluster=[]
 for r in [.2,.3,.4,.5,.6,.7,.8]:
  o=met/f"clustering_{r}.h5ad";cluster.append(cp(o))
  if not o.exists():run_component("data_processors","precompute_clustering_run",{"input":cp(proc),"resolution":r,"output":cp(o)},met)
 merged=met/"processed_clustered.h5ad"
 if not merged.exists():run_component("data_processors","precompute_clustering_merge",{"input":cp(proc),"clusterings":cluster,"output":cp(merged)},met)
 for name in ["clustering_overlap","graph_connectivity","pcr","cell_cycle_conservation","lisi","isolated_label_f1"]:
  out=met/(name+".h5ad")
  if out.exists():continue
  try:run_component("metrics",name,{"input_integrated":cp(merged),"input_solution":cp(sol),"output":cp(out),"resolutions":[.2,.3,.4,.5,.6,.7,.8]},met)
  except Exception as e:print("METRIC_FAILURE",base,name,str(e),flush=True)
 (met/"fast_done.json").write_text(json.dumps({"completed":time.time()}))
def main():
 done=set()
 while True:
  changed=False
  bases=sorted(list((EXP/"bernn").glob("*/*"))+list((EXP/"comparators").glob("*/*/*")))
  # Cheap/core-compatible endpoints before exact-distance BRAS; one metric process at a time.
  for base in bases:
   if not (base/"scores.json").exists():continue
   ds=base.name
   if not (base/"metrics/fast_done.json").exists() and (str(base),"fast") not in done:
    try:process(base,ds)
    except Exception:traceback.print_exc()
    done.add((str(base),"fast"));changed=True
  # Publish initial-pass BRAS while repeats train; repeat BRAS follows in the same worker.
  for base in bases:
   if not (base/"scores.json").exists() or (base/"metrics/bras.h5ad").exists() or (str(base),"bras") in done:continue
   try:process(base,base.name,True)
   except Exception:traceback.print_exc()
   done.add((str(base),"bras"));changed=True
   break
  (EXP/"metrics_worker_status.json").write_text(json.dumps({"updated":time.time(),"attempted":len(done)}))
  if not changed:time.sleep(30)
if __name__=="__main__":main()
