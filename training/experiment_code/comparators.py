#!/usr/bin/env python3
"""Frozen rc3 comparator queue. CPU methods may overlap GPU fits; GPU methods wait for BERNN queue."""
import argparse,json,os,shutil,subprocess,time,signal
from pathlib import Path
ROOT=Path("/home/simonp/task_batch_integration_bernn");EXP=ROOT/"experiments/single_cell_paper_20261019"
REF=Path("/home/simonp/task_batch_integration_hpo_rc3/config/openproblems_public_reference_20261005")
PREFIX="ghcr.io/openproblems-bio/task_batch_integration/"
CPU=["combat","harmony","harmonypy","batchelor_fastmnn","liger","pyliger"]
GPU=["scanvi","scvi","scalex","uce"]
DATASETS=["dkd","gtex_v9","hypomap","mouse_pancreas_atlas","immune_cell_atlas","tabula_sapiens"]
def cpath(p):return "/repo/"+str(Path(p).relative_to(ROOT))
def write(p,x):p.write_text(json.dumps(x,indent=2))
def base_docker(image,name,gpu=False):
 cmd=["docker","run","--rm","--name",name,"--cpus=5","--memory=140g","--ipc=host","-e","OMP_NUM_THREADS=5","-e","OPENBLAS_NUM_THREADS=5","-e","NUMBA_NUM_THREADS=5","-e","PYTHONHASHSEED=42"]
 if gpu:
  cmd+=["--device=/dev/nvidia0","--device=/dev/nvidia1","--device=/dev/nvidiactl","--device=/dev/nvidia-uvm","--device=/dev/nvidia-uvm-tools",
  "-v","/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro","-v","/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro","-e","CUDA_VISIBLE_DEVICES="+os.environ.get("SC_PAPER_GPU_DEVICE","1")]
 cmd+=["-e","PYTHONUNBUFFERED=1","-v",str(ROOT)+":/repo","-w","/repo/experiments/single_cell_paper_20261019","--entrypoint","python",image]
 return cmd
def command(cmd,log,timeout=None,name=None):
 started=time.time()
 with log.open("a") as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   if name:subprocess.run(["docker","stop","-t","10",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   os.killpg(p.pid,signal.SIGTERM)
   code=124
 return code,time.time()-started
def frozen():
 d=EXP/"reference";d.mkdir(exist_ok=True)
 for name in ["results.json","method_info.json","provenance.json","task_info.json","metric_info.json","dataset_info.json"]:
  if not (d/name).exists():shutil.copy2(REF/name,d/name)
 r=json.loads((d/"results.json").read_text());mi=json.loads((d/"method_info.json").read_text())
 control={x["method_id"] for x in mi if x.get("is_baseline")};scores={}
 for x in r:
  if x["method_id"] in control:continue
  scores.setdefault(x["method_id"],[]).append(x["mean_score"])
 ranking=sorted([{"method":k,"mean_score":sum(v)/len(v),"datasets":len(v)} for k,v in scores.items()],key=lambda x:-x["mean_score"])[:10]
 write(d/"top10.json",{"ranking":ranking,"aggregation":"mean of official per-dataset 13-metric means; NA=0 per reference","source_provenance":json.loads((d/"provenance.json").read_text()),"frozen_at":time.time()})
 return {x["method"] for x in ranking}
def fit(method,ds,gpu,seed=42):
 out=EXP/"comparators"/str(seed)/method/ds;out.mkdir(parents=True,exist_ok=True)
 # GPU0's existing worker sees reserved runtime and safely skips seed42-43.
 # GPU1's separate worker may process ONLY its own reservation marker.
 rt=out/"runtime.json"
 if rt.exists():
  reserved=(method=="scanvi" and os.environ.get("SC_PAPER_SCANVI_GPU1_RESERVED")=="1" and
            json.loads(rt.read_text()).get("status")=="reserved_for_parallel_scanvi_gpu1")
  if not reserved:return
 b=EXP/"bernn"/str(seed)/ds
 if not (b/"scores.json").exists():return
 bs=json.loads((b/"scores.json").read_text());cap=None  # No arbitrary wall-clock cutoff
 inp=ROOT/"resources_full/task_batch_integration/datasets/cellxgene_census"/ds/"log_cp10k/dataset.h5ad"
 split=b/"split.json";raw=out/"method_output.h5ad";name=f"sc-paper-{method}-{ds}".replace("_","-")
 if method=="scanvi" and os.environ.get("SC_PAPER_SCANVI_GPU1_RESERVED")=="1":
  name+=f"-parallel-gpu1-seed{seed}" # separate Docker names from GPU0's original queue
 write(out/"protocol.json",{"method":method,"dataset":ds,"seed":seed,"fit_scope":"transductive; no evaluation labels used for supervised training","time_cap_policy":"no artificial wall-clock cap; allow integration to complete",
      "test_labels_available_to_integrator":False,"fit_time_cap_seconds":cap,"reference_bernn_fit_seconds":bs["fit_seconds"],
      "split":str(split),"comparison_note":"BERNN encoder trained only on training cells; transductive comparator regime reported separately"})
 if method=="uce":
  write(out/"runtime.json",{"status":"blocked_missing_model_asset","detail":"UCE pretrained bundle not resolved locally; not downloading a large bundle without checking disk budget","cap_seconds":cap});return
 image=(os.environ.get("SC_PAPER_SCVI_IMAGE",PREFIX+"methods/"+method+":test") if method in ["scvi","scanvi"] else PREFIX+"methods/"+method+":test")
 par={"input":cpath(inp),"output":cpath(raw)}
 if method in ["harmony","batchelor_fastmnn","liger"]:
  wrapper=EXP/"code/wrappers"/method
  if not wrapper.parent.exists():wrapper.parent.mkdir(parents=True)
  if not wrapper.exists():
   shutil.copytree(ROOT/"target/executable/methods"/method,wrapper)
   f=wrapper/method;s=f.read_text().replace(f"methods/{method}:dev",f"methods/{method}:test");f.write_text(s)
  cmd=[str(wrapper/method),"---cpus","5","---memory","140GB","--input",str(inp),"--output",str(raw)]
 else:
  method_source=("code/wrappers/scanvi/script.py" if method=="scanvi" else f"/repo/src/methods/{method}/script.py")
  cmd=base_docker(image,name,gpu)+["code/component.py","--source",method_source,"--par",json.dumps(par)]
  if method=="scanvi":cmd+=["--mask",cpath(split)]
 code,secs=command(cmd,out/"fit.log",cap,name if method not in ["harmony","batchelor_fastmnn","liger"] else None)
 write(out/"runtime.json",{"status":"fitted" if code==0 and raw.is_file() and raw.stat().st_size>0 else ("timeout" if code==124 else "failed"),"exit":code,"fit_seconds":secs,"cap_seconds":cap,"image":image,"output_verified":bool(raw.is_file() and raw.stat().st_size>0)})
 if code or not raw.is_file() or not raw.stat().st_size:return
 # Standard OpenProblems transform: corrected features -> 50D PCA, or retain embedding.
 proc=out/"integrated.h5ad";im=PREFIX+"data_processors/transform:test"
 pars={"input_integrated":cpath(raw),"input_dataset":cpath(inp),"expected_method_types":["feature"] if method=="combat" else ["embedding"],"output":cpath(proc)}
 c,t=command(base_docker(im,name+"-process")+["code/component.py","--source","/repo/src/data_processors/transform/script.py","--par",json.dumps(pars)],out/"process.log",None,name+"-process")
 if c:write(out/"evaluation_failure.json",{"stage":"process","exit":c});return
 # Identical probe, same six-dataset batch manifest; no test-based selection.
 im="bernn-openproblems-vm2:cu121-fast-644b5a9"
 probe_args=["code/probe.py","--input",cpath(inp),"--integrated",cpath(proc),"--split",cpath(split),"--out",cpath(out),"--method",method]
 if method=="scanvi":probe_args+=["--native-output",cpath(raw)]
 c,t=command(base_docker(im,name+"-probe")+probe_args,out/"probe.log",None,name+"-probe")
 if c:write(out/"evaluation_failure.json",{"stage":"probe","exit":c})
def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=["cpu","gpu"],required=True);a=p.parse_args();allowed=frozen()
 if a.mode=="gpu":
  while True:
   q=EXP/"queue_status.json"
   if q.exists() and json.loads(q.read_text()).get("state")=="finished":break
   time.sleep(30)
 methods=CPU if a.mode=="cpu" else GPU
 while True:
  pending=False
  for ds in DATASETS:
   if not (EXP/"bernn/42"/ds/"scores.json").exists():pending=True;continue
   for method in methods:
    if method not in allowed:continue
    print("COMPARATOR",method,ds,flush=True)
    try:fit(method,ds,a.mode=="gpu")
    except Exception as e:
     out=EXP/"comparators"/"42"/method/ds;out.mkdir(parents=True,exist_ok=True)
     write(out/"runtime.json",{"status":"orchestration_failed","error":repr(e)})
  if not pending:break
  time.sleep(30)
 print("COMPARATOR_QUEUE_FINISHED",a.mode,flush=True)
if __name__=="__main__":main()
