#!/usr/bin/env python3
"""Execute an existing component source with explicit parameters, without changing it."""
import argparse,ast,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--par",required=True);p.add_argument("--mask",default=None)
a=p.parse_args();src=Path(a.source)
# Route the previously queued official scanvi script to the audited wrapper.
# This is resolved when each component process starts, so the already-running
# GPU queue need not be interrupted. Official method script remains unchanged.
if a.mask and str(src)=="/repo/src/methods/scanvi/script.py":
 src=Path("code/wrappers/scanvi/script.py")
tree=ast.parse(src.read_text());defaults={}
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="par" for t in n.targets):
  try:defaults.update(ast.literal_eval(n.value))
  except (ValueError,TypeError):pass
tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ["par","meta"] for t in n.targets))]
par=defaults|json.loads(a.par);meta={"name":src.parent.name,"resources_dir":"/repo/src/utils","cpus":5,"memory_mb":300000}
if a.mask:
 import h5py,numpy as np
 from anndata.experimental import read_elem
 manifest=json.loads(Path(a.mask).read_text())
 # Patch read utility locally so only allowed training labels enter a supervised integrator.
 sys.path.insert(0,"/repo/src/utils")
 import read_anndata_partial
 original=read_anndata_partial.read_anndata
 def read_masked(*args,**kwargs):
  x=original(*args,**kwargs)
  if "cell_type" in x.obs:
   b=x.obs.batch.astype(str);labels=x.obs.cell_type.astype(str).to_numpy()
   labels[~np.isin(b,manifest["batch_lists"]["train"])]="UnknownUnknown"
   x.obs["cell_type"]=labels
  return x
 read_anndata_partial.read_anndata=read_masked
exec(compile(tree,str(src),"exec"),{"par":par,"meta":meta,"__name__":"__main__"})
