#!/usr/bin/env python3
"""BERNN CV3 transductive: unsupervised all-cell expression, supervised train labels only.
Three disjoint outer test batch folds with independent batch-level validation.
Records out-of-sample TEST scores after (and only after) checkpoint selection.
"""
import argparse, ast, copy, gc, hashlib, importlib.metadata, json, os, random, sys, time
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

ROOT=Path("/repo")
sys.path.insert(0,str(ROOT/"src/utils"))
source=(ROOT/"src/methods/bernn_gkf1/script.py").read_text()
tree=ast.parse(source)
# Reuse definitions only, never execute the historical adapter's fit or write path.
defs=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=defs,type_ignores=[]),str(ROOT/"src/methods/bernn_gkf1/script.py"),"exec"),globals())
_ORIGINAL_BERNN_SCALE_DATA=_bernn_holdout_module.scale_data

def atomic_json(p,obj):
    q=Path(str(p)+".tmp");q.write_text(json.dumps(obj,indent=2,default=str));q.replace(p)

def splits(obs,seed):
    """Use a frozen, audited three-fold batch partition, never recompute splits."""
    manifest_path=ROOT/"experiments/single_cell_paper_20261019/cv_manifests"/f"{CV_DATASET}_fold{CV_FOLD}.json"
    m=json.loads(manifest_path.read_text())
    assert m["dataset"]==CV_DATASET and m["fold"]==CV_FOLD
    assert seed==42+CV_FOLD
    batch=obs.batch.astype(str).to_numpy()
    groups=m["batch_lists"]
    ix={k:np.flatnonzero(np.isin(batch, v)) for k,v in groups.items()}
    for group in ("train","valid","test"):
        assert len(ix[group])==m["n_cells"][group],(group,len(ix[group]),m["n_cells"][group])
    assert sum(map(len,ix.values()))==len(batch)
    assert not(set(groups["train"])&set(groups["test"]) or set(groups["valid"])&set(groups["test"]) or set(groups["train"])&set(groups["valid"]))
    return ix,groups

def decode(trainer,encoded):
    with torch.no_grad():
        p=np.concatenate([trainer.ae.classifier(torch.as_tensor(encoded[j:j+4096],device=trainer.args.device)).argmax(1).cpu().numpy() for j in range(0,len(encoded),4096)])
    labs=np.asarray(trainer.unique_labels)[p]
    return trainer._label_encoder.inverse_transform(labs.astype(int)) if trainer._label_encoder is not None else labs

def encode(trainer,frame):
    return np.concatenate([np.asarray(trainer.transform(frame.iloc[j:j+4096]),dtype=np.float32) for j in range(0,len(frame),4096)])

class BatchAwareScaler:
    """Unsupervised per-batch preprocessing, including held-out validation/test batches.

    Each batch's scaler is fit on all unlabelled expression rows of that batch.
    No labels, predictions or MCC scores are used in preprocessing. For global
    scaling, the scaler is fit to training rows only, as before.
    """
    def __init__(self, global_scaler, per_batch, batch_lookup):
        self.global_scaler=global_scaler
        self.per_batch=per_batch
        self.batch_lookup=batch_lookup

    def transform(self, values):
        if not isinstance(values,pd.DataFrame):
            raise TypeError("Batch-aware scaler requires DataFrame with retained cell indexes")
        batches=self.batch_lookup.reindex(values.index)
        if batches.isna().any():
            raise ValueError("Input contains cells not found in the preprocessing batch manifest")
        result=np.empty(values.shape,dtype=np.float32)
        x=np.asarray(values,dtype=np.float32)
        for batch in pd.unique(batches):
            mask=(batches.to_numpy()==batch)
            scaler=self.per_batch.get(str(batch))
            if self.per_batch and scaler is None:
                raise ValueError(f"Missing normalization statistics for batch {batch!r}")
            if scaler is None:scaler=self.global_scaler
            result[mask]=scaler.transform(x[mask]).astype(np.float32,copy=False)
        return result

def transductive_per_batch_scaling(scale, data, batch_lookup, full_expression):
    """Per-batch scalers fit separately on each batch's unlabeled expression rows.

    This explicitly includes validation and outer-test batches for their OWN
    batch statistics, as the transductive preprocessing policy requires; this
    function does not use any cell labels. The model fit still uses train only.
    """
    if scale=="standard":
        return _memory_efficient_standard_scale_data(scale,data)
    if scale not in ("robust","standard_per_batch","robust_per_batch"):
        raise ValueError(f"Unsupported scaler: {scale}")
    cls=RobustScaler if "robust" in scale else StandardScaler
    train=data["inputs"]["train"]
    global_scaler=cls().fit(np.asarray(train,dtype=np.float32))
    by_batch={}
    if scale.endswith("_per_batch"):
        full_batch_ids=batch_lookup.reindex(full_expression.index).to_numpy()
        if pd.isna(full_batch_ids).any():
            raise ValueError("Expression rows missing batch IDs")
        for batch in pd.unique(full_batch_ids):
            indices=np.flatnonzero(full_batch_ids==batch)
            assert len(indices)>0
            # Crucial: obtain stats from this batch, even if it is valid/test.
            raw_batch=np.asarray(full_expression.iloc[indices],dtype=np.float32)
            by_batch[str(batch)]=cls().fit(raw_batch)
    fitted=BatchAwareScaler(global_scaler,by_batch,batch_lookup)
    for group in ("train","valid","test"):
        frame=data["inputs"].get(group)
        if frame is not None and len(frame):
            norm=fitted.transform(frame)
            data["inputs"][group]=pd.DataFrame(norm,index=frame.index,columns=frame.columns)
    data["inputs"]["all"]=data["inputs"]["train"]
    return data,fitted

class PaperTrainer(_MemoryEfficientOpenProblemsTrainer):
    def warmup_loop(self, optimizer_ae, scheduler, ae, celoss, loader, triplet_loss, mseloss,
                    warmup, epoch, optimizer_b, values, loggers, loaders, run, mapping=True):
        # BERNN legacy holdout trainer invokes the post-warmup unsupervised
        # loop twice per epoch; avoid duplicate gradient updates and unfair budget.
        if not warmup:
            if getattr(self,"_td_last_post_warmup_epoch",None)==epoch:
                return 1,ae,False
            self._td_last_post_warmup_epoch=epoch
        return super().warmup_loop(optimizer_ae,scheduler,ae,celoss,loader,triplet_loss,
              mseloss,warmup,epoch,optimizer_b,values,loggers,loaders,run,mapping)

    def make_samples_weights(self):
        # Keep BERNN's existing inverse-class-frequency sampling as unchanged control.
        super().make_samples_weights()
        # Includes all batch identities in the UNSUPERVISED warmup, but no
        # outer-test cell is added to the supervised 'train' loader.
        known=np.unique(np.concatenate([np.asarray(self.data["batches"][k]) for k in ("train","valid")]))
        self.paper_known_batch_ids=known.tolist()
        self.paper_outer_batch_map={
           v:i+int(np.max(known))+1 for i,v in enumerate(self.paper_outer_only_batches)}
        self.n_batches=len(known)+len(self.paper_outer_batch_map)
        assert self.n_batches==len(self.paper_full_batch_names)
        print("TRANSDUCTIVE_BATCH_AUDIT",json.dumps({
             "known_train_valid_batch_ids":len(known),
             "outer_only_batch_names":list(self.paper_outer_only_batches),
             "total_domain_classes":self.n_batches}),flush=True)
        mode=getattr(self,"paper_sampler_mode","class")
        if mode=="class":
            audit={"mode":"class","meaning":"original BERNN inverse-class-frequency weights","sampler":"WeightedRandomSampler replacement=True"}
            atomic_json(self.paper_out/"sampler_audit.json",audit)
            print("SAMPLER_AUDIT",json.dumps(audit),flush=True)
            return
        if mode!="class_batch":
            raise ValueError("Unknown sampler mode: "+str(mode))
        labels=np.asarray(self.data["labels"]["train"]).astype(str)
        batches=np.asarray(self.data["batches"]["train"]).astype(str)
        assert len(labels)==len(batches)==len(self.samples_weights["train"])
        cl,ci=np.unique(labels,return_inverse=True)
        combos,pairs=np.unique(np.column_stack([labels,batches]),axis=0,return_inverse=True)
        nc=np.bincount(ci)
        ncb=np.bincount(pairs)
        # Approximate uniform mass for each observed class-by-batch cell.
        # Maintain equal TOTAL mass per class. Cap rare combination oversampling.
        raw=1.0/ncb[pairs].astype(np.float64)
        raw=raw/np.bincount(ci,weights=raw)[ci]
        relative=raw*nc[ci]
        clipped=np.clip(relative,0.1,10.0)/nc[ci]
        weights=clipped/np.bincount(ci,weights=clipped)[ci]
        assert np.isfinite(weights).all() and (weights>0).all()
        assert np.allclose(np.bincount(ci,weights=weights),1.0)
        self.samples_weights["train"]=weights.tolist()
        perclass=np.bincount(ci,weights=weights)
        audit={"mode":"class_batch","n_train":len(labels),"n_classes":len(cl),
               "n_observed_class_batch_pairs":len(combos),
               "n_batches":len(np.unique(batches)),
               "relative_weight_clip":[0.1,10.0],
               "n_clipped":int(np.sum((relative<0.1)|(relative>10.0))),
               "pair_size_min":int(ncb.min()),"pair_size_max":int(ncb.max()),
               "class_weight_total_min":float(perclass.min()),
               "class_weight_total_max":float(perclass.max()),
               "sampler":"WeightedRandomSampler replacement=True",
               "uses_labels":"train only","uses_batches":"train only"}
        atomic_json(self.paper_out/"sampler_audit.json",audit)
        print("SAMPLER_AUDIT",json.dumps(audit),flush=True)
    def get_mccs(self,lists,traces):
        traces=super().get_mccs(lists,traces)
        # Full, ordered validation cells, including classes absent from training.
        # This value drives the library's existing checkpoint and early-stop rules.
        self.ae.eval()
        pred=decode(self,encode(self,self.paper_valid_frame))
        traces["valid"]["mcc"]=float(matthews_corrcoef(self.paper_valid_labels,pred))
        return traces
    def _notify_epoch(self,payload):
        self.paper_history.append(dict(payload))
        with (self.paper_out/"epochs.jsonl").open("a") as f: f.write(json.dumps(payload,default=str)+"\n")

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--out",required=True);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=40);p.add_argument("--warmup",type=int,default=12);p.add_argument("--preflight",action="store_true")
    p.add_argument("--hpo-config",required=True)
    p.add_argument("--fold",type=int,choices=(0,1,2),required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    global CV_DATASET, CV_FOLD
    CV_DATASET=a.dataset;CV_FOLD=a.fold
    assert a.seed==42+a.fold, "Seed and fold ID must match"
    start=time.time();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(5)
    inp=ROOT/"resources_full/task_batch_integration/datasets/cellxgene_census"/a.dataset/"log_cp10k/dataset.h5ad"
    with h5py.File(inp) as f:
        from anndata.experimental import read_elem
        obs=read_elem(f["obs"]);var=read_elem(f["var"]);uns=read_elem(f["uns"])
    idx,batch_names=splits(obs,a.seed);y=obs.cell_type.astype(str).to_numpy()
    train_labels=set(y[idx["train"]])
    frozen=ROOT/"experiments/single_cell_paper_20261019"/"cv_manifests"/f"{a.dataset}_fold{a.fold}.json"
    frozen_manifest=json.loads(frozen.read_text())
    assert batch_names==frozen_manifest["batch_lists"],"frozen train/valid/test batches differ"
    assert {k:len(v) for k,v in idx.items()}==frozen_manifest["n_cells"],"different batch split cell counts"
    manifest={"dataset":a.dataset,"seed":a.seed,"batch_lists":batch_names,
              "n_cells":{k:len(v) for k,v in idx.items()},
              "split_policy":"three outer batch-disjoint folds; inner validation on separate training-side batches; test used unsupervised only"}
    search=json.loads(Path(a.hpo_config).read_text())
    allowed={"lr","wd","dropout","gamma","beta","margin","smoothing","layer1","layer2","n_layers","batch_size",
               "variational","kan","class_triplet","class_triplet_w","dloss","scaler","nu","warmup","sampler_mode"}
    assert set(search)==allowed,("invalid HPO keys",set(search)^allowed)
    assert search["dloss"] in {"inverseTriplet","DANN","revDANN","normae","no"}
    assert search["scaler"] in {"standard","robust","standard_per_batch","robust_per_batch"}
    assert search["sampler_mode"] in ("class","class_batch")
    assert search["variational"] is False and search["kan"] is False
    assert search["scaler"] in {"robust", "standard", "robust_per_batch", "standard_per_batch"}
    assert 1 <= int(search["warmup"]) <= 150 and int(search["warmup"]) == a.warmup, "warmup config/CLI mismatch"
    atomic_json(out/"split.json",manifest)
    print("SPLIT",json.dumps(manifest),flush=True)
    cell_split=np.empty(len(obs),dtype=object)
    for k,v in idx.items():cell_split[v]=k
    # No per-cell labels or test predictions are written during hyperparameter selection.
    if a.preflight:return
    status={"state":"reading","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"dataset":a.dataset,"seed":a.seed}
    atomic_json(out/"status.json",status)
    data=read_anndata(str(inp),X="layers/normalized",obs="obs",var="var",uns="uns")
    features=_select_feature_indices(data,2000)
    frame=_dense_frame(data,features,"X");frame.index=obs.index
    feature_names=[str(x) for x in data.var.index[features]]
    del data;gc.collect()
    cfg=TrainingConfig(optimize_hyperparams=False,dloss=str(search["dloss"]),
        class_triplet=bool(search["class_triplet"]),class_triplet_w=float(search["class_triplet_w"]),
        variational=bool(search["variational"]),kan=bool(search["kan"]),n_layers=2,layer1=512,tied_weights=False,use_mapping=True,
        rec_loss="l1",scaler=str(search["scaler"]),log1p=False,use_l1=True,prune_network=False,update_grid=False,
        n_epochs=a.epochs,warmup=a.warmup,n_repeats=1,train_after_warmup=True,
        warmup_after_warmup=True,
        bs=min(1024,max(2,len(idx["train"])//2)),num_workers=0,groupkfold=True,device="cuda:0",
        dataset=a.dataset,exp_id="single_cell_paper_transductive_pilot",early_stop=50,early_warmup_stop=0)
    cfg.lr=float(search["lr"]);cfg.wd=float(search["wd"]);cfg.dropout=float(search["dropout"])
    cfg.margin=float(search["margin"]);cfg.smoothing=float(search["smoothing"])
    cfg.nu=float(search["nu"]);cfg.thres=0.0;cfg.gamma=float(search["gamma"]);cfg.beta=float(search["beta"])
    cfg.n_layers=int(search["n_layers"]);cfg.layer1=int(search["layer1"])
    cfg.bs=min(int(search["batch_size"]),max(2,len(idx["train"])//2))
    trainer=PaperTrainer(config=cfg,groupkfold=True,pools=False,keep_models=False,
        log_inputs=False,log_plots=False,log_tb=False,log_mlflow=False,log_dvclive=False)
    assert bool(getattr(trainer.args,"train_after_warmup",False)), "BERNN invariant failed: train_after_warmup must remain True for joint encoder/classifier updates"
    assert bool(getattr(trainer.args,"warmup_after_warmup",False)), "BERNN transductive invariant failed: unsupervised all_loader must run between supervised epochs"
    trainer.paper_valid_frame=frame.iloc[idx["valid"]];trainer.paper_valid_labels=y[idx["valid"]]
    trainer.paper_history=[];trainer.paper_out=out
    trainer.seed=a.seed;trainer.verbose=1
    trainer.paper_sampler_mode=search["sampler_mode"]
    # Only BATCH IDENTITIES of outer-test are carried through. Actual TEST LABELS
    # never enter any supervised loss, data-loader or model-selection logic.
    train_valid_names=set(batch_names["train"])|set(batch_names["valid"])
    trainer.paper_full_batch_names=sorted(set(obs.batch.astype(str)))
    trainer.paper_known_batch_name_to_id={
       name:i for i,name in enumerate(sorted(train_valid_names))}
    trainer.paper_outer_only_batches=sorted(set(batch_names["test"])-train_valid_names)
    assert len(trainer.paper_outer_only_batches)==len(batch_names["test"])
    params={"lr":.001,"dropout":0.0,"wd":1e-5,"margin":1.0,"smoothing":.1,"scaler":"standard",
            "gamma":.1,"beta":0.0,"nu":1.0,"thres":0.0,"prune_threshold":0.0,"warmup":a.warmup,
            "l1":0.0,"reg_entropy":0.0,"layer1":512,"layer2":256,"n_layers":2}
    params.update({k:search[k] for k in ("lr","wd","dropout","gamma","beta","margin",
                                         "smoothing","layer1","layer2","n_layers","scaler","nu")})
    atomic_json(out/"config.json",{"params":params, "hpo_search":search,"sampler_mode":search["sampler_mode"],"warmup":a.warmup,"epochs":a.epochs,"early_stop":50,"early_warmup_stop":0,
            "batch_size":cfg.bs,"seed":a.seed,"variational":bool(search["variational"]),
        "kan":bool(search["kan"]),"class_triplet":bool(search["class_triplet"]),
        "dloss":str(search["dloss"]),"scaler":str(search["scaler"]),"nu":float(search["nu"]),
        "selection":"inner validation MCC, outer test is predicted exactly once after checkpoint selection",
         "transductive_preprocessing":str(search["scaler"]).endswith("_per_batch"),
        "transductive_representation_training":True,
        "unlabeled_reconstruction_domains":["train","valid","outer_test"],
        "supervised_cell_scope":"train batches only",
         "fit_scope":"classification gradients train only; unsupervised representation gradients use expression from train, valid, outer-test",
        "preprocessing_policy":"per-batch scaler fitted on all unlabeled cells of each respective batch, including held-out batches",
        "feature_selection":"benchmark hvg_score if input wider than 2000; shared public preprocessing, not claimed train-only upstream HVG discovery",
        "training_image":os.environ.get("TRAINING_IMAGE_ID"),"versions":{k:importlib.metadata.version(k) for k in ["bernn","torch","numpy","scikit-learn","anndata"]},
        "adapter_sha256":hashlib.sha256(source.encode()).hexdigest()})
    # Validation labels are only for checkpoint selection, never for unsupervised
    # gradient updates or in target-domain triplet pools.
    # Unknown validation classes are excluded from cross-entropy with sentinel -1;
    # true strings above are retained for full-cell checkpoint MCC and final reporting.
    val_y=np.asarray([v if v in train_labels else "-1" for v in y[idx["valid"]]])
    fit_start=time.time();status["state"]="fitting";atomic_json(out/"status.json",status)
    batch_lookup=pd.Series(obs.batch.astype(str).to_numpy(),index=obs.index)
    # Add unlabeled VALID + OUTER TEST expression rows ONLY to the unsupervised
    # 'all' loader. BERNN's train loader remains strictly supervised TRAIN.
    # Label sentinels (-1) and nontrain 'sets' tags protect class triplets.
    def scale_and_add_unlabeled(scaling,raw,device="cpu"):
        data, fitted=transductive_per_batch_scaling(scaling,raw,batch_lookup,frame)
        train_rows=data["inputs"]["train"]
        # BERNN discards validation rows whose cell types were unseen in train.
        # For transductive reconstruction we must use the ORIGINAL full
        # validation expression, including those otherwise filtered rows.
        original_valid=frame.iloc[idx["valid"]]
        valid_rows=pd.DataFrame(fitted.transform(original_valid),
                   index=original_valid.index,columns=original_valid.columns)
        valid_batch_ids=np.asarray([
           trainer.paper_known_batch_name_to_id[b]
           for b in obs.batch.astype(str).to_numpy()[idx["valid"]]
        ],dtype=np.int64)
        heldout_frame=frame.iloc[idx["test"]]
        heldout_normalized=fitted.transform(heldout_frame)
        heldout=pd.DataFrame(heldout_normalized,index=heldout_frame.index,
                             columns=heldout_frame.columns)
        heldout_batch_ids=np.asarray([trainer.paper_outer_batch_map[b]
                       for b in obs.batch.astype(str).to_numpy()[idx["test"]]],dtype=np.int64)
        assert np.min(heldout_batch_ids)>np.max(data["batches"]["valid"])
        train_cats=np.asarray(data["cats"]["train"],dtype=np.int64)
        ntrain=len(train_rows);nvalid=len(valid_rows);ntest=len(heldout)
        assert ntrain==len(idx["train"]) and nvalid==len(idx["valid"])
        assert ntest==len(idx["test"])
        # Audit the fitting-time domain remapping. Validation filtering can
        # drop rows, but must not change any retained domain encodings.
        filtered_valid_idx=data["inputs"]["valid"].index
        if len(filtered_valid_idx):
            expected_mapped=valid_batch_ids[original_valid.index.get_indexer(filtered_valid_idx)]
            assert np.array_equal(np.asarray(data["batches"]["valid"],dtype=np.int64),
                                  expected_mapped), "Batch ID mismatch: forbid silent domain remapping"
        data["inputs"]["all"]=pd.concat((train_rows,valid_rows,heldout),axis=0,copy=False)
        data["cats"]["all"]=np.concatenate((train_cats,
                                 np.full(nvalid+ntest,-1,dtype=np.int64)))
        data["labels"]["all"]=np.concatenate((
             np.asarray(data["labels"]["train"]).astype(str),
             np.repeat("UnknownUnknown",nvalid+ntest)))
        data["batches"]["all"]=np.concatenate((
             np.asarray(data["batches"]["train"],dtype=np.int64),
             valid_batch_ids,
             heldout_batch_ids))
        data["names"]["all"]=np.concatenate((
             np.asarray(data["names"]["train"]).astype(str),
             np.asarray(["transductive_valid_"+str(i) for i in range(nvalid)]),
             np.asarray(["transductive_test_"+str(i) for i in range(ntest)])))
        data["sets"]["all"]=np.concatenate((
             np.repeat("train",ntrain),
             np.repeat("valid_unlabeled",nvalid),
             np.repeat("test_unlabeled",ntest)))
        data["orders"]["all"]=np.arange(ntrain+nvalid+ntest)
        assert len(data["inputs"]["all"])==ntrain+nvalid+ntest
        assert np.all(data["cats"]["all"][ntrain:]==-1)
        assert not np.any(data["sets"]["all"][ntrain:]=="train")
        assert np.array_equal(np.unique(data["batches"]["all"]),np.arange(trainer.n_batches))
        audit={"training_mode":"transductive_semi_supervised",
              "n_train_supervised":ntrain,"n_validation_unlabeled":nvalid,
              "n_outer_test_unlabeled":ntest,"n_all_unlabeled_unsupervised_pool":ntrain+nvalid+ntest,
              "n_domain_batches":int(trainer.n_batches),
              "known_labels_in_unsupervised_pool_only_from_train":True,
              "nontrain_labels_sentinel":-1,
              "outer_test_labels_read_by_optimizer":False,
              "representation_objectives":"reconstruction and domain loss on all",
              "supervised_objectives":"classification and class_triplet on train only"}
        atomic_json(out/"transductive_audit.json",audit)
        print("TRANSDUCTIVE_POOL_AUDIT",json.dumps(audit),flush=True)
        return data,fitted
    _bernn_holdout_module.scale_data=scale_and_add_unlabeled
    try:
        # The library needs a test bookkeeping loader. It aliases validation only.
        # No true outer-test expression, batches or labels are passed to fit.
        trainer.fit(frame.iloc[idx["train"]],y[idx["train"]],groups_train=obs.batch.astype(str).to_numpy()[idx["train"]],
            X_valid=trainer.paper_valid_frame,y_valid=val_y,groups_valid=obs.batch.astype(str).to_numpy()[idx["valid"]],
            X_test=trainer.paper_valid_frame,y_test=val_y,groups_test=obs.batch.astype(str).to_numpy()[idx["valid"]],
            params=params,internal_validation=False)
    finally:_bernn_holdout_module.scale_data=_ORIGINAL_BERNN_SCALE_DATA
    fit_seconds=time.time()-fit_start
    trainer.ae.eval()
    validation_predictions=decode(trainer,encode(trainer,trainer.paper_valid_frame))
    validation_mcc=float(matthews_corrcoef(trainer.paper_valid_labels,validation_predictions))
    # Test labels and test predictions never enter objective or checkpoint selection.
    # Per-batch scaling MAY use unlabeled test expression for its own scaler.
    score={"dataset":a.dataset,"seed":a.seed,"valid_mcc":validation_mcc,
           "fit_seconds":fit_seconds,"total_seconds":time.time()-start,
           "epochs_completed":len(trainer.paper_history),
            "hpo_config":search, "sampler_mode":search["sampler_mode"],"transductive_representation_training":True,
            "test_scored":True,
           "selection":"full-cell whole-batch validation MCC; held-out test never evaluated"}
    atomic_json(out/"validation_score.json",score)
    # Exactly one final test evaluation, after selection. Test labels are
    # used here for metrics only; they are not passed to the optimizer.
    emb=encode(trainer,frame)
    assert emb.shape[0]==len(obs) and np.isfinite(emb).all()
    native=decode(trainer,emb).astype(str)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import SGDClassifier
    tr=idx["train"]
    z=emb
    pc=None
    if z.shape[1]>50:
        pc=PCA(n_components=min(50,len(tr)-1),svd_solver="randomized",random_state=a.seed).fit(z[tr])
        z=pc.transform(z)
    scaler=StandardScaler().fit(z[tr]);z=scaler.transform(z)
    probe=SGDClassifier(loss="log_loss",class_weight="balanced",alpha=1e-4,
        random_state=a.seed,max_iter=600,tol=1e-3).fit(z[tr],y[tr])
    pred=probe.predict(z)
    result={"dataset":a.dataset,"fold":a.fold,"seed":a.seed,
        "fit_scope":"transductive expression from train+valid+test; supervised labels train only",
        "native":{k:float(matthews_corrcoef(y[ids],native[ids])) for k,ids in idx.items()},
        "common_probe":{k:float(matthews_corrcoef(y[ids],pred[ids])) for k,ids in idx.items()},
        "n_cells":{k:len(ids) for k,ids in idx.items()},
        "selected_on":"validation MCC only; test score computed after fitting",
        "transductive_audit_file":"transductive_audit.json"}
    ad.AnnData(obs=obs[[]],obsm={"X_emb":emb},uns={"dataset_id":uns["dataset_id"],
        "normalization_id":uns["normalization_id"],"method_id":"bernn_cv3_transductive",
        "fit_scope":result["fit_scope"]}).write_h5ad(out/"integrated.h5ad",compression="gzip")
    cell_split=np.empty(len(obs),dtype=object)
    for k,ids in idx.items():cell_split[ids]=k
    pd.DataFrame({"cell_id":obs.index.astype(str),"batch":obs.batch.astype(str).to_numpy(),
        "split":cell_split,"true_label":y,"native_prediction":native,
        "prediction":pred}).to_csv(out/"predictions.csv.gz",index=False)
    import joblib
    joblib.dump({"pca":pc,"scaler":scaler,"probe":probe},out/"probe.joblib")
    atomic_json(out/"scores.json",result)
    print("CV3_TRANSDUCTIVE_COMPLETED",json.dumps(result),flush=True)

    status.update(state="completed",completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  valid_mcc=validation_mcc,test_scored=True)
    atomic_json(out/"status.json",status)
    print("CV3_VALIDATION_AND_TEST_COMPLETED",json.dumps(score),flush=True)
if __name__=="__main__":main()
