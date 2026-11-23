"""Frozen B50k localization diagnostic, isolated from production and training."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.localization_metrics import (choose_sample, residual_features, center_size_oracles,
                                        pair_iou, support_bounds, summarize, xyxy, paired_coverage)
from tools.selection_retention_metrics import iou_matrix, clipped_pixels, coco_xyxy

ROOT=Path(__file__).resolve().parents[1]
PRIOR=Path('/root/autodl-tmp/exps/selection_retention_b50k_20260917_full01')
ANNOTATION=Path('/root/autodl-tmp/datasets/DIOR_HBB_COCO/coco/annotations/instances_val.json')


def write_json(path,value):
    with open(path,'x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def load_json(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def records(path):
    with open(path,encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def dataset():
    data=load_json(ANNOTATION)
    images={v['id']:dict(v) for v in data['images']}
    anns=defaultdict(list)
    for a in data['annotations']:anns[a['image_id']].append(a)
    for i,v in images.items():v['gt_count']=len(anns[i])
    return images,anns,data['categories']


def base_record(a,image_id,image):
    x,y,w,h=a['bbox']
    return dict(image_id=int(image_id),annotation_id=int(a['id']),category_id=int(a['category_id']),
                small=bool(a['area']<=1024),short_bin=int(np.searchsorted([8,12,20],min(w,h),side='right')),
                density_bin=int(np.searchsorted([10,50],image['gt_count'],side='left')),
                short_px=float(min(w,h)),long_px=float(max(w,h)),area=float(a['area']),
                scene_gt=image['gt_count'],aspect=float(max(w,h)/min(w,h)),
                border_px=float(min(x,y,image['width']-x-w,image['height']-y-h)))


def verify_old_trace(new,old):
    for k in old:
        if k not in new or new[k].dtype!=old[k].dtype or not np.array_equal(new[k],old[k]):
            raise RuntimeError('new capture differs from original trace: '+k)


def plain_inputs(inputs):
    # LazyConfig dataset substitution wraps each mapping as DictConfig. Restore
    # ordinary batch dictionaries before hashing/model use, preserving tensors.
    return [dict(item) for item in inputs]


def check_sources():
    from tools.evaluate_detail_floor import sha256
    pre=load_json(PRIOR/'preflight.json')
    for name,digest in pre['source_manifest'].items():
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        if sha256(p)!=digest:raise RuntimeError('original source changed: '+str(p))
    if sha256(ANNOTATION)!=pre['immutable_hashes']['annotation']:raise RuntimeError('annotation changed')
    return len(pre['source_manifest'])


def prepare(output):
    check_sources()
    output.mkdir(exist_ok=False)
    images,annotations,categories=dataset()
    old=list(records(PRIOR/'gt_records.jsonl'))
    sample=choose_sample(old,images)
    ids=sorted(r['image_id'] for r in sample)
    digest=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
    expected='009bd4cd3d46901c62cab1f49e6bd514f781dd2a7013ed9baa3b0ae5b2446040'
    if digest!=expected:raise RuntimeError('sample differs from independent auditor: '+digest)
    write_json(output/'sample.json',dict(images=sample,image_ids=ids,ids_sha256=digest,
               interpretation='Outcome/size/density-stratified diagnostic subset, NOT population AP estimate.'))
    previous={r['annotation_id']:r for r in old}
    del old
    result=[]
    with open(output/'census.jsonl','x',encoding='utf-8') as stream:
        for index,i in enumerate(sorted(images)):
            with np.load(PRIOR/'traces'/f'{i}.npz') as t:
                raw=coco_xyxy(clipped_pixels(t['raw_boxes'],t['resized_hw'],t['original_hw']))
                gt=np.array([a['bbox'] for a in annotations[i]],dtype=np.float64);gt[:,2:]+=gt[:,:2]
                overlap=iou_matrix(raw,gt)
                best=overlap.argmax(0);boxes=raw[best]
                res=residual_features(boxes,gt);oracle=center_size_oracles(boxes,gt)
                for g,a in enumerate(annotations[i]):
                    r=base_record(a,i,images[i]);v=previous[a['id']]['thresholds']['0.75']
                    best_iou=float(overlap[best[g],g])
                    if not np.isclose(best_iou,v['best_iou']['raw_geometry'],rtol=0,atol=1e-12):
                        raise RuntimeError('census geometry differs from full trace')
                    r.update(best_iou=best_iou,raw_fail=best_iou<.75,
                             matched=v['official']['small' if r['small'] else 'all']['matched'],
                             bucket=v['pre_matching_bucket'],best_step=int(best[g]//900),best_query=int(best[g]%900),
                             **{k:float(x[g]) for k,x in res.items()},
                             center_only_iou=float(oracle['center_only'][g]),size_only_iou=float(oracle['size_only'][g]),
                             center_only_recovers=bool(best_iou<.75 and oracle['center_only'][g]>=.75),
                             size_only_recovers=bool(best_iou<.75 and oracle['size_only'][g]>=.75))
                    stream.write(json.dumps(r,allow_nan=False)+'\n');result.append(r)
            if (index+1)%500==0: print(f'CENSUS {index+1}/{len(images)}',flush=True)
    write_json(output/'census_summary.json',dict(status='complete',images=len(images),gt_count=len(result),
               groups=summarize(result),categories=categories,
               oracle_note='Fixed best-actual-refined candidate. Center/size replacement uses GT: not deployable or AP gains. Counts overlap; both yields1.'))
    print('CENSUS_DONE',len(result),flush=True)


def capture(output,smoke=False,smoke_tag='smoke'):
    import torch
    from detectron2.config import LazyConfig,instantiate
    from tools.diagnose_selection_retention import (CONFIG,CHECKPOINT,CHECKPOINT_SHA,set_smoke_dataset,_rng_state,_same_rng)
    from tools.evaluate_detail_floor import (strict_promote_ema,seed_image,sha256,tensor_digest,input_digest,verify_saved_config)
    from tools.localization_trace import LocalizationTrace
    check_sources()
    selected=[735,8677,9663] if smoke else load_json(output/'sample.json')['image_ids']
    if Path(smoke_tag).name!=smoke_tag:raise ValueError('smoke tag must be a single directory name')
    dest=output/(smoke_tag if smoke else 'capture');dest.mkdir(exist_ok=False)
    trace_dir=dest/'traces';trace_dir.mkdir()
    cfg=LazyConfig.load(str(CONFIG));verify_saved_config(cfg,CHECKPOINT.parent/'config.yaml')
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('checkpoint mismatch')
    cfg.model.device='cuda'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=instantiate(cfg.model).to('cuda').eval()
    ckpt=torch.load(CHECKPOINT,map_location='cpu')
    if ckpt['trainer']['iteration']!=49999:raise RuntimeError('wrong iteration')
    model.load_state_dict(ckpt['model'],strict=True)
    if strict_promote_ema(model,ckpt['ema_state'])!=682:raise RuntimeError('wrong EMA tensor count')
    del ckpt;gc.collect()
    digest=tensor_digest(model)
    if digest!=load_json(PRIOR/'preflight.json')['model_tensor_sha256']:raise RuntimeError('wrong frozen model')
    data=instantiate(cfg.dataloader.test.dataset)
    data=[item for item in data if int(item['image_id']) in selected]
    if {int(v['image_id']) for v in data}!=set(selected):raise RuntimeError('missing images')
    set_smoke_dataset(cfg,data)
    old_digests={r['image_id']:r['input_digest'] for r in records(PRIOR/'input_digests.jsonl')}
    owned={str(p):sha256(p) for p in (ROOT/'tools').glob('*localization*.py')}
    write_json(dest/'preflight.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
               model_tensor_sha256=digest,images=selected,owned_source_hashes=owned,training=False))
    seen=[];equivalence=[]
    with torch.inference_mode(),open(dest/'input_digests.jsonl','x') as logs:
        for inputs in instantiate(cfg.dataloader.test):
            inputs=plain_inputs(inputs)
            i=int(inputs[0]['image_id']);ih=input_digest(inputs)
            if i in seen or ih!=old_digests[i]:raise RuntimeError('duplicate or input differs')
            seed_image(i)
            with LocalizationTrace(model) as trace:outputs=model(inputs)
            rng=_rng_state(torch)
            arrays=trace.finish(outputs,(inputs[0]['height'],inputs[0]['width']))
            with np.load(PRIOR/'traces'/f'{i}.npz') as old:verify_old_trace(arrays,old)
            if smoke or len(seen)<2:
                seed_image(i);plain=model(inputs)
                trace._compare_instances(outputs[0]['instances'],plain[0]['instances'],'plain equality')
                if not _same_rng(torch,rng,_rng_state(torch)):raise RuntimeError('trace RNG drift')
                equivalence.append(i)
            if input_digest(inputs)!=ih:raise RuntimeError('input mutated')
            with open(trace_dir/f'{i}.npz','xb') as f:np.savez_compressed(f,**arrays)
            logs.write(json.dumps(dict(image_id=i,input_digest=ih))+'\n');logs.flush();seen.append(i)
            if len(seen)%25==0 or len(seen)==len(selected):print(f'CAPTURE {len(seen)}/{len(selected)} EXACT_OLD_EQUAL',flush=True)
    if set(seen)!=set(selected) or len(seen)!=len(selected):raise RuntimeError('incomplete capture')
    if tensor_digest(model)!=digest or model.training or sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('state drift')
    if {p:sha256(Path(p)) for p in owned}!=owned:raise RuntimeError('diagnostic source drift')
    count=check_sources()
    write_json(dest/'summary.json',dict(status='complete',images=len(seen),image_ids=seen,
               old_trace_exact_equal=True,plain_rng_equal_images=equivalence,hashes_unchanged=True,
               original_source_entries=count,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
               model_tensor_sha256=digest,training=False))
    print('CAPTURE_DONE',flush=True)


def analyze(output):
    check_sources()
    capture_summary=load_json(output/'capture/summary.json')
    if capture_summary['status']!='complete':raise RuntimeError('incomplete capture')
    census=list(records(output/'census.jsonl'))
    for r in census:
        r['center_size_interaction']=1-r['center_only_iou']-r['size_only_iou']+r['best_iou']
    write_json(output/'census_factorial_summary.json',dict(groups=summarize(census),
               note='2x2 IoU interaction = both(1)-center_only-size_only+actual; GT oracles, not AP.'))
    del census
    images,annotations,_=dataset();result=[]
    with open(output/'refinement.jsonl','x') as stream:
        for ii,i in enumerate(capture_summary['image_ids']):
            with np.load(output/'capture/traces'/f'{i}.npz') as t:
                gt=np.array([a['bbox'] for a in annotations[i]],dtype=np.float64);gt[:,2:]+=gt[:,:2]
                coarse_norm=t['coarse_cxcywh'].reshape(-1,4)
                coarse=coco_xyxy(clipped_pixels(xyxy(coarse_norm),t['resized_hw'],t['original_hw']))
                refined=coco_xyxy(clipped_pixels(t['raw_boxes'],t['resized_hw'],t['original_hw']))
                ci,ri=iou_matrix(coarse,gt),iou_matrix(refined,gt)
                paired=paired_coverage(ci,ri)
                cb,rb=paired['coarse_ids'],paired['refined_ids']
                cmax,rmax=paired['coarse_max'],paired['refined_max']
                normalized_gt=gt/np.tile([images[i]['width'],images[i]['height']],2)
                scale=float(t['reg_scale']);bins=t['support_bins'].astype(np.float64)
                logits=t['corner_logits'].reshape(-1,4,len(bins)).astype(np.float64)
                probs=np.exp(logits-logits.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
                entropy=-(probs*np.log(np.maximum(probs,1e-300))).sum(-1)/np.log(len(bins))
                offsets=(probs*bins).sum(-1)
                decoded=t['decoded_cxcywh'].reshape(-1,4)
                decoded_pixels=coco_xyxy(clipped_pixels(xyxy(decoded),t['resized_hw'],t['original_hw']))
                pair_l=np.empty(len(gt));pair_u=np.empty(len(gt))
                for g,j in enumerate(cb):
                    low,high=support_bounds(coarse_norm[j:j+1],normalized_gt[g:g+1],bins.min(),bins.max(),scale)
                    pair_l[g],pair_u[g]=low[0,0],high[0,0]
                fail=np.flatnonzero(rmax<.75)
                reachable={};limited={}
                # Fixed observed references: counterfactual head capacity, NOT rerun-flow AP.
                for start in range(0,len(fail),24):
                    fs=fail[start:start+24]
                    low,high=support_bounds(coarse_norm,normalized_gt[fs],bins.min(),bins.max(),scale)
                    for g,l,h in zip(fs,low.max(0),high.max(0)):
                        reachable[int(g)]=bool(l>.75001);limited[int(g)]=bool(h<.74999)
                for g,a in enumerate(annotations[i]):
                    j=int(cb[g]);r=base_record(a,i,images[i])
                    ref=coarse_norm[j].astype(np.float64);target=normalized_gt[g]
                    needed=np.r_[(ref[:2]-target[:2])/(ref[2:]/scale)-scale/2,
                                 (target[2:]-ref[:2])/(ref[2:]/scale)-scale/2]
                    pred_shift=np.abs(refined[j]-coarse[j]).mean()
                    required=np.abs(gt[g]-coarse[j]).mean()
                    r.update(raw_fail=bool(rmax[g]<.75),best_iou=float(rmax[g]),
                        coarse_max_iou=float(cmax[g]),refined_max_iou=float(rmax[g]),
                        coarse_anchor_step=j//900,coarse_anchor_query=j%900,
                        coarse_anchor_delta=float(paired['coarse_delta'][g]),refined_anchor_delta=float(paired['refined_delta'][g]),
                        coarse_covers75=bool(cmax[g]>=.75),refined_covers75=bool(rmax[g]>=.75),
                        coverage_rescued=bool(cmax[g]<.75<=rmax[g]),coverage_damaged=bool(rmax[g]<.75<=cmax[g]),
                        paired_rescued=bool(ci[j,g]<.75<=ri[j,g]),paired_damaged=bool(ri[j,g]<.75<=ci[j,g]),
                        predicted_shift_px=float(pred_shift),required_shift_px=float(required),
                        distribution_entropy=float(entropy[j].mean()),extreme_mass=float(probs[j][:,[0,-1]].sum(-1).mean()),
                        exact_gt_outside=bool((needed<bins.min()).any() or (needed>bins.max()).any()),
                        paired_support_lower=float(pair_l[g]),paired_support_upper=float(pair_u[g]),
                        predicted_offset_abs_max=float(np.abs(offsets[j]).max()),
                        required_offset_abs_max=float(np.abs(needed).max()),
                        offset_near_support_edge=bool(((offsets[j]-bins.min())<.05*(bins.max()-bins.min())).any()
                            or ((bins.max()-offsets[j])<.05*(bins.max()-bins.min())).any()),
                        sanitize_iou_delta=float(ri[j,g]-pair_iou(decoded_pixels[j],gt[g])),
                        sanitize_shift_px=float(np.abs(refined[j]-decoded_pixels[j]).mean()),
                        sanitize_changes=bool(not np.array_equal(t['decoded_cxcywh'].reshape(-1,4)[j],t['refined_cxcywh'].reshape(-1,4)[j])))
                    if g in reachable:r.update(all_query_support_reachable=reachable[g],all_query_support_limited=limited[g],
                                               all_query_support_unresolved=not(reachable[g] or limited[g]))
                    stream.write(json.dumps(r,allow_nan=False)+'\n');result.append(r)
            if (ii+1)%50==0:print(f'ANALYZE {ii+1}/{capture_summary["images"]}',flush=True)
    write_json(output/'refinement_summary.json',dict(status='complete',images=capture_summary['images'],gt_count=len(result),
               groups=summarize(result),scope='Diagnostic stratified subset, not validation-wide AP. Coarse/refined anchors are selection-conditioned; max coverage is symmetric set statistic.',
               support_note='Continuous support given observed coarse references; strict interior feasible witness / relaxed upper; IoU margin1e-5. No model change.'))
    check_sources();print('ANALYSIS_DONE',len(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','smoke','capture','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke-tag',default='smoke')
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.output)
    elif args.phase=='analyze':analyze(args.output)
    else:capture(args.output,args.phase=='smoke',args.smoke_tag)
