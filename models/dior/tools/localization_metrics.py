"""CPU-only diagnostic geometry. Oracle boxes are not predicted detections/AP."""
from collections import defaultdict
import hashlib
import numpy as np


def pair_iou(a, b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    intersection=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1)
    aa=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)
    bb=np.maximum(b[...,2:]-b[...,:2],0).prod(-1)
    return intersection/np.maximum(aa+bb-intersection,1e-30)


def xyxy(c):
    c=np.asarray(c)
    return np.concatenate((c[...,:2]-c[...,2:]/2,c[...,:2]+c[...,2:]/2),axis=-1)


def residual_features(box,gt):
    box,gt=np.asarray(box),np.asarray(gt)
    d=(box[...,:2]+box[...,2:]-gt[...,:2]-gt[...,2:])/2
    size=np.maximum(gt[...,2:]-gt[...,:2],1e-12)
    ratio=np.maximum(box[...,2:]-box[...,:2],1e-12)/size
    return dict(center_l2_px=np.linalg.norm(d,axis=-1),
                center_relative=np.linalg.norm(d/size,axis=-1),
                abs_log_size=np.abs(np.log(ratio)).mean(-1),
                width_ratio=ratio[...,0],height_ratio=ratio[...,1],
                edge_mae_px=np.abs(box-gt).mean(-1))


def center_size_oracles(box,gt):
    box,gt=np.asarray(box),np.asarray(gt)
    pc,ps=(box[...,:2]+box[...,2:])/2,box[...,2:]-box[...,:2]
    gc,gs=(gt[...,:2]+gt[...,2:])/2,gt[...,2:]-gt[...,:2]
    return dict(actual=pair_iou(box,gt),center_only=pair_iou(xyxy(np.concatenate((gc,ps),-1)),gt),
                size_only=pair_iou(xyxy(np.concatenate((pc,gs),-1)),gt),
                both=pair_iou(gt,gt))


def paired_coverage(coarse_iou,refined_iou):
    """Two selection-conditioned anchors, distinct from symmetric set maxima."""
    c,r=np.asarray(coarse_iou),np.asarray(refined_iou)
    cb,rb=c.argmax(0),r.argmax(0);g=np.arange(c.shape[1])
    return dict(coarse_ids=cb,refined_ids=rb,coarse_max=c[cb,g],refined_max=r[rb,g],
                coarse_delta=r[cb,g]-c[cb,g],refined_delta=r[rb,g]-c[rb,g])


def _axis(l,r):
    mid=np.clip((l+r)/2,1e-4,1-1e-4)
    size=np.clip(r-l,1e-4,1)
    return np.clip(mid-size/2,0,1),np.clip(mid+size/2,0,1)


def _envelope(l0,l1,r0,r1):
    # Exact marginal sanitized/clipped edge extrema. Remaining axis is evaluated
    # at endpoints and every center/width clipping breakpoint.
    def left(l):
        rs=np.stack([r0,r1,np.clip(2e-4-l,r0,r1),np.clip(2*(1-1e-4)-l,r0,r1),
                     np.clip(l+1e-4,r0,r1),np.clip(l+1,r0,r1)])
        return _axis(l,rs)[0]
    def right(r):
        ls=np.stack([l0,l1,np.clip(2e-4-r,l0,l1),np.clip(2*(1-1e-4)-r,l0,l1),
                     np.clip(r-1e-4,l0,l1),np.clip(r-1,l0,l1)])
        return _axis(ls,r)[1]
    return left(l0).min(0),left(l1).max(0),right(r0).min(0),right(r1).max(0)


def support_bounds(coarse, gt, low, high, scale):
    """Return interior feasible-witness IoU and certified relaxed upper bound.

    coarse N,cxcywh; gt G,xyxy; all normalized to unpadded image axes. Bounds
    include actual center/size sanitization and final clipping. Upper bound
    relaxes dependencies between final edges, never proves attainability.
    Lower witness stays 1e-6 of support span inside endpoints (finite logits).
    Both are real arithmetic; caller uses >=1e-5 IoU boundary margin for FP32.
    """
    c=np.asarray(coarse,dtype=np.float64)
    g=np.asarray(gt,dtype=np.float64)
    if (c.ndim!=2 or g.ndim!=2 or c.shape[1]!=4 or g.shape[1]!=4 or not low<high or scale<=0
            or not np.isfinite([low,high,scale]).all()):
        raise ValueError('invalid support geometry')
    if not (np.isfinite(c).all() and np.isfinite(g).all() and (c[:,2:]>0).all()
            and (g[:,2:]>g[:,:2]).all()):
        raise ValueError('nonfinite or invalid box')
    center,width=c[:,:2],c[:,2:]
    l0=center-(.5+high/scale)*width
    l1=center-(.5+low/scale)*width
    r0=center+(.5+low/scale)*width
    r1=center+(.5+high/scale)*width
    lm,lM,rm,rM=_envelope(l0,l1,r0,r1)
    upper_box=np.concatenate((np.clip(g[None,:,:2],lm[:,None],lM[:,None]),
                              np.clip(g[None,:,2:],rm[:,None],rM[:,None])),axis=-1)
    margin=(high-low)*1e-6/scale*width
    wl=np.clip(g[None,:,:2],(l0+margin)[:,None],(l1-margin)[:,None])
    wr=np.clip(g[None,:,2:],(r0+margin)[:,None],(r1-margin)[:,None])
    sl,sr=_axis(wl,wr)
    lower=pair_iou(np.concatenate((sl,sr),-1),g[None])
    upper=pair_iou(upper_box,g[None])
    if np.any(lower>upper+1e-9):
        raise RuntimeError('support lower exceeds upper bound')
    return lower,upper


def choose_sample(rows,images,limit=400):
    seed='localization-20260918'
    selected=[]; seen=set()
    def add(image_id,group,anchor=None):
        if image_id in seen or len(selected)>=limit:return False
        seen.add(image_id); selected.append(dict(image_id=image_id,group=group,anchor=anchor)); return True
    for i in sorted(images):
        if (images[i]['width'],images[i]['height'])!=(800,800): add(i,'non_square')
    def rank(value): return hashlib.sha256(value.encode()).hexdigest()
    for i in sorted(images,key=lambda i:rank(f'{seed}:uniform:{i}')):
        if len(selected)>=min(40,limit):break
        add(i,'uniform')
    cells=defaultdict(list)
    for r in rows:
        if not r['small'] or r['category_id'] not in (19,14,16):continue
        e=r['thresholds']['0.75']
        outcome='raw_fail' if not e['covered']['raw_geometry'] else 'matched' if e['official']['small']['matched'] else 'selection_loss'
        sb=int(np.searchsorted([8,12,20],min(r['bbox'][2:]),side='right'))
        db=int(np.searchsorted([10,50],images[r['image_id']]['gt_count'],side='left'))
        key=(r['category_id'],outcome,sb,db)
        cells[key].append((rank(f"{seed}:{key[0]}:{outcome}:{sb}:{db}:{r['image_id']}:{r['annotation_id']}"),r))
    for values in cells.values():values.sort(key=lambda t:t[0],reverse=True)
    for category in (19,14,16):
        for outcome in ('raw_fail','matched','selection_loss'):
            quota=80 if category==19 else 20
            count=0
            while count<quota and len(selected)<limit:
                progress=False
                for sb in range(4):
                    for db in range(3):
                        values=cells[(category,outcome,sb,db)]
                        while values:
                            r=values.pop()[1]
                            if add(r['image_id'],f'{category}/{outcome}/{sb}/{db}',r['annotation_id']):
                                count+=1;progress=True;break
                        if count>=quota or len(selected)>=limit:break
                    if count>=quota or len(selected)>=limit:break
                if not progress:break
    for i in sorted(images,key=lambda i:rank(f'{seed}:fallback:{i}')):
        if len(selected)>=limit:break
        add(i,'fallback')
    return selected


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        size='small' if r['small'] else 'larger'
        for key in ('all',size,f"{size}/{r['category_id']}",
                    f"{size}/{r['category_id']}/short{r['short_bin']}/density{r['density_bin']}"):
            groups[key].append(r)
        if r['raw_fail']: groups[f"{size}/{r['category_id']}/raw_fail"].append(r)
    result={}
    for k,rs in groups.items():
        v=dict(n=len(rs),raw_fail=sum(r['raw_fail'] for r in rs))
        numeric=('best_iou','center_l2_px','center_relative','abs_log_size','edge_mae_px',
                 'width_ratio','height_ratio','center_only_iou','size_only_iou',
                 'coarse_max_iou','refined_max_iou','coarse_anchor_delta','refined_anchor_delta',
                 'predicted_shift_px','required_shift_px','distribution_entropy','extreme_mass',
                 'paired_support_lower','paired_support_upper','center_size_interaction',
                 'sanitize_iou_delta','sanitize_shift_px','predicted_offset_abs_max','required_offset_abs_max')
        for name in numeric:
            vals=[r[name] for r in rs if name in r]
            if vals:v[name]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)),p90=float(np.quantile(vals,.9)))
        flags=('center_only_recovers','size_only_recovers','coarse_covers75','refined_covers75',
               'coverage_rescued','coverage_damaged','paired_rescued','paired_damaged',
               'exact_gt_outside','all_query_support_reachable','all_query_support_limited',
               'all_query_support_unresolved','sanitize_changes','gt_has_cross_class_best','offset_near_support_edge')
        for name in flags:
            vals=[r[name] for r in rs if name in r]
            if vals:v[name]=dict(yes=sum(vals),n=len(vals))
        result[k]=v
    return result
