"""Four-image backward-only AMP observation; no optimizer, EMA or checkpoint writes.

Each fixed policy runs in a separate process. A first observed nonfinite
boundary is an observation, never proof of the causal origin.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/root/autodl-tmp/exps/geoobs_amp_20260927_0055')
POLICIES = {'amp65536':65536,'amp4':4,'fp32':1}
FIXTURE_SHA = '48dde79f6e80d48a30a421a381b695c7a944f0e0730c980a0c83561be2b723b5'

# Load by file path before any project imports, preserving strict bootstrap.
spec = importlib.util.spec_from_file_location('geoobs_amp_verifier',ROOT/'tools/verify_geo_observation.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def summarize_tensor(tensor):
    value=tensor.detach()
    finite=torch.isfinite(value)
    selected=value[finite]
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'finite_count':int(finite.sum()),'nan_count':int(torch.isnan(value).sum()),
            'positive_inf_count':int(torch.isposinf(value).sum()),
            'negative_inf_count':int(torch.isneginf(value).sum()),
            'finite_abs_max':float(selected.abs().max()) if selected.numel() else None}


def nonfinite(summary):
    return summary['nan_count']+summary['positive_inf_count']+summary['negative_inf_count'] > 0


def accumulated_gradients(model):
    summaries={name:summarize_tensor(p.grad) for name,p in model.named_parameters() if p.grad is not None}
    if not summaries:
        raise ValueError('backward disconnected from all model parameters')
    return summaries


def tensor_leaves(value,path):
    if torch.is_tensor(value):
        yield path,value
    elif isinstance(value,dict):
        for key,item in value.items(): yield from tensor_leaves(item,f'{path}.{key}')
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value): yield from tensor_leaves(item,f'{path}[{index}]')


class GradientObserver:
    def __init__(self,model):
        self.handles=[]
        self.events=[]
        self.event_count=0
        self.first_nonfinite=None
        self.microstep=None
        for name,module in model.named_modules():
            if not list(module.children()):
                def forward(module,args,output,name=name):
                    for path,tensor in tensor_leaves(output,name+'.output'):
                        if tensor.requires_grad:
                            self.handles.append(tensor.register_hook(self.hook(path,'output_gradient')))
                self.handles.append(module.register_forward_hook(forward))
        for name,parameter in model.named_parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(self.hook(name,'parameter_incoming')))

    def hook(self,path,kind):
        def observe(gradient):
            summary=summarize_tensor(gradient)
            if nonfinite(summary):
                event={'microstep':self.microstep,'kind':kind,'path':path,'summary':summary}
                self.event_count+=1
                if self.first_nonfinite is None: self.first_nonfinite=event
                if len(self.events)<48: self.events.append(event)
            return None
        return observe

    def remove(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()


def reserve_policy(root,policy):
    if policy not in POLICIES: raise ValueError('fixed diagnostic policy required')
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    work=root/policy
    work.mkdir(exist_ok=False)
    return work


def authenticate_prior_source(current,previous):
    exception='tools/verify_geo_observation.py'
    if set(current)!=set(previous): raise ValueError('prior source manifest keys changed')
    for path,record in current.items():
        if path!=exception and record!=previous[path]:
            raise ValueError('prior source drift: '+path)
    return {'previous_verifier_sha256':previous[exception]['sha256'],
            'current_verifier_sha256':current[exception]['sha256'],
            'explicit_exception':'verifier initial-scale support and finite updated-tensor checks only; review required'}


def run(args,report):
    cfg,instantiate,trainer,helpers,provenance=verifier.bootstrap(ROOT,False)
    report['source']=provenance
    prior=verifier.require_pass(verifier.OUTPUT,'prepare')
    report['prior_source_authentication']=authenticate_prior_source(provenance,prior['source'])
    report['prior_reports']={'prepare':str(verifier.OUTPUT/'prepare/report.json'),
                             'checks':[str(verifier.OUTPUT/'check'/mode/'report.json') for mode in verifier.MODES]}
    for mode in verifier.MODES:
        checked=verifier.require_pass(verifier.OUTPUT,'check',mode)
        if checked['source']!=prior['source']: raise ValueError('prior check source mismatch')
    if prior['fixture_sha256']!=FIXTURE_SHA: raise ValueError('pinned fixture mismatch')
    if verifier.sha256(verifier.CHECKPOINT)!=verifier.CHECKPOINT_SHA: raise ValueError('checkpoint SHA mismatch')
    report['fixture_sha256']=FIXTURE_SHA
    data=verifier.fixtures(verifier.OUTPUT)
    model,initial,loading=verifier.construct(cfg,'off',helpers)
    if any(name.startswith(verifier.PREFIX) for name in verifier.named_tensors(model)):
        raise ValueError('off model unexpectedly contains observation tensors')
    before=verifier.state_signature(model)
    report['loading']=loading
    report['loaded_state_digest']=verifier.digest(before)
    model.train()
    model.zero_grad(set_to_none=True)
    helpers['verify_proposal_flow'].seed_all(20260926)
    report['initial_rng_digest']=verifier.digest(verifier.rng_state())
    scaler=torch.cuda.amp.GradScaler(init_scale=POLICIES[args.policy]) if args.policy!='fp32' else None
    if scaler is not None and scaler.get_scale()!=POLICIES[args.policy]:
        raise ValueError('actual initial scaler differs from fixed policy')
    observer=GradientObserver(model)
    from detectron2.utils.events import EventStorage
    try:
        with EventStorage(0):
            for index in range(4):
                sample=copy.deepcopy(data['updates'][index])
                expected=data['manifest'][index]['digest']
                if verifier.digest(sample)!=expected: raise ValueError('initial full input mismatch')
                observer.microstep=index
                report['consumed_microimages']+=1
                with torch.cuda.amp.autocast(enabled=args.policy!='fp32'):
                    losses=model([sample])
                    total=sum(losses.values())
                if not isinstance(losses,dict) or not losses: raise ValueError('loss mapping required')
                loss_values={}
                for name,value in losses.items():
                    if value.numel()!=1 or not torch.isfinite(value).all(): raise ValueError('nonfinite/non-scalar forward loss '+name)
                    loss_values[name]=float(value.detach())
                objective=total/4
                if not torch.isfinite(objective).all(): raise ValueError('nonfinite forward objective')
                (scaler.scale(objective) if scaler is not None else objective).backward()
                if verifier.digest(sample)!=expected: raise ValueError('model mutated full input')
                gradients=accumulated_gradients(model)
                report['microsteps'].append({'index':index,'input_digest':expected,'losses':loss_values,
                                             'accumulated_parameter_gradients_before_clipping':gradients})
    finally:
        observer.remove()
        report['first_observed_nonfinite_backward_boundary']=observer.first_nonfinite
        report['nonfinite_events']=observer.events
        report['nonfinite_event_count']=observer.event_count
        report['final_rng_digest']=verifier.digest(verifier.rng_state())
        report['state_unchanged']=before==verifier.state_signature(model)
        if not report['state_unchanged']: raise ValueError('complete model state changed during backward-only diagnostic')
    report['all_losses_finite']=True
    report['all_gradients_finite']=all(not nonfinite(s) for row in report['microsteps'] for s in row['accumulated_parameter_gradients_before_clipping'].values()) and observer.event_count==0
    report['pass']=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True,choices=tuple(POLICIES))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(argv)
    report={'policy':args.policy,'scale':POLICIES[args.policy],'pass':False,'consumed_microimages':0,
            'optimizer_updates':0,'ema_updates':0,'checkpoint_writes':0,'microsteps':[],
            'source_checkpoint_sha256':verifier.CHECKPOINT_SHA,'seed':20260926,
            'observation_scope':'first OBSERVED boundary, not proved causal root'}
    work=None
    try:
        if not __debug__: raise ValueError('python -O forbidden')
        if args.output.resolve()!=OUTPUT: raise ValueError('exact fresh diagnostic output root required')
        work=reserve_policy(args.output,args.policy)
        lock=args.output/'active.lock'
        with lock.open('x') as stream: stream.write(f'{os.getpid()} {args.policy}\n')
        try:
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            report['numeric_flags']={'cudnn_benchmark':False,'cudnn_deterministic':True,'matmul_tf32':False,'cudnn_tf32':False}
            run(args,report)
        finally: lock.unlink()
    except Exception as exc:
        report.update({'pass':False,'reason':f'{type(exc).__name__}: {exc}'})
    if work is not None: verifier.write_json(work/'report.json',report)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report['pass'] else 1


if __name__=='__main__': raise SystemExit(main())
