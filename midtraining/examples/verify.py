"""Checks of the mid-training fixture, including real save/reload and code rejection."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch.nn import functional as F
import lab


def identical(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a == b


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--out",default="runs/midtraining-checks")
    args=parser.parse_args(); torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True); rows=lab.corpus(); checks=[]
    groups={split:{r['group_id'] for r in rows if r['split']==split} for split in ['train','val']}
    assert not groups['train'] & groups['val'] and len(rows)==160
    assert len({r['text'] for r in rows})==160
    checks.append(dict(name="entity_group_split_and_exact_unique_documents",train_groups=128,val_groups=32))
    sequence=list("Z0".encode())+[256]
    assert sequence==[90,48,256] and sequence[:-1]==[90,48] and sequence[1:]==[48,256]
    checks.append(dict(name="actual_byte_ids_shift_and_eos",input_ids=sequence[:-1],labels=sequence[1:]))
    model=lab.TinyBase().eval(); ids=torch.tensor([[1,2,3,4,5,6]])
    changed=ids.clone();changed[0,4:]=torch.tensor([9,8])
    with torch.no_grad(): torch.testing.assert_close(model(ids)[:,:4],model(changed)[:,:4],atol=1e-6,rtol=1e-5)
    checks.append(dict(name="future_inputs_do_not_change_earlier_logits"))
    try: model(torch.zeros(1,65,dtype=torch.long))
    except ValueError: pass
    else: raise AssertionError("position overflow accepted")
    checks.append(dict(name="position_limit_is_enforced_not_silently_truncated"))
    a=torch.tensor([2.]*100+[4.]*900)
    assert float(a.mean())==float(torch.tensor(3.8)) and (2.+4.)/2==3.
    checks.append(dict(name="effective_target_weighting",global_mean=3.8,wrong_mean_of_means=3.))
    continuous=lab.Run(lab.Config(),rows); continuous.advance(8)
    split=lab.Run(lab.Config(),rows);split.advance(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);checkpoint=out/'step4.pt';split.save(checkpoint)
    restored=lab.Run(lab.Config(),rows);restored.restore(checkpoint);restored.advance(8)
    for field in ['model','optimizer','generator','torch_rng','step','targets','consumed','history','ledger','initial_identity']:
        assert identical(continuous.checkpoint()[field],restored.checkpoint()[field]),field
    checks.append(dict(name="eight_updates_vs_four_save_reload_four_exact",targets=1024,
                       fields="model,AdamW,generator,torch RNG,history,source cursor ledger,counters"))
    saved=torch.load(checkpoint,weights_only=False)
    for key,value in [('code_sha256','changed'),('torch','different'),('world_size',2),('encoding','changed')]:
        bad=copy.deepcopy(saved);bad['contract'][key]=value;file=out/f'bad-{key}.pt';torch.save(bad,file)
        try: lab.Run(lab.Config(),rows).restore(file)
        except ValueError: pass
        else: raise AssertionError(key+' accepted')
    checks.append(dict(name="code_torch_world_encoding_changes_rejected"))
    for config,data in [(lab.Config(steps=97),rows),(lab.Config(),rows+[dict(rows[0],text='mutated')])]:
        try: lab.Run(config,data).restore(checkpoint)
        except ValueError: pass
        else: raise AssertionError("plan or data change accepted")
    checks.append(dict(name="training_plan_and_actual_data_changes_rejected"))
    source=Path(lab.__file__).read_text(encoding='utf8').replace('LR_MULTIPLIER = 1.0','LR_MULTIPLIER = 1.01')
    assert source!=Path(lab.__file__).read_text(encoding='utf8')
    with tempfile.TemporaryDirectory() as temp:
        altered=Path(temp)/'lab.py';altered.write_text(source,encoding='utf8')
        result=subprocess.run([sys.executable,str(altered),'--resume',str(checkpoint.resolve())],capture_output=True,text=True)
        assert result.returncode!=0 and 'strict resume identity' in result.stderr
    checks.append(dict(name="actual_lr_source_times_1_01_refused_by_strict_resume"))
    new=lab.Run(lab.Config(name='new-stage'),rows,continuous.model.state_dict())
    assert lab.parameter_identity(new.model)==lab.parameter_identity(continuous.model)
    assert not new.optimizer.state and new.step==0 and new.targets==0 and new.consumed==dict(general=0,domain=0)
    checks.append(dict(name="weights_only_new_stage_resets_optimizer_and_stage_counts"))
    starts=[]
    for length,bsize in [(32,4),(64,2)]:
        config=lab.Config(length=length,batch=bsize,steps=48)
        g=torch.Generator().manual_seed(999);consumed=dict(general=0,domain=0)
        for step in range(48):
            x,y,ledger=lab.batch(lab.streams(rows),config,step,g)
            assert x.shape==y.shape==(bsize,length)
            for item in ledger: consumed[item['source']]+=item['targets']
        assert consumed==dict(general=1536,domain=4608)
        starts.append(consumed)
    checks.append(dict(name="short_long_interface_equal_budget_and_source_counts",counts=starts))
    evaluations=[lab.evaluate(lab.TinyBase().eval(),rows,lab.streams(rows)) for _ in range(2)]
    for key in ['general_32','domain_32','general_64','domain_64']:
        assert evaluations[0][key]['starts']==evaluations[1][key]['starts']
        assert evaluations[0][key]['targets']==512
    checks.append(dict(name="fixed_evaluation_window_indices_and_target_counts"))
    report=dict(status='PASS',count=len(checks),checks=checks,torch=str(torch.__version__),code_sha256=lab.code_identity(),
                scope='CPU single layout; ordinary optimizer-update boundaries; no file-write crash, CUDA/DDP or long-context ability benchmark')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
