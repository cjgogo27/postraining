"""CPU operator trace with explicit phases; not a GPU throughput/MFU benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
from torch.nn import functional as F
import lab

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    run=lab.Run(lab.Config(seed=42,steps=6),lab.corpus())
    def update(index):
        with torch.profiler.record_function('mid/data'):
            x,y,_=lab.batch(run.data,run.config,index,run.generator)
        run.optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function('mid/forward_loss'):
            loss=F.cross_entropy(run.model(x).reshape(-1,lab.VOCAB),y.reshape(-1))
        with torch.profiler.record_function('mid/backward'):loss.backward()
        with torch.profiler.record_function('mid/clip_update'):
            torch.nn.utils.clip_grad_norm_(run.model.parameters(),1.);run.optimizer.step()
        assert torch.isfinite(loss)
    for index in range(2):update(index)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
        for index in range(2,6):update(index);prof.step()
    prof.export_chrome_trace(str(out/'trace.json'))
    events=[]
    for event in prof.key_averages():
        if event.key.startswith('mid/'):
            events.append(dict(name=event.key,calls=event.count,cpu_total_us=event.cpu_time_total,self_cpu_us=event.self_cpu_time_total))
    assert len(events)==4 and all(e['calls']==4 for e in events)
    report=dict(status='PASS',code_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest(),
                lab_sha256=lab.code_identity(),environment=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',threads=1),
                warmup_updates=2,recorded_updates=4,recorded_targets=512,events=events,
                scope='Fresh random model, fixed lr .001, CPU-only four-update instrumented trace; no evaluation/save/I/O service/communication/GPU, no MFU, no throughput claim. Inclusive times overlap with child operators and profiler adds overhead.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
