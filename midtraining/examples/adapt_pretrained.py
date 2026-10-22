"""Optional real-base causal-LM adaptation; default cache-only, CPU, no chat SFT.

Uses the original fictional corpus from lab.py; no real maintenance facts.
No resume promise. Small fixed experiment, not a production recipe.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import torch
from torch.nn import functional as F
import lab

MODEL='HuggingFaceTB/SmolLM2-135M'
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/midtraining-pretrained')
    parser.add_argument('--download',action='store_true',help='allow initial public weight/tokenizer download')
    args=parser.parse_args();out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    start=time.perf_counter()
    snapshot=snapshot_download(MODEL,revision=REVISION,local_files_only=not args.download)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32)
    rows=lab.corpus();data={}
    for source in ['general','domain']:
        for split in ['train','val']:
            ids=[]
            for row in rows:
                if row['source']==source and row['split']==split:
                    ids+=tokenizer.encode(row['text'],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(ids)<129:raise ValueError('stream too short')
            data[(source,split)]=torch.tensor(ids,dtype=torch.long)

    @torch.no_grad()
    def evaluate():
        model.eval();result={}
        for source in ['general','domain']:
            stream=data[(source,'val')];starts=torch.linspace(0,len(stream)-65,4).long().tolist()
            full=torch.stack([stream[s:s+65] for s in starts])
            logits=model(input_ids=full[:,:-1],use_cache=False).logits
            nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),full[:,1:].reshape(-1),reduction='sum')
            result[source]=dict(nll=float(nll)/256,targets=256,starts=starts,stream_sha256=lab.digest(stream.tolist()))
        outputs=[]
        for row in [r for r in rows if r['source']=='domain' and r['split']=='val'][:2]:
            tokens=tokenizer.encode(row['prefix'],add_special_tokens=False);new=[];ended=False
            for _ in range(12):
                token=int(model(input_ids=torch.tensor([tokens]),use_cache=False).logits[0,-1].argmax())
                if token==tokenizer.eos_token_id:ended=True;break
                new.append(token);tokens.append(token)
            text=tokenizer.decode(new,skip_special_tokens=False)
            outputs.append(dict(id=row['id'],prefix=row['prefix'],expected=row['answer'],output_ids=new,
                                output=text,ended_eos=ended,exact=ended and text==row['answer']))
        result['generation']=outputs;return result

    initial=copy.deepcopy(model.state_dict());before=evaluate()
    order=['general','domain','replay25'];random.Random(42).shuffle(order)
    report=dict(model=MODEL,revision=REVISION,seed=42,steps_per_arm=8,batch=2,length=64,lr=2e-5,
                objective='all shifted causal-LM targets, including EOS; no prompt/answer mask',
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),corpus_code_sha256=lab.code_identity(),
                data_sha256=lab.digest(rows),environment=dict(python=platform.python_version(),torch=str(torch.__version__),
                transformers=importlib.metadata.version('transformers'),huggingface_hub=importlib.metadata.version('huggingface-hub'),
                device='cpu',dtype='float32',threads=4),parameters=sum(p.numel() for p in model.parameters()),
                run_order=order,before=before,arms={},files={})
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors']:
        file=Path(snapshot)/name
        if file.exists():report['files'][name]=hashlib.sha256(file.read_bytes()).hexdigest()
    for name in order:
        model.load_state_dict(initial);torch.manual_seed(42);generator=torch.Generator().manual_seed(1042)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        fraction=dict(general=1.,domain=0.,replay25=.25)[name]
        history=[];consumed=dict(general=0,domain=0)
        for step in range(8):
            general=int((step+1)*2*fraction)-int(step*2*fraction)
            sources=['general']*general+['domain']*(2-general);windows=[]
            for index in torch.randperm(2,generator=generator).tolist():
                source=sources[index];stream=data[(source,'train')]
                offset=int(torch.randint(len(stream)-64,(),generator=generator))
                windows.append(stream[offset:offset+65]);consumed[source]+=64
            full=torch.stack(windows);model.train();optimizer.zero_grad(set_to_none=True)
            logits=model(input_ids=full[:,:-1],use_cache=False).logits
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),full[:,1:].reshape(-1))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step();history.append(dict(update=step+1,targets=(step+1)*128,train_nll=float(loss.detach())))
        report['arms'][name]=dict(consumed=consumed,targets=1024,history=history,final=evaluate())
        if name=='replay25':
            model.save_pretrained(out/'adapted');tokenizer.save_pretrained(out/'adapted')
        del optimizer,logits,loss;gc.collect()
    del model,initial;gc.collect()
    model=AutoModelForCausalLM.from_pretrained(out/'adapted',local_files_only=True,dtype=torch.float32)
    reloaded=evaluate();expected=report['arms']['replay25']['final']
    for source in ['general','domain']:
        assert abs(reloaded[source]['nll']-expected[source]['nll'])<1e-6
    assert reloaded['generation']==expected['generation']
    report['reload']=dict(status='PASS',generation_identical=True,nll_atol=1e-6)
    report['wall_seconds_including_load_evaluate_train_save_reload']=time.perf_counter()-start
    report['scope']='One seed, 8 updates per arm, held-out entity IDs sharing templates; no real domain/long-context/SFT/RL capability claim; no optimizer resume'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',parameters=report['parameters'],run_order=order,elapsed=report['wall_seconds_including_load_evaluate_train_save_reload'])))


if __name__=='__main__':main()
