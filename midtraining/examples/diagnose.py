"""Read trusted lab checkpoints; add fixed, answer-region diagnostics without training."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
from torch.nn import functional as F
import lab

@torch.no_grad()
def inspect(model,rows):
    result={}
    for source in ['general','domain']:
        probes=[r for r in rows if r['source']==source and r['split']=='val'][:4]
        details=[];prefix_sum=answer_sum=0.;prefix_n=answer_n=0
        for row in probes:
            full=list(row['text'].encode())+[lab.EOS]
            x=torch.tensor([full[:-1]]);y=torch.tensor(full[1:])
            losses=F.cross_entropy(model(x)[0],y,reduction='none')
            boundary=len(row['prefix'].encode())-1
            prefix_sum+=float(losses[:boundary].sum());prefix_n+=boundary
            answer_sum+=float(losses[boundary:].sum());answer_n+=len(losses)-boundary
            candidates=[]
            for answer in [r['answer'] for r in probes]:
                seq=list((row['prefix']+answer).encode())+[lab.EOS]
                scores=model(torch.tensor([seq[:-1]]))[0].log_softmax(-1)
                targets=torch.tensor(seq[1:])
                logp=float(scores.gather(-1,targets[:,None]).squeeze(-1)[boundary:].sum())
                candidates.append(dict(answer=answer,logp=logp))
            chosen=max(candidates,key=lambda x:x['logp'])['answer']
            details.append(dict(id=row['id'],expected=row['answer'],answer_targets=len(losses)-boundary,
                                answer_nll=float(losses[boundary:].mean()),candidate_choice=chosen,
                                candidate_correct=chosen==row['answer'],candidates=candidates))
        result[source]=dict(prefix_nll=prefix_sum/prefix_n,prefix_targets=prefix_n,
                            answer_including_eos_nll=answer_sum/answer_n,answer_targets=answer_n,
                            candidate_correct=sum(x['candidate_correct'] for x in details),probes=details)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    source=Path(args.run);reference=json.loads((source/'report.json').read_text(encoding='utf8'))
    rows=lab.corpus();assert lab.digest(rows)==reference['data_sha256']
    assert lab.code_identity()==reference['code_sha256']
    report=dict(protocol='Added after original experiments: fixed first four held-out IDs per source; full document teacher forcing; prefix vs answer+EOS and four-candidate raw sequence logp. No tuning or new training.',
                code_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf8').encode()).hexdigest(),
                lab_sha256=lab.code_identity(),data_sha256=lab.digest(rows),
                environment=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',threads=1),blocks=[])
    for block in reference['blocks']:
        results={};seed=block['seed']
        for name,record in [('base',block['base']),*block['arms'].items()]:
            saved=torch.load(source/f'{name}-{seed}.pt',map_location='cpu',weights_only=False)
            assert saved['contract']['code_sha256']==lab.code_identity()
            model=lab.TinyBase().eval();model.load_state_dict(saved['model'])
            assert lab.parameter_identity(model)==record['final_identity']
            results[name]=dict(identity=record['final_identity'],regions=inspect(model,rows),
                               original_greedy_correct=sum(x['exact'] for x in record['final']['generation']))
        report['blocks'].append(dict(seed=seed,models=results))
    report['scope']='Post-hoc diagnostics on shared-template probes. Candidate ranking differs from free generation; not independent test data, new training evidence or a unique cause of failure.'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',blocks=len(report['blocks']),models=8,out=str(out))))

if __name__=='__main__':main()
