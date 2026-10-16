"""Reload the optional saved model locally and compare fixed outputs/NLL to report."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pretrained_sft import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',default='runs/pretrained-artifact-checks')
    args=parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root=Path(args.run)
    report=json.loads((root/'report.json').read_text(encoding='utf8'))
    torch.set_num_threads(report['config']['threads'])
    torch.use_deterministic_algorithms(True)
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,dtype=torch.float32,attn_implementation='eager')
    tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
    assert sum(p.numel() for p in model.parameters())==report['parameters']
    assert hashlib.sha256((Path(__file__).parent/'pretrained_sft.py').read_bytes()).hexdigest()==report['code_sha256']
    for split in ('target','retention'):
        actual=evaluate(model,tokenizer,report['data'][split]);expected=report['after'][split]
        assert abs(actual['response_nll']-expected['response_nll'])<1e-6
        assert actual['response_positions']==expected['response_positions']
        for a,b in zip(actual['outputs'],expected['outputs'],strict=True):
            for key in ('id','generated','generated_ids','end_reason','content_exact','strict_success'):
                assert a[key]==b[key],(split,key,a,b)
    result=dict(status='PASS',parameters=report['parameters'],torch=str(torch.__version__),
                checked=['training_source_identity','saved_parameter_count','all_9_fixed_outputs_and_stop_reasons','both_nll_within_1e-6'],
                scope='Same CPU environment; local saved model reload, not training resume or cross-hardware equality')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__': main()
