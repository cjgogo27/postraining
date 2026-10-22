"""Independent uniform-logit hand oracle for new read-only diagnostics."""
import argparse
import json
import math
from pathlib import Path
import torch
import diagnose
import lab

class Uniform(torch.nn.Module):
    def forward(self,ids):return torch.zeros((*ids.shape,257),dtype=torch.float32)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    model=Uniform().eval();before=torch.get_rng_state().clone()
    result=diagnose.inspect(model,lab.corpus());expected=math.log(257)
    for source in ['domain','general']:
        region=result[source]
        assert region['prefix_targets']==104 and region['answer_targets']==28
        assert abs(region['prefix_nll']-expected)<1e-6
        assert abs(region['answer_including_eos_nll']-expected)<1e-6
        assert len(region['probes'])==4 and region['candidate_correct']==1
        for row in region['probes']:
            assert row['answer_targets']==7
            assert len(row['candidates'])==4
            assert all(abs(candidate['logp']+7*expected)<1e-5 for candidate in row['candidates'])
    assert torch.equal(before,torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',checks=['uniform_ce_ln257','prefix104_answer_eos28','four_equal_candidates_seven_targets','tie_chooses_first_not_free_generation','torch_rng_unchanged'],expected_nll=expected,torch=str(torch.__version__),scope='Independent constant-logit arithmetic oracle; not a pretrained-model capability test')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
