"""Render recorded fixed-window evaluations; optional matplotlib, no retraining."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='midtraining-recorded-results'

root=Path(__file__).resolve().parents[1]
report=json.loads((root/'examples/results/lab-reference.json').read_text(encoding='utf8'))
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
colors={'general':'#2878b5','domain':'#cf563b','replay25':'#2b9565'}
for row,block in enumerate(report['blocks']):
 for col,source in enumerate(['domain','general']):
  ax=axes[row,col];key=source+'_32'
  for name,arm in block['arms'].items():
   points=[(0,block['base']['final'][key]['nll'])]+[(h['targets'],h['evaluation'][key]['nll']) for h in arm['history'] if 'evaluation' in h]
   ax.plot(*zip(*points),label=name,color=colors[name],marker='o',markersize=3)
  ax.set_title(f"seed {block['seed']} | {source} fixed S32 windows")
  ax.set_ylabel('NLL (nats / target)');ax.grid(alpha=.25)
  ax.legend(fontsize=8)
  if row==1:ax.set_xlabel('Additional training targets (repeated consumption)')
fig.suptitle('Matched budgets: 12,288 targets per arm | Greedy domain probes: all 0/4',fontsize=11)
fig.tight_layout()
out=root/'public/figures';out.mkdir(parents=True,exist_ok=True)
svg=out/'mixture-curves.svg'
fig.savefig(svg,metadata={'Date':None})
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
fig.savefig(out/'mixture-curves.png',dpi=130)
print('Rendered actual JSON curves with matplotlib '+matplotlib.__version__)
