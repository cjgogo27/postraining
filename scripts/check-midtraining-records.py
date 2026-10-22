"""Check published experiment provenance/counts without retraining or torch imports."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'midtraining'
def read(name):return json.loads((root/'examples/results'/name).read_text(encoding='utf8'))
def identity(path):return hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()
lab=read('lab-reference.json');pre=read('pretrained-reference.json');checks=read('checks-reference.json');corpus=read('corpus-reference.json')
assert identity(root/'examples/lab.py')==lab['code_sha256']==checks['code_sha256']==pre['corpus_code_sha256']
assert identity(root/'examples/adapt_pretrained.py')==pre['code_sha256']
assert hashlib.sha256(json.dumps(corpus,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==lab['data_sha256']==pre['data_sha256']
assert len(corpus)==160 and len({x['text'] for x in corpus})==160
assert checks['status']=='PASS' and checks['count']==12
for block in lab['blocks']:
 assert block['base']['targets']==15360
 for name,arm in block['arms'].items():
  assert arm['initial_identity']==block['base']['final_identity']
  assert arm['targets']==12288==sum(arm['consumed'].values())
  if name=='replay25':assert arm['consumed']==dict(general=3072,domain=9216)
  assert all(not x['exact'] for x in arm['final']['generation'])
  for source in ['general','domain']:
   key=source+'_32';assert arm['final'][key]['targets']==512
   assert arm['final'][key]['starts']==block['base']['final'][key]['starts']
for arm in lab['length_interface']:
 assert arm['targets']==6144 and arm['consumed']==dict(general=1536,domain=4608)
 assert arm['initial_identity']==lab['blocks'][0]['arms']['replay25']['final_identity']
assert pre['parameters']==134515008
print(json.dumps(dict(status='PASS',scope='published source/data identities, matched starts/budgets, stage inheritance and preserved failures; no retraining')))
