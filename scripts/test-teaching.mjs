import test from 'node:test';
import assert from 'node:assert/strict';
import {dpoMetrics,dpoPresets,byteAlignment,groupStep} from '../src/lib/objectives.mjs';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('DPO hand value and unequal candidates at current=reference',()=>{
  close(dpoMetrics(...dpoPresets.hand.values).loss,.5981388693815918);
  close(dpoMetrics(...dpoPresets.equal.values).loss,Math.log(2));
});
test('DPO margin can improve as both absolute probabilities fall',()=>{
  const [pc,pr,rc,rr,beta]=dpoPresets.bothDown.values;
  assert.ok(pc<rc && pr<rr);
  close(dpoMetrics(pc,pr,rc,rr,beta).margin,.1*Math.log(5));
});
test('DPO rejects invalid and overflowing inputs, stable large margin',()=>{
  for(const values of [[1,-1,-1,-1,1],[-1,-1,-1,-1,0],[NaN,-1,-1,-1,1],[-1e308,0,0,-1e308,2]]) assert.throws(()=>dpoMetrics(...values));
  close(dpoMetrics(0,-10000,0,0,1).loss,0);
});
test('Real byte alignment and only two response targets including EOS',()=>{
  const rows=byteAlignment();
  assert.deepEqual(rows.map(x=>x.input),[81,58,50,43,51,10,65,58,53]);
  assert.deepEqual(rows.map(x=>x.label),[58,50,43,51,10,65,58,53,256]);
  assert.deepEqual(rows.filter(x=>x.supervised).map(x=>x.index),[7,8]);
});
test('Group population std, gradient and one-update probabilities',()=>{
  const x=groupStep([0,0,1,1]);
  close(x.mean,.5);close(x.std,.5);
  assert.deepEqual(x.advantages,[-1,-1,1,1]);
  assert.deepEqual(x.gradients,[.25,.25,-.25,-.25]);
  close(x.probabilities.reduce((a,b)=>a+b,0),1);
  close(x.probabilities[2],1/(2+2*Math.exp(-.2)));
});
test('Both all-wrong and all-right groups have no policy update',()=>{
  for(const rewards of [[0,0,0,0],[1,1,1,1]]) {
    const x=groupStep(rewards); assert.deepEqual(x.advantages,[0,0,0,0]);
    assert.deepEqual(x.probabilities,[.25,.25,.25,.25]);
  }
  assert.throws(()=>groupStep([0,1])); assert.throws(()=>groupStep([NaN,0,1,1]));
});
test('Foundation softmax SGD example',()=>{
  const p=[.2,.3,.5],z=p.map((v,i)=>Math.log(v)-.1*(v-Number(i===0)));
  const exp=z.map(Math.exp), next=exp[0]/exp.reduce((a,b)=>a+b,0);
  assert.ok(next>.2 && next<.23);
  console.log(`Foundation probability after SGD: ${next.toFixed(8)}`);
});
