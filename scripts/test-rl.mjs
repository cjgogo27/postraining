import test from 'node:test';
import assert from 'node:assert/strict';
import {clippedSurrogate} from '../src/lib/ppo.mjs';
test('PPO four regions match independent hand answers',()=>{
  for(const [ratio,a,want,plateau] of [[1.5,1,1.2,true],[1.5,-1,-1.5,false],[.5,1,.5,false],[.5,-1,-.8,true]]) {
    const result=clippedSurrogate(ratio,a);
    assert.ok(Math.abs(result.objective-want)<1e-12);assert.equal(result.plateau,plateau);
  }
});
test('PPO plateau has zero finite-difference slope while harmful move remains penalized',()=>{
  const slope=(rho,a)=>(clippedSurrogate(rho+1e-5,a).objective-clippedSurrogate(rho-1e-5,a).objective)/2e-5;
  assert.ok(Math.abs(slope(1.5,1))<1e-9);assert.ok(Math.abs(slope(.5,-1))<1e-9);
  assert.ok(Math.abs(slope(1.5,-1)+1)<1e-9);assert.ok(Math.abs(slope(.5,1)-1)<1e-9);
});
test('PPO scalar rejects invalid ratios and epsilon',()=>{
  for(const values of [[0,1],[NaN,1],[-1,1],[1,Infinity],[1,1,0],[1,1,1]]) assert.throws(()=>clippedSurrogate(...values));
  assert.equal(clippedSurrogate(1,0).objective,0);
});
