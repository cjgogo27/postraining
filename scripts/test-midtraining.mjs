import test from 'node:test';
import assert from 'node:assert/strict';
import {budget} from '../midtraining/src/lib/budget.mjs';
test('Mid-training default target budget and source counts',()=>{
 assert.deepEqual(budget(4,32,1,1,96,25),{perUpdate:128,total:12288,general:3072,domain:9216});
});
test('Doubling length and halving microbatch preserves full-target budget',()=>{
 assert.deepEqual(budget(4,32,1,1,96,25),budget(2,64,1,1,96,25));
});
test('DP and accumulation multiply original targets; pure-source endpoints',()=>{
 assert.equal(budget(1,4096,4,2,1,0).total,32768);
 assert.equal(budget(4,32,1,1,96,100).domain,0);
 assert.equal(budget(4,32,1,1,96,0).general,0);
});
test('Invalid or unsafe budgets rejected',()=>{
 for(const args of [[0,32,1,1,96,25],[1.1,32,1,1,96,25],[4,32,1,1,96,101],[4,32,1,1,NaN,25],[2**40,2**30,1,1,1,25]])assert.throws(()=>budget(...args));
});
