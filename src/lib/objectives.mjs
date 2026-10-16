// Stable scalar illustration of the original sum-log-prob DPO objective.
export function dpoMetrics(pc, pr, rc, rr, beta) {
  if (![pc,pr,rc,rr,beta].every(Number.isFinite) || [pc,pr,rc,rr].some(x=>x>0) || beta<=0) throw new Error('请输入有限且不大于0的联合log概率，β须大于0。');
  const margin=beta*((pc-pr)-(rc-rr));
  if(!Number.isFinite(margin)) throw new Error('数值范围过大，请使用可计算的联合log概率。');
  const loss=Math.max(0,-margin)+Math.log1p(Math.exp(-Math.abs(margin)));
  return {margin,loss};
}

export const dpoPresets = {
  hand: { values: [-3,-5,-4,-4,0.1], hint: '手算：当前差2，参考差0，m=0.2。' },
  equal: { values: [-2,-4,-2,-4,0.1], hint: '当前等于参考：候选原本不等概率，但相对变化为0，loss=ln2。' },
  bothDown: { values: [Math.log(.1),Math.log(.01),Math.log(.2),Math.log(.1),0.1], hint: '参考→当前：chosen 0.2→0.1，rejected 0.1→0.01；两者都降，但比值2→10，margin上升。剩余概率属于其他回答。' },
};

// Shared pure model for the token alignment view and its numerical checks.
export function byteAlignment() {
  const sequence=[...new TextEncoder().encode('Q:2+3\nA:5'),256];
  return sequence.slice(0,-1).map((input,index)=>({index,input,label:sequence[index+1],supervised:index>=7}));
}

// One sampled group, four distinct candidate actions, initial uniform policy.
// No resampling, KL or multi-epoch clipping: one initial policy-gradient SGD step.
export function groupStep(rewards, learningRate=.4) {
  if(rewards.length!==4 || rewards.some(x=>!Number.isFinite(x)) || !Number.isFinite(learningRate) || learningRate<=0) throw new Error('需要四个有限奖励与正学习率。');
  const mean=rewards.reduce((a,b)=>a+b,0)/4;
  const std=Math.sqrt(rewards.reduce((a,b)=>a+(b-mean)**2,0)/4);
  const advantages=rewards.map(x=>(x-mean)/Math.max(std,1e-8));
  const average=advantages.reduce((a,b)=>a+b,0)/4;
  const gradients=advantages.map(a=>average/4-a/4);
  const logits=gradients.map(g=>-learningRate*g);
  const exp=logits.map(x=>Math.exp(x-Math.max(...logits)));
  const denominator=exp.reduce((a,b)=>a+b,0);
  return {mean,std,advantages,gradients,probabilities:exp.map(x=>x/denominator)};
}
