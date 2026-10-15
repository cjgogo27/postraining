// Stable scalar illustration of the original sum-log-prob DPO objective.
export function dpoMetrics(pc, pr, rc, rr, beta) {
  if (![pc,pr,rc,rr,beta].every(Number.isFinite) || [pc,pr,rc,rr].some(x=>x>0) || beta<=0) throw new Error('请输入有限且不大于0的联合log概率，β须大于0。');
  const margin=beta*((pc-pr)-(rc-rr));
  if(!Number.isFinite(margin)) throw new Error('数值范围过大，请使用可计算的联合log概率。');
  const loss=Math.max(0,-margin)+Math.log1p(Math.exp(-Math.abs(margin)));
  return {margin,loss};
}
