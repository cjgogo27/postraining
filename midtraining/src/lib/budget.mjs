// Hypothetical full-target, no-padding DP budget. TP/PP/CP do not multiply tokens.
export function budget(batch,length,accumulation,dp,updates,generalPercent){
  const values=[batch,length,accumulation,dp,updates];
  if(values.some(x=>!Number.isSafeInteger(x)||x<=0)||!Number.isFinite(generalPercent)||generalPercent<0||generalPercent>100)throw new Error('批量/长度/累积/DP/更新需为正整数，回放比例为0–100。');
  const perUpdate=batch*length*accumulation*dp,total=perUpdate*updates;
  if(!Number.isSafeInteger(total))throw new Error('计数超过安全整数范围');
  return {perUpdate,total,general:total*generalPercent/100,domain:total*(1-generalPercent/100)};
}
