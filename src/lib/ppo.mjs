// Scalar clipped surrogate, maximized. No model update or sampling is performed.
export function clippedSurrogate(ratio, advantage, epsilon=.2) {
  if (![ratio,advantage,epsilon].every(Number.isFinite) || ratio<=0 || epsilon<=0 || epsilon>=1) {
    throw new Error('需要正ratio、有限优势及0到1之间的ε。');
  }
  const raw=ratio*advantage;
  const candidate=Math.max(1-epsilon,Math.min(1+epsilon,ratio))*advantage;
  return {raw,candidate,objective:Math.min(raw,candidate),plateau:candidate<raw};
}
