interface Point {
  x: number;
  y: number;
  z: number;
}
type Vec = [number, number, number];

function eigenvectors(m: number[][]): Vec[] {
  const a = m.map((row) => [...row]);
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    if (a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2 < 1e-18) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t =
        (theta < 0 ? -1 : 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        [a[k][p], a[k][q]] = [
          c * a[k][p] - s * a[k][q],
          s * a[k][p] + c * a[k][q],
        ];
      }
      for (let k = 0; k < 3; k++) {
        [a[p][k], a[q][k]] = [
          c * a[p][k] - s * a[q][k],
          s * a[p][k] + c * a[q][k],
        ];
      }
      for (let k = 0; k < 3; k++) {
        [v[k][p], v[k][q]] = [
          c * v[k][p] - s * v[k][q],
          s * v[k][p] + c * v[k][q],
        ];
      }
    }
  }
  return [0, 1, 2]
    .sort((i, j) => a[j][j] - a[i][i])
    .map((i) => [v[0][i], v[1][i], v[2][i]]);
}

const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a: Vec, k: number): Vec => [a[0] * k, a[1] * k, a[2] * k];

export function principalAxes(points: Point[]): [Vec, Vec, Vec] {
  const n = points.length || 1;
  const mean = [0, 1, 2].map(
    (i) => points.reduce((sum, p) => sum + [p.x, p.y, p.z][i], 0) / n,
  );
  const centered = points.map(
    (p): Vec => [p.x - mean[0], p.y - mean[1], p.z - mean[2]],
  );
  const cov = [0, 1, 2].map((i) =>
    [0, 1, 2].map((j) => centered.reduce((s, p) => s + p[i] * p[j], 0) / n),
  );
  const [e1, e2] = eigenvectors(cov);
  const skew = (axis: Vec) =>
    centered.reduce((s, p) => s + dot(p, axis) ** 3, 0) / n;

  const x = skew(e1) > 0 ? scale(e1, -1) : e1;
  const y = skew(e2) < 0 ? scale(e2, -1) : e2;
  const z: Vec = [
    x[1] * y[2] - x[2] * y[1],
    x[2] * y[0] - x[0] * y[2],
    x[0] * y[1] - x[1] * y[0],
  ];
  return [x, y, z];
}

export function defaultOrientation(points: Point[]) {
  const [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] =
    principalAxes(points);
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    return [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  }
  if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  }
  if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  }
  const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
  return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
}
