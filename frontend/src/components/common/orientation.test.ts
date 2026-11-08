import { expect, it } from "vitest";
import { defaultOrientation } from "./orientation.ts";

// lopsided and slightly bumpy, so every axis and sign is well defined
const MOLECULE = [
  [0, 0, 0],
  [1, 0.4, 0.1],
  [2, 0, -0.1],
  [3, 0.9, 0],
  [5, 0.1, 0.2],
  [1.5, 2, 0.05],
  [-1, 0.5, -0.1],
];

function rotate([x, y, z]: number[], [qx, qy, qz, qw]: number[]) {
  // v + 2w(q x v) + 2 q x (q x v)
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    x + qw * tx + (qy * tz - qz * ty),
    y + qw * ty + (qz * tx - qx * tz),
    z + qw * tz + (qx * ty - qy * tx),
  ];
}

function oriented(points: number[][]) {
  const q = defaultOrientation(points.map(([x, y, z]) => ({ x, y, z })));
  const mean = [0, 1, 2].map(
    (i) => points.reduce((s, p) => s + p[i], 0) / points.length,
  );
  return points.map((p) =>
    rotate(
      p.map((c, i) => c - mean[i]),
      q,
    ),
  );
}

it("lays the longest extent along x and the flattest along the view axis", () => {
  const out = oriented(MOLECULE);
  const spread = [0, 1, 2].map((i) => {
    const v = out.map((p) => p[i]);
    return Math.max(...v) - Math.min(...v);
  });

  expect(spread[0]).toBeGreaterThan(spread[1]);
  expect(spread[1]).toBeGreaterThan(spread[2]);
});

it("gives the same view however the molecule arrives rotated", () => {
  // 90 degrees about z, then 90 degrees about x
  const turned = MOLECULE.map(([x, y, z]) => [x, -z, y]).map(([x, y, z]) => [
    -y,
    x,
    z,
  ]);

  const a = oriented(MOLECULE);
  const b = oriented(turned);

  a.forEach((p, i) => p.forEach((c, k) => expect(b[i][k]).toBeCloseTo(c, 6)));
});
