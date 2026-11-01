import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    coverage: {
      provider: "v8",
      reporter: ["text", "cobertura"],
      include: ["src/**"],
      exclude: ["src/server.ts"],
    },
  },
});
