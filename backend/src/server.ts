import { createApp } from "./app.ts";
import { env } from "./config/env.ts";

createApp().listen(env.port, () => {
  console.log(`backend listening on http://localhost:${env.port}`);
});
