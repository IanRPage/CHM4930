# Hosting the inference server on personal compute

The deployed backend reaches the server through a named Cloudflare Tunnel
(quick tunnels change URL on every restart).

1. Download a published version or copy the five `mufu_seed_<seed>.pt` files and
   `mufu_ensemble.json` into a folder only the server uses, e.g.
   `~/.local/share/mufu/<version>/`. Don't serve `checkpoints/` directly:
   training overwrites it, so a restart mid-training would fail to load, and one
   after training would switch models unannounced. To ship a new model, copy it
   to a new folder, update `MUFU_CHECKPOINT_DIR` and restart the server.
2. Put the env vars in `~/.config/mufu/server.env` (`chmod 600`), with
   `MUFU_CHECKPOINT_DIR` set to that folder's absolute path, and
   `HF_HUB_OFFLINE=1`. Start the server once without it first: the first start downloads
   MolFormer into the Hugging Face cache, and offline mode can't start without it.
3. Create the tunnel: `cloudflared tunnel login`,
   `cloudflared tunnel create mufu`, `cloudflared tunnel route dns mufu <hostname>`.
4. `~/.cloudflared/config.yml`:

   ```yaml
   tunnel: mufu
   credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
   ingress:
     - hostname: <hostname>
       service: http://127.0.0.1:8000
     - service: http_status:404
   ```

5. `~/.config/systemd/user/mufu-server.service`:

   ```ini
   [Service]
   WorkingDirectory=%h/<path to repo>/ai
   EnvironmentFile=%h/.config/mufu/server.env
   Environment=PYTHONPATH=src
   ExecStart=<conda env prefix>/bin/python -m mufu_server
   Restart=on-failure
   RestartSec=10

   [Install]
   WantedBy=default.target
   ```

   `mufu-tunnel.service` is the same, with
   `ExecStart=/usr/bin/cloudflared tunnel run mufu` and `Restart=always`.

6. `systemctl --user enable --now mufu-server mufu-tunnel`, and
   `loginctl enable-linger $USER` so they start at boot.
7. Disable suspend on AC power and add an uptime check on
   `https://<hostname>/health` that expects `"status": "ready"`; it returns 200
   with `"loading"` while the model loads.

Keep the key and `~/.cloudflared/*.json` out of the repo.
