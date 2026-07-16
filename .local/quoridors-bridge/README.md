# Quoridors Bridge (local, gitignored)

Chrome extension under `.local/quoridors-bridge/` that hooks [quoridors.com](https://quoridors.com) game traffic and runs the Titanium WASM engine for hints and autoplay.

Based on the wallz-bridge extension; adapted using the scraped site sources in `.local/quoridors.com/scraped/`.

This folder is listed in `.gitignore` as `.local/` — it will not be committed.

## Install

1. Chrome → `chrome://extensions` → Developer mode → **Load unpacked**
2. Select: `.local/quoridors-bridge/extension`
3. Copy the extension ID (32-char string)

## Quick test

1. Open https://quoridors.com and start a game (vs Computer, online, or hotseat).
2. The HUD appears to the left of the board: eval bar, **Autoplay**, **Play best**.
3. Extension popup (toolbar icon) exposes think-time / delay sliders and a debug **Force play** box.

## In-page API (quoridors.com tab console)

When the hook is active:

```javascript
__QUORIDORS_BRIDGE__.getState();
__QUORIDORS_BRIDGE__.playAlgebraic("e2");
__QUORIDORS_BRIDGE__.buildBridgeDetail();
```

## What it intercepts

| Direction | Channel                         | Notes                                                      |
| --------- | ------------------------------- | ---------------------------------------------------------- |
| Out/In    | WebSocket `/ws/game/{id}`       | JSON `{ type: "state" }`, `{ type: "move", action }`, etc. |
| Out       | REST `POST /api/game/{id}/move` | Local AI / hotseat moves                                   |
| In        | REST responses                  | Game state snapshots after each move                       |

Move payload (from scraped `api.js` / `interaction.js`):

- Pawn: `{ type: "pawn", to: [row, col] }` — row 0 = top, col 0 = `a`
- Wall: `{ type: "wall", orientation: "H"|"V", slot: [row, col] }`

Algebraic notation for Titanium: `e2` (pawn), `d2h` / `d2v` (walls) — column letter + rank `9 - row`.

## Wire into your site

Serve or copy `bridge/quoridors-bridge-client.js`, then:

```html
<script src="/quoridors-bridge-client.js"></script>
<script>
  QuoridorsBridgeClient.setExtensionId("YOUR_EXTENSION_ID");

  QuoridorsBridgeClient.connectPort((msg) => {
    if (msg.type === "state:update") console.log("snapshot", msg.detail);
  });

  QuoridorsBridgeClient.playAlgebraic("e2").then(console.log);
</script>
```

`externally_connectable` allows `quoridors.com`, `localhost`, and `127.0.0.1`.

## Titanium engine

The WASM worker is copied from wallz-bridge (`extension/engine/`). It runs fully in-browser via the offscreen document — no local server required.

## Notes

- **Bot game history is blocked** by default: `API.recordCasual({ kind: "ai" })`, matching `fetch` POSTs to `/api/games/casual`, and `Stats.add` for bot games never reach the server. Check `__QUORIDORS_BRIDGE__.getState().blockedBotHistory` in the console.
- Vs Computer: the hook wraps `API.move` and auto-fetches `API.aiMove` when needed.
- Full move history comes from `state.history` (walls included), unlike the wallz React-tree workaround.
