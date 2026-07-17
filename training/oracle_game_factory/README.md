# Titanium Oracle Game Factory

Oracle runs only self-play game generation. It never trains, promotes, or
writes laptop state.

## Runtime Layout

```text
/opt/titanium-game-factory      immutable code + engine source/build
/var/lib/titanium-game-factory  generations, spool, token, status
/var/log/titanium-game-factory  optional logs; journald is primary
```

## API

All endpoints require:

```text
Authorization: Bearer <token>
```

Endpoints:

- `GET /health`
- `GET /status`
- `GET /generation`
- `POST /generation/stage`
- `POST /generation/activate`
- `GET /results?limit=N&after=GAME_ID`
- `GET /results/<game_id>`
- `POST /ack`
- `POST /pause`
- `POST /resume`
- `POST /drain`

Website-finished-game submission uses a separate limited token created at
`/var/lib/titanium-game-factory/website_submit_token`:

```text
POST /submit/website-game
X-Website-Submit-Token: <website_submit_token>
```

This endpoint only queues completed website games into the durable result spool.
The laptop importer later pulls them through the normal `/results` + `/ack`
flow and imports them with source `website_public`.

Configure the static site with:

```text
VITE_ORACLE_SUBMIT_URL=https://<oracle-host>/submit/website-game
VITE_ORACLE_SUBMIT_TOKEN=<website_submit_token>
```

The server binds to `127.0.0.1:8765`. Use an SSH tunnel from the laptop.

## Safety

- Filesystem spool is authoritative.
- Games are written to a temporary file and atomically renamed into `ready/`.
- Acknowledgement moves the file to `archive/`.
- No unacknowledged game is deleted for space pressure.
- Backpressure warns at 10 GB and stops scheduling new games at 20 GB.
- Active generations are staged, hash-verified, then atomically activated.
