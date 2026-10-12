"""Browser transport, with a single human controller and read-only spectators."""
import asyncio
import json
from pathlib import Path

from aiohttp import web


SESSION = web.AppKey("session", object)


def create_app(session):
    app = web.Application(client_max_size=4096)
    app[SESSION] = session

    async def index(request):
        return web.FileResponse(Path(__file__).with_name("index.html"))

    async def websocket(request):
        # Only the served origin may acquire keyboard control.
        origin = request.headers.get("Origin")
        if origin and origin != "{}://{}".format(request.scheme, request.host):
            raise web.HTTPForbidden(text="Cross-origin control is disabled")
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=4096)
        await ws.prepare(request)
        controller = session.controller is None
        if controller:
            session.controller = ws
        session.clients.add(ws)
        await ws.send_json({"type": "role", "controller": controller})
        await ws.send_json(dict(session.view(), type="state"))
        try:
            async for message in ws:
                if message.type != web.WSMsgType.TEXT:
                    continue
                try:
                    if not controller:
                        raise ValueError("This connection is a spectator")
                    session.command(json.loads(message.data))
                except (ValueError, TypeError, KeyError):
                    await ws.send_json({"type": "error", "message": "Invalid command"})
        finally:
            session.clients.discard(ws)
            if session.controller is ws:
                session.command({"type": "pause"})
                session.controller = None
        return ws

    async def lifecycle(app):
        task = asyncio.create_task(session.run())
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for ws in tuple(session.clients):
            await ws.close()

    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.cleanup_ctx.append(lifecycle)
    return app
