#!/usr/bin/env python3
"""
Proxy TLS per LiveKit
Accetta connessioni WSS e richieste HTTP e le inoltra a LiveKit
"""

import asyncio
import ssl
import logging
from pathlib import Path
import aiohttp
from aiohttp import web

try:
    import websockets
    from websockets.asyncio.client import connect
except ImportError:
    print("Installando websockets...")
    import subprocess
    subprocess.run(["pip", "install", "websockets"], check=True)
    import websockets
    from websockets.asyncio.client import connect

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LIVEKIT_WS_URL = "ws://localhost:7880"
LIVEKIT_HTTP_URL = "http://localhost:7880"
PROXY_HOST = "0.0.0.0"
PROXY_PORT = 7443

# Path certificati
CERT_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


async def proxy_handler(client_ws):
    """Gestisce una connessione client e la proxya a LiveKit"""
    # #region agent log
    import json; from pathlib import Path; log_data = {"location": "livekit_tls_proxy.py:proxy_handler", "message": "New connection", "data": {"has_request": hasattr(client_ws, 'request'), "remote": str(getattr(client_ws, 'remote_address', 'unknown')), "ws_type": type(client_ws).__name__}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    # #endregion
    # Estrai il path completo (con query string) dalla connessione
    if hasattr(client_ws, 'ws') and hasattr(client_ws.ws, 'request'):  # aiohttp WebSocketResponse
        path_qs = client_ws.ws.request.path_qs  # Include query string
    elif hasattr(client_ws, 'request'):
        if hasattr(client_ws.request, 'path_qs'):
            path_qs = client_ws.request.path_qs
        elif hasattr(client_ws.request, 'path'):
            path_qs = client_ws.request.path
        else:
            path_qs = "/"
    else:
        path_qs = "/"
    
    # IMPORTANTE: Traduci /rtc/v1 -> /rtc per compatibilità con LiveKit server 1.x
    # Il client SDK v2 usa /rtc/v1 ma il server 1.x usa /rtc
    # Mantieni la query string!
    if path_qs.startswith("/rtc/v1"):
        path_qs = path_qs.replace("/rtc/v1", "/rtc", 1)
        logger.info(f"📝 Path tradotto: /rtc/v1 -> /rtc (mantenendo query string)")
    
    target_url = f"{LIVEKIT_WS_URL}{path_qs}"
    
    logger.info(f"🔗 Nuova connessione: {path_qs[:80]}...")
    
    # #region agent log
    log_data = {"location": "livekit_tls_proxy.py:proxy_handler", "message": "Connecting to LiveKit", "data": {"path_qs": path_qs, "target_url": target_url}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path.open("a").write(json.dumps(log_data) + "\n")
    # #endregion
    
    try:
        # Gestisci sia websockets che aiohttp WebSocketResponse
        if hasattr(client_ws, 'ws'):  # aiohttp WebSocketResponse
            ws = client_ws.ws
            async with connect(target_url) as livekit_ws:
                logger.info(f"✅ Connesso a LiveKit")
                # #region agent log
                log_data = {"location": "livekit_tls_proxy.py:proxy_handler", "message": "Connected to LiveKit", "data": {"target_url": target_url}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path.open("a").write(json.dumps(log_data) + "\n")
                # #endregion
                
                async def client_to_livekit():
                    try:
                        logger.debug("Inizio forwarding client->livekit")
                        async for message in ws:
                            # aiohttp WSMessage ha un attributo .data che contiene i dati effettivi
                            if hasattr(message, 'data'):
                                data = message.data
                            else:
                                data = message
                            
                            logger.debug(f"Messaggio ricevuto da client: {type(data).__name__}, len={len(data) if hasattr(data, '__len__') else 'N/A'}")
                            
                            # LiveKit si aspetta bytes
                            if isinstance(data, str):
                                message_bytes = data.encode('utf-8')
                            elif isinstance(data, bytes):
                                message_bytes = data
                            else:
                                # Altri tipi, prova a convertire
                                message_bytes = bytes(str(data), 'utf-8')
                            await livekit_ws.send(message_bytes)
                        logger.debug("Fine forwarding client->livekit (loop terminato)")
                    except websockets.exceptions.ConnectionClosed:
                        logger.debug("Connessione client->livekit chiusa")
                    except Exception as e:
                        logger.error(f"Errore client->livekit: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                async def livekit_to_client():
                    try:
                        logger.debug("Inizio forwarding livekit->client")
                        async for message in livekit_ws:
                            logger.debug(f"Messaggio ricevuto da LiveKit: {type(message).__name__}, len={len(message) if hasattr(message, '__len__') else 'N/A'}")
                            # LiveKit invia messaggi binari (protobuf), non stringhe
                            if isinstance(message, bytes):
                                await ws.send_bytes(message)
                            elif isinstance(message, str):
                                await ws.send_bytes(message.encode('utf-8'))
                            else:
                                # Altri tipi, prova a convertire
                                await ws.send_bytes(bytes(str(message), 'utf-8'))
                        logger.debug("Fine forwarding livekit->client (loop terminato)")
                    except websockets.exceptions.ConnectionClosed:
                        logger.debug("Connessione livekit->client chiusa")
                    except Exception as e:
                        logger.error(f"Errore livekit->client: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Esegui entrambi i task e aspetta che uno finisca
                done, pending = await asyncio.wait(
                    [asyncio.create_task(client_to_livekit()), asyncio.create_task(livekit_to_client())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancella i task pending
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        else:  # websockets
            async with connect(target_url) as livekit_ws:
                logger.info(f"✅ Connesso a LiveKit")
                # #region agent log
                log_data = {"location": "livekit_tls_proxy.py:proxy_handler", "message": "Connected to LiveKit", "data": {"target_url": target_url}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path.open("a").write(json.dumps(log_data) + "\n")
                # #endregion
                
                async def client_to_livekit():
                    try:
                        async for message in client_ws:
                            await livekit_ws.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    except Exception as e:
                        logger.error(f"Errore client->livekit: {e}")
                
                async def livekit_to_client():
                    try:
                        async for message in livekit_ws:
                            await client_ws.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    except Exception as e:
                        logger.error(f"Errore livekit->client: {e}")
                
                await asyncio.wait([asyncio.create_task(client_to_livekit()), asyncio.create_task(livekit_to_client())], return_when=asyncio.FIRST_COMPLETED)
            
    except Exception as e:
        logger.error(f"❌ Errore proxy: {e}")
        # #region agent log
        import json; from pathlib import Path; log_data = {"location": "livekit_tls_proxy.py:proxy_handler", "message": "Proxy error", "data": {"error": str(e), "error_type": type(e).__name__}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
        # #endregion
    finally:
        logger.info("🔌 Connessione chiusa")


async def http_proxy_handler(request):
    """Gestisce richieste HTTP e le inoltra a LiveKit"""
    # #region agent log
    import json; log_data = {"location": "livekit_tls_proxy.py:http_proxy_handler", "message": "HTTP request", "data": {"path": request.path, "method": request.method}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "D"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    # #endregion
    target_url = f"{LIVEKIT_HTTP_URL}{request.path_qs}"
    logger.info(f"🌐 HTTP Proxy: {request.method} {request.path_qs} -> {target_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                request.method,
                target_url,
                headers=dict(request.headers),
                data=await request.read() if request.can_read_body else None
            ) as resp:
                # #region agent log
                log_data = {"location": "livekit_tls_proxy.py:http_proxy_handler", "message": "HTTP response", "data": {"status": resp.status, "target_url": target_url}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "D"}; log_path.open("a").write(json.dumps(log_data) + "\n")
                # #endregion
                response = web.Response(
                    body=await resp.read(),
                    status=resp.status,
                    headers=dict(resp.headers)
                )
                return response
    except Exception as e:
        logger.error(f"❌ Errore HTTP proxy: {e}")
        # #region agent log
        log_data = {"location": "livekit_tls_proxy.py:http_proxy_handler", "message": "HTTP proxy error", "data": {"error": str(e)}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "D"}; log_path.open("a").write(json.dumps(log_data) + "\n")
        # #endregion
        return web.Response(status=502, text=f"Proxy error: {e}")


async def websocket_handler(request):
    """Gestisce upgrade WebSocket e delega al proxy_handler"""
    # #region agent log
    import json; from pathlib import Path; log_data = {"location": "livekit_tls_proxy.py:websocket_handler", "message": "WebSocket upgrade request", "data": {"path_qs": request.path_qs, "headers": dict(request.headers)}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "hypothesisId": "C"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    # #endregion
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Crea un wrapper per compatibilità - mantieni path_qs completo
    class WSWrapper:
        def __init__(self, ws, request):
            self.ws = ws
            self.request = type('obj', (object,), {'path_qs': request.path_qs})()
            self.remote_address = request.remote
    
    wrapped_ws = WSWrapper(ws, request)
    await proxy_handler(wrapped_ws)
    return ws


async def handler(request):
    """Handler unificato che distingue tra WebSocket e HTTP"""
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return await websocket_handler(request)
    else:
        return await http_proxy_handler(request)


async def main():
    # Configura SSL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)
    
    # Crea app HTTP che gestisce sia HTTP che WebSocket
    http_app = web.Application()
    http_app.router.add_route('*', '/{path:.*}', handler)
    
    # Runner HTTP
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, PROXY_HOST, PROXY_PORT, ssl_context=ssl_context)
    
    logger.info(f"🔒 LiveKit TLS Proxy avviato su wss://0.0.0.0:{PROXY_PORT}")
    logger.info(f"   WebSocket forwarding a: {LIVEKIT_WS_URL}")
    logger.info(f"   HTTP forwarding a: {LIVEKIT_HTTP_URL}")
    
    # Avvia HTTP server (gestisce anche WebSocket)
    await http_site.start()
    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())


