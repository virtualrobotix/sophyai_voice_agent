#!/usr/bin/env python3
"""
Proxy TLS per LiveKit
Accetta connessioni WSS e le inoltra a LiveKit WS
"""

import asyncio
import ssl
import logging
from pathlib import Path

try:
    import websockets
    from websockets.asyncio.client import connect
except ImportError:
    print("Installando websockets...")
    import subprocess
    subprocess.run(["pip", "install", "websockets"], check=True)
    import websockets
    from websockets.asyncio.client import connect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

LIVEKIT_WS_URL = "ws://localhost:7880"
PROXY_HOST = "0.0.0.0"
PROXY_PORT = 7443

# Path certificati
CERT_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


async def proxy_handler(client_ws):
    """Gestisce una connessione client e la proxya a LiveKit"""
    # Estrai il path dalla connessione (nuova API websockets)
    path = client_ws.request.path if hasattr(client_ws, 'request') else "/"
    
    # IMPORTANTE: Traduci /rtc/v1 -> /rtc per compatibilità con LiveKit server 1.x
    # Il client SDK v2 usa /rtc/v1 ma il server 1.x usa /rtc
    if path.startswith("/rtc/v1"):
        path = path.replace("/rtc/v1", "/rtc", 1)
        logger.info(f"📝 Path tradotto: /rtc/v1 -> /rtc")
    
    target_url = f"{LIVEKIT_WS_URL}{path}"
    
    logger.info(f"🔗 Nuova connessione: {path[:80]}...")
    
    try:
        async with connect(target_url) as livekit_ws:
            logger.info(f"✅ Connesso a LiveKit")
            
            # Crea task per il forwarding bidirezionale
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
            
            # Esegui entrambi i task
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_livekit()),
                    asyncio.create_task(livekit_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancella i task pending
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
    except Exception as e:
        logger.error(f"❌ Errore proxy: {e}")
    finally:
        logger.info("🔌 Connessione chiusa")


async def main():
    # Configura SSL
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)
    
    logger.info(f"🔒 LiveKit TLS Proxy avviato su wss://0.0.0.0:{PROXY_PORT}")
    logger.info(f"   Forwarding a: {LIVEKIT_WS_URL}")
    
    async with websockets.serve(
        proxy_handler,
        PROXY_HOST,
        PROXY_PORT,
        ssl=ssl_context,
        ping_interval=20,
        ping_timeout=60,
        max_size=10 * 1024 * 1024  # 10MB max message
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())


