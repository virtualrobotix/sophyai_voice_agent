"""
Remote LLM Module
Integrazione con server LLM remoti con autenticazione Bearer Token.
Supporta formato custom con message e collection.
"""

import os
import asyncio
import json
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
import httpx
from loguru import logger

# #region agent log - debug helper
DEBUG_LOG_PATH = "/app/config/debug.log"
def _debug_log(hypothesis_id, location, message, data=None):
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
    except: pass
# #endregion

# LiveKit imports
try:
    from livekit.agents import llm
    from livekit.plugins import openai as openai_plugin
except ImportError:
    logger.warning("LiveKit agents not available")
    llm = None
    openai_plugin = None


@dataclass
class RemoteLLMResponse:
    """Risposta del server LLM remoto"""
    text: str
    conversation_id: str
    done: bool
    raw_response: Dict[str, Any] = None


class RemoteLLM:
    """
    LLM Engine per server remoti personalizzati.
    
    Supporta server con formato:
    - Request: POST /chat {"message": "...", "collection": "..."}
    - Response: {"response": "...", "conversation_id": "..."}
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:5006",
        token: str = "",
        collection: str = "",
        timeout: float = 60.0,
        system_prompt: Optional[str] = None
    ):
        """
        Inizializza il client per server LLM remoto.
        
        Args:
            server_url: URL base del server (es. http://10.0.0.133:5006)
            token: Bearer token per autenticazione
            collection: Nome della collection da usare
            timeout: Timeout per le richieste in secondi
            system_prompt: Prompt di sistema opzionale
        """
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.collection = collection
        self.timeout = timeout
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_conversation_id: Optional[str] = None
        
        logger.info(f"Inizializzazione Remote LLM: url={server_url}, collection={collection}")
    
    def _default_system_prompt(self) -> str:
        """Prompt di sistema predefinito per l'assistente vocale"""
        return """Sei un assistente vocale intelligente e amichevole che risponde in italiano.
Le tue risposte devono essere:
- Concise e dirette (ideali per essere lette ad alta voce)
- Naturali e conversazionali
- Utili e informative
- Mai troppo lunghe (max 2-3 frasi per risposta)

Rispondi sempre in italiano, anche se la domanda è in un'altra lingua.
Evita di usare formattazioni come elenchi puntati o markdown, preferisci frasi scorrevoli."""
    
    def _get_headers(self) -> Dict[str, str]:
        """Costruisce gli headers per le richieste"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    async def chat(
        self,
        message: str,
        include_history: bool = True
    ) -> RemoteLLMResponse:
        """
        Invia un messaggio al server remoto.
        
        Args:
            message: Messaggio dell'utente
            include_history: Se includere la cronologia nella richiesta
            
        Returns:
            RemoteLLMResponse con la risposta
        """
        # Costruisci il messaggio completo con system prompt se è il primo
        full_message = message
        if not self.conversation_history and self.system_prompt:
            full_message = f"[Sistema: {self.system_prompt}]\n\nUtente: {message}"
        
        payload = {
            "message": full_message,
            "collection": self.collection
        }
        
        # Aggiungi conversation_id se disponibile per mantenere il contesto
        if self.current_conversation_id:
            payload["conversation_id"] = self.current_conversation_id
        
        logger.debug(f"Remote LLM request: {message[:50]}...")
        
        # #region agent log - H1: verifica richiesta
        _debug_log("H1", "remote_llm.py:chat:entry", "Inizio chiamata chat", {"server_url": self.server_url, "collection": self.collection, "message_len": len(message)})
        # #endregion
        
        try:
            # #region agent log - H2: verifica URL e payload
            _debug_log("H2", "remote_llm.py:chat:pre_request", "Preparazione richiesta", {"url": f"{self.server_url}/chat", "payload_keys": list(payload.keys()), "has_token": bool(self.token)})
            # #endregion
            
            # #region agent log - H10: payload completo
            _debug_log("H10", "remote_llm.py:chat:full_payload", "Payload completo inviato", {
                "message_preview": payload.get("message", "")[:300],
                "message_full_length": len(payload.get("message", "")),
                "collection": payload.get("collection", "MISSING"),
                "conversation_id": payload.get("conversation_id", "NONE"),
                "has_system_prompt_in_msg": "[Sistema:" in payload.get("message", "")
            })
            # #endregion
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.server_url}/chat",
                    headers=self._get_headers(),
                    json=payload
                )
                
                # #region agent log - H3: verifica risposta HTTP
                _debug_log("H3", "remote_llm.py:chat:response", "Risposta ricevuta", {"status_code": resp.status_code, "content_type": resp.headers.get("content-type", "unknown")})
                # #endregion
                
                if resp.status_code != 200:
                    error_text = resp.text
                    logger.error(f"Remote LLM error: {resp.status_code} - {error_text}")
                    # #region agent log - H4: errore HTTP
                    _debug_log("H4", "remote_llm.py:chat:error", "Errore HTTP", {"status": resp.status_code, "error": error_text[:200]})
                    # #endregion
                    return RemoteLLMResponse(
                        text=f"Errore server: {resp.status_code}",
                        conversation_id="",
                        done=True,
                        raw_response={"error": error_text}
                    )
                
                data = resp.json()
                
                # Estrai risposta dal formato del server
                response_text = data.get("response", "")
                conversation_id = data.get("conversation_id", "")
                
                # Salva conversation_id per richieste successive
                if conversation_id:
                    self.current_conversation_id = conversation_id
                
                # Aggiorna cronologia
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Limita cronologia
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                logger.debug(f"Remote LLM response: {response_text[:100]}...")
                
                # #region agent log - H3: risposta parsata con successo
                _debug_log("H3", "remote_llm.py:chat:success", "Risposta parsata", {"response_len": len(response_text), "has_conversation_id": bool(conversation_id), "response_preview": response_text[:100] if response_text else "EMPTY"})
                # #endregion
                
                return RemoteLLMResponse(
                    text=response_text,
                    conversation_id=conversation_id,
                    done=True,
                    raw_response=data
                )
                
        except httpx.TimeoutException:
            logger.error("Remote LLM timeout")
            # #region agent log - H5: timeout
            _debug_log("H5", "remote_llm.py:chat:timeout", "Timeout connessione", {"server_url": self.server_url, "timeout": self.timeout})
            # #endregion
            return RemoteLLMResponse(
                text="Timeout: il server non risponde",
                conversation_id="",
                done=True
            )
        except httpx.ConnectError as e:
            logger.error(f"Remote LLM connection error: {e}")
            # #region agent log - H1: errore connessione
            _debug_log("H1", "remote_llm.py:chat:connect_error", "Errore connessione", {"server_url": self.server_url, "error": str(e)})
            # #endregion
            return RemoteLLMResponse(
                text="Errore connessione al server remoto",
                conversation_id="",
                done=True
            )
        except Exception as e:
            logger.error(f"Remote LLM error: {e}")
            # #region agent log - H5: errore generico
            _debug_log("H5", "remote_llm.py:chat:exception", "Eccezione generica", {"error": str(e), "type": type(e).__name__})
            # #endregion
            return RemoteLLMResponse(
                text=f"Errore: {str(e)}",
                conversation_id="",
                done=True
            )
    
    async def chat_stream(
        self,
        message: str
    ) -> AsyncGenerator[RemoteLLMResponse, None]:
        """
        Genera risposta in streaming (simulato per server non-streaming).
        
        Per server che non supportano streaming nativo, invia la risposta
        completa in un singolo chunk.
        
        Args:
            message: Messaggio dell'utente
            
        Yields:
            RemoteLLMResponse con chunk della risposta
        """
        # Per ora il server remoto non supporta streaming,
        # quindi simuliamo con una risposta completa
        response = await self.chat(message)
        
        # Yield la risposta completa
        yield response
    
    def clear_history(self) -> None:
        """Pulisce la cronologia della conversazione"""
        self.conversation_history = []
        self.current_conversation_id = None
        logger.info("Cronologia conversazione cancellata")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Imposta un nuovo prompt di sistema.
        
        Args:
            prompt: Nuovo prompt di sistema
        """
        self.system_prompt = prompt
        logger.info("Prompt di sistema aggiornato")
    
    def set_collection(self, collection: str) -> None:
        """
        Cambia la collection in uso.
        
        Args:
            collection: Nome della nuova collection
        """
        self.collection = collection
        logger.info(f"Collection aggiornata: {collection}")
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Testa la connessione al server remoto.
        
        Returns:
            Dict con status e dettagli della connessione
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Prova prima un health check se disponibile
                try:
                    health_resp = await client.get(
                        f"{self.server_url}/health",
                        headers=self._get_headers()
                    )
                    if health_resp.status_code == 200:
                        return {
                            "status": "ok",
                            "message": "Server raggiungibile",
                            "endpoint": "/health",
                            "data": health_resp.json() if health_resp.headers.get("content-type", "").startswith("application/json") else {}
                        }
                except:
                    pass
                
                # Se health non disponibile, prova un messaggio di test
                test_payload = {
                    "message": "test connection",
                    "collection": self.collection
                }
                
                resp = await client.post(
                    f"{self.server_url}/chat",
                    headers=self._get_headers(),
                    json=test_payload
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "ok",
                        "message": "Connessione riuscita",
                        "endpoint": "/chat",
                        "response_preview": data.get("response", "")[:100]
                    }
                elif resp.status_code == 401:
                    return {
                        "status": "error",
                        "message": "Token non valido o mancante",
                        "code": 401
                    }
                elif resp.status_code == 403:
                    return {
                        "status": "error", 
                        "message": "Accesso negato",
                        "code": 403
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Errore HTTP {resp.status_code}",
                        "code": resp.status_code,
                        "detail": resp.text[:200]
                    }
                    
        except httpx.TimeoutException:
            return {
                "status": "error",
                "message": "Timeout: server non risponde"
            }
        except httpx.ConnectError:
            return {
                "status": "error",
                "message": f"Impossibile connettersi a {self.server_url}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_collections(self) -> List[str]:
        """
        Recupera la lista delle collection disponibili (se supportato dal server).
        
        Returns:
            Lista di nomi collection
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.server_url}/collections",
                    headers=self._get_headers()
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Supporta vari formati di risposta
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return data.get("collections", data.get("data", []))
                    return []
                else:
                    logger.warning(f"Impossibile ottenere collections: {resp.status_code}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Errore recupero collections: {e}")
            return []


class RemoteLLMWrapper:
    """
    Wrapper per usare RemoteLLM con l'interfaccia LiveKit LLM.
    Converte le chiamate LiveKit nel formato del server remoto.
    """
    
    def __init__(
        self,
        server_url: str,
        token: str,
        collection: str,
        system_prompt: Optional[str] = None
    ):
        self.remote_llm = RemoteLLM(
            server_url=server_url,
            token=token,
            collection=collection,
            system_prompt=system_prompt
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Interfaccia chat compatibile con LiveKit.
        
        Args:
            messages: Lista di messaggi nel formato [{"role": "...", "content": "..."}]
            
        Returns:
            Testo della risposta
        """
        # Estrai l'ultimo messaggio utente
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return "Nessun messaggio da elaborare"
        
        response = await self.remote_llm.chat(user_message)
        return response.text


def get_remote_llm(
    server_url: str = None,
    token: str = None,
    collection: str = None
) -> Optional[RemoteLLM]:
    """
    Factory function per creare un'istanza RemoteLLM.
    
    Args:
        server_url: URL del server remoto
        token: Bearer token
        collection: Collection da usare
        
    Returns:
        Istanza RemoteLLM o None se configurazione mancante
    """
    url = server_url or os.getenv("REMOTE_LLM_URL", "")
    tok = token or os.getenv("REMOTE_LLM_TOKEN", "")
    coll = collection or os.getenv("REMOTE_LLM_COLLECTION", "")
    
    if not url:
        logger.warning("Remote LLM URL non configurato")
        return None
    
    return RemoteLLM(
        server_url=url,
        token=tok,
        collection=coll
    )
