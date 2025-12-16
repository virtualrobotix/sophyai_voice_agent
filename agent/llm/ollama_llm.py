"""
Ollama LLM Module
Integrazione con Ollama per la generazione di risposte.
"""

import os
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
import ollama
from ollama import AsyncClient
from loguru import logger


@dataclass
class LLMResponse:
    """Risposta del LLM"""
    text: str
    model: str
    done: bool
    tokens_generated: int = 0


class OllamaLLM:
    """
    LLM Engine basato su Ollama.
    Supporta sia generazione sincrona che streaming asincrono.
    """
    
    def __init__(
        self,
        model: str = "gpt-oss",
        host: str = "http://localhost:11434",
        system_prompt: Optional[str] = None
    ):
        """
        Inizializza il client Ollama.
        
        Args:
            model: Nome del modello da usare
            host: URL del server Ollama
            system_prompt: Prompt di sistema opzionale
        """
        self.model = model
        self.host = host
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation_history: list[dict] = []
        
        # Client sincrono
        self.client = ollama.Client(host=host)
        
        # Client asincrono
        self.async_client = AsyncClient(host=host)
        
        logger.info(f"Inizializzazione Ollama LLM: model={model}, host={host}")
    
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
    
    def generate(self, user_message: str) -> LLMResponse:
        """
        Genera una risposta in modo sincrono.
        
        Args:
            user_message: Messaggio dell'utente
            
        Returns:
            LLMResponse con la risposta generata
        """
        # Costruisci i messaggi
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        logger.debug(f"Generazione risposta per: {user_message[:50]}...")
        
        response = self.client.chat(
            model=self.model,
            messages=messages
        )
        
        assistant_message = response["message"]["content"]
        
        # Aggiorna la cronologia
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        # Limita la cronologia per evitare contesti troppo lunghi
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return LLMResponse(
            text=assistant_message,
            model=self.model,
            done=True,
            tokens_generated=response.get("eval_count", 0)
        )
    
    async def generate_stream(self, user_message: str) -> AsyncGenerator[LLMResponse, None]:
        """
        Genera una risposta in streaming asincrono.
        
        Args:
            user_message: Messaggio dell'utente
            
        Yields:
            LLMResponse con chunk della risposta
        """
        # Costruisci i messaggi
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        logger.debug(f"Generazione streaming per: {user_message[:50]}...")
        
        full_response = ""
        tokens = 0
        
        async for part in await self.async_client.chat(
            model=self.model,
            messages=messages,
            stream=True
        ):
            chunk = part["message"]["content"]
            full_response += chunk
            tokens += 1
            
            yield LLMResponse(
                text=chunk,
                model=self.model,
                done=part.get("done", False),
                tokens_generated=tokens
            )
        
        # Aggiorna la cronologia
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Limita la cronologia
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self) -> None:
        """Pulisce la cronologia della conversazione"""
        self.conversation_history = []
        logger.info("Cronologia conversazione cancellata")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Imposta un nuovo prompt di sistema.
        
        Args:
            prompt: Nuovo prompt di sistema
        """
        self.system_prompt = prompt
        logger.info("Prompt di sistema aggiornato")
    
    async def check_model_available(self) -> bool:
        """
        Verifica se il modello è disponibile.
        
        Returns:
            True se il modello è disponibile
        """
        try:
            models = await self.async_client.list()
            available_models = [m["name"] for m in models.get("models", [])]
            
            # Controlla sia il nome esatto che con :latest
            is_available = (
                self.model in available_models or 
                f"{self.model}:latest" in available_models
            )
            
            if not is_available:
                logger.warning(f"Modello {self.model} non trovato. Disponibili: {available_models}")
            
            return is_available
        except Exception as e:
            logger.error(f"Errore verifica modello: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Ritorna informazioni sul modello"""
        try:
            return self.client.show(self.model)
        except Exception as e:
            logger.error(f"Errore recupero info modello: {e}")
            return {}


