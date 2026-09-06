"""Groq-hosted LLaMA 3.3 Streaming Inference Client.
Produces fast streaming tokens (low TTFT) strictly grounded in GP policies.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional
from .rag_retriever import PolicyChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TriagePulse, an AI voice receptionist for St. Jude Medical Centre (an NHS GP surgery).
Your primary role is to assist patients with booking appointments, checking surgery hours, and repeat prescriptions.

CRITICAL OPERATIONAL RULES:
1. Speak in a warm, concise, professional British English conversational tone suitable for phone calls.
2. Keep responses brief (1-3 sentences maximum). NEVER use bullet points, markdown bolding, or lists in voice output.
3. STRICTLY ground your information in the PROVIDED SURGERY POLICIES. If an answer is not in the policies, politely state you will transfer them to human reception.
4. NEVER provide clinical medical advice, diagnose illnesses, or suggest specific drug dosages.
5. If the patient asks to book an appointment, collect their full name, preferred time slot (e.g. Tuesday morning), and brief reason for appointment.
"""


class GroqLLMClient:
    """Streams LLaMA 3.3 tokens from Groq Cloud API with local mock fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mock_mode: Optional[bool] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if mock_mode is None:
            mock_env = os.getenv("MOCK_PROVIDERS", "true").lower()
            self.mock_mode = (mock_env in ("true", "1", "yes")) or not bool(self.api_key)
        else:
            self.mock_mode = mock_mode

        self.groq_client = None
        if not self.mock_mode and self.api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}. Defaulting to mock.")
                self.mock_mode = True

    async def stream_response(
        self,
        user_utterance: str,
        policy_chunks: List[PolicyChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens one by one for low Time-to-First-Token (TTFT)."""
        context_str = "\n\n".join([f"[{c.source_file} - {c.section_title}]:\n{c.content}" for c in policy_chunks])
        user_prompt = f"RELEVANT GP POLICIES:\n{context_str}\n\nPATIENT UTTERANCE:\n{user_utterance}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-4:])
        messages.append({"role": "user", "content": user_prompt})

        if not self.mock_mode and self.groq_client:
            try:
                loop = asyncio.get_running_loop()
                stream = await loop.run_in_executor(
                    None,
                    lambda: self.groq_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=250,
                        stream=True,
                    ),
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
                return
            except Exception as e:
                logger.error(f"Groq streaming error: {e}. Falling back to simulation.")

        # High-fidelity mock generator grounded in the query
        async for token in self._simulate_response(user_utterance, policy_chunks):
            yield token

    async def _simulate_response(self, query: str, chunks: List[PolicyChunk]) -> AsyncGenerator[str, None]:
        """Produce realistic receptionist tokens matching the user query."""
        q_lower = query.lower()
        
        if "hour" in q_lower or "open" in q_lower or "time" in q_lower:
            text = (
                "St. Jude Medical Centre is open Monday to Friday from 8:00 AM to 6:30 PM, "
                "with phone lines opening at 8:00 AM. We also hold extended evening clinics "
                "on Tuesdays and Thursdays until 8:00 PM."
            )
        elif "prescription" in q_lower or "medication" in q_lower:
            text = (
                "Repeat prescription requests require two full working days to process. "
                "You can submit your request via the NHS App or online through our practice website."
            )
        elif "book" in q_lower or "appointment" in q_lower or "doctor" in q_lower or "see" in q_lower:
            text = (
                "I can certainly help you book an appointment. "
                "For routine consultations we have slots available next Tuesday at 10:30 AM or Thursday at 2:00 PM. "
                "Could you please confirm your full name?"
            )
        elif "blood" in q_lower or "test" in q_lower:
            text = (
                "Our phlebotomy blood test clinic runs Monday to Friday from 8:30 AM to 12:30 PM with our practice nurse. "
                "Would you like me to book a morning slot for you?"
            )
        else:
            text = (
                "Hello, thank you for calling St. Jude Medical Centre. "
                "How may I assist you with your appointment or inquiry today?"
            )

        # Break into natural token chunks
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            # Simulate Groq TTFT and streaming speed (15-25ms per token)
            await asyncio.sleep(0.02)
            yield token
