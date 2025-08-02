"""
phi_bloom_bridge.py
====================

This module provides a simple bridging interface between a symbolic BLOOM-based
neural module and a natural‑language processing module exposed via
the Ollama Python library.  The goal is to allow outputs from the
Phi‑Coder NLP layer to be consumed by the BLOOM engine (for example to
update Bloom vector directives) and, in turn, feed the BLOOM
engine’s manifest back into the NLP layer for recursive refinement.

The design here is intentionally minimal and composable: callers can
subclass ``BloomModelInterface`` and ``PhiModelInterface`` to
customise the actual generation behaviour, or swap in different
implementations (e.g. HuggingFace Transformers, custom BLOOM
implementations, or remote API endpoints).

Note:
  • This file is a proposed bridge for the Phi‑Coder project and does not
    replace any existing implementation.  The real BLOOM module may
    already expose a suitable API; adjust the ``BloomModelInterface``
    accordingly.
  • Ollama must be installed and running for the default
    ``OllamaPhiModel`` to function.  See the Ollama Python library
    documentation for details【230781044008479†L270-L314】.

Usage example:

.. code-block:: python

    from phi_bloom_bridge import BloomModelInterface, OllamaPhiModel, PhiBloomBridge

    # Initialise BLOOM and Phi models
    bloom_model = BloomModelInterface()
    phi_model   = OllamaPhiModel(model_name='phi')

    # Create a bridge
    bridge = PhiBloomBridge(bloom=bloom_model, phi=phi_model)

    # Process a user prompt
    response = bridge.process("Describe harmonic recursion in Φπε")
    print(response)

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
    # Import within try/except so that this module can be imported
    # even if ollama is not installed.  The default implementation
    # will raise if used without the dependency.
    from ollama import chat as ollama_chat
except ImportError:  # pragma: no cover -- handled at runtime
    ollama_chat = None  # type: ignore


@dataclass
class BloomModelInterface:
    """Abstract interface for a BLOOM‑like model.

    The BLOOM model is responsible for converting symbolic directives
    (e.g. from the Φπε field or Phi‑Coder) into execution‑ready
    manifests.  Concrete implementations should override
    ``generate_from_manifest``.
    """

    # Optionally pass model configuration
    model_name: str = field(default="bigscience/bloom-560m")

    def generate_from_manifest(self, manifest: str) -> str:
        """Generate a response from a manifest string.

        A default implementation is provided using HuggingFace
        Transformers.  Override this method if your project uses a
        different BLOOM engine or requires bespoke logic.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch must be installed to use the default BLOOM model"
            ) from e

        # Lazy load model/tokenizer to avoid heavy initialisation on import
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        inputs = tokenizer.encode(manifest, return_tensors="pt")
        # Generate a continuation; adjust parameters as needed
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_length=256, do_sample=True, temperature=0.7
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


@dataclass
class PhiModelInterface:
    """Abstract interface for a Phi‑Coder NLP module.

    Concrete implementations should override ``generate`` to return
    NLP responses for a given prompt.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OllamaPhiModel(PhiModelInterface):
    """Default Phi‑Coder implementation using the Ollama Python library.

    Args:
        model_name: Name of the local Ollama model to use (e.g. ``'phi'``).
        system_prompt: Optional system prompt passed to the model on the
            first message, encoding high‑level instructions (for example,
            alignment with Φπε logic or recursion constraints).
        host: Host URL of the running Ollama server (defaults to
            ``http://localhost:11434``).
    """

    model_name: str = field(default="phi")
    system_prompt: str | None = field(default=None)
    host: str = field(default="http://localhost:11434")
    _initialised: bool = field(init=False, default=False)

    def _ensure_available(self) -> None:
        if ollama_chat is None:
            raise RuntimeError(
                "Ollama is not installed; run `pip install ollama` and ensure"
                " the Ollama daemon is running."
            )

        # Optionally, check that the model is available by listing or pulling
        # The Python API provides `ollama.pull(model)` but requires network
        # access; we avoid this here to prevent side effects.

    def generate(self, prompt: str) -> str:
        self._ensure_available()
        messages: List[Dict[str, str]] = []
        # Prepend a system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use the Ollama Python API to chat with the model
        response = ollama_chat(
            model=self.model_name,
            messages=messages,
            host=self.host,
        )
        # The response is a dictionary with a message field
        return response["message"]["content"]


@dataclass
class PhiBloomBridge:
    """Bridge class to connect Phi‑Coder NLP outputs to BLOOM manifests.

    This class orchestrates a two‑step process:

    1. It sends the user prompt (or the previous BLOOM output) to the
       Phi‑Coder model to obtain an NLP interpretation or refinement.
    2. It forwards the returned content to the BLOOM model, producing
       a new manifest or reply.

    By alternating between these two steps, callers can implement
    recursive interactions (e.g. iterative refinement of manifests or
    conversational flows).  The ``process`` method performs a single
    round‑trip; callers can loop over it as needed.
    """

    bloom: BloomModelInterface
    phi: PhiModelInterface

    def process(self, prompt: str) -> Dict[str, str]:
        """Process a prompt through the Phi → BLOOM pipeline.

        Args:
            prompt: A natural‑language query or command to send into the
                Phi‑Coder layer.

        Returns:
            A dictionary with two keys:
              * ``phi_output`` – the raw content returned by the Phi model.
              * ``bloom_output`` – the BLOOM model’s response given the
                ``phi_output`` as input manifest.
        """

        # Step 1: Generate output from Phi‑Coder
        phi_output = self.phi.generate(prompt)
        # Step 2: Feed that output into BLOOM
        bloom_output = self.bloom.generate_from_manifest(phi_output)
        return {
            "phi_output": phi_output,
            "bloom_output": bloom_output,
        }


__all__ = [
    "BloomModelInterface",
    "PhiModelInterface",
    "OllamaPhiModel",
    "PhiBloomBridge",
]
