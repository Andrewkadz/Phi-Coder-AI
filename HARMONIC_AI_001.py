import sys
from data_core.hemispheric_bloom import hemispheric_bloom_cycle, RecursionPacket
try:
    # Import Ollama Python API for the Phi model
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None

def generate_bloom_manifest(user_input: str):
    """Process the user input through the Bloom (hemispheric) layer to get a manifest."""
    # Prepare the recursion packet with the input
    packet = RecursionPacket(user_input)
    # Seed an initial L4 logic vector (symbols guiding the Bloom process)
    packet.annotations["L4_logic_vector"] = [
        {"symbol": "Φ", "entropy_resolution": "collapse", "depth": 2, "memory_tag": "root"},
        {"symbol": "Ψ", "entropy_resolution": "collapse", "depth": 1, "memory_tag": "branch"},
        # Additional symbols (e.g. Θ) can be included if needed for deeper logic
    ]
    # Run the hemispheric Bloom cycle (layers 5–8) to produce the manifest
    result_packet = hemispheric_bloom_cycle(packet)
    # Extract the Bloom manifest (symbolic directives) from the result
    manifest = result_packet.annotations.get("bloom_manifest", {})
    return manifest

def query_phi_coder(manifest: dict, model_name: str = "phi", host: str = "http://localhost:11434"):
    """Send the Bloom manifest to the Phi LLM (via Ollama) and get the model's response."""
    if ollama_chat is None:
        raise RuntimeError("Ollama library is not installed or available.")
    # Format the manifest into a prompt string for the LLM
    # (We include a header and list each directive line for clarity)
    prompt_lines = ["🧬 Bloom Manifest – auto-generated from input –"]
    for directive in manifest.get("linear_directives", []):
        line = f"- [{directive['priority']}] {directive['action']} ({directive['tag']} @ {directive['path']}) | confidence: {directive['confidence']}"
        prompt_lines.append(line)
    prompt_str = "\n".join(prompt_lines)
    # Prepare the message payload for the Ollama chat API
    messages = [{"role": "user", "content": prompt_str}]
    # Optionally, a system prompt could be prepended here via {"role": "system", "content": "..."} if needed
    # Call the Ollama API to generate a response from the Phi model
    response = ollama_chat(model=model_name, messages=messages, host=host)
    # Extract the content of the assistant's message (Phi model's answer)
    return response["message"]["content"]

def main():
    # Get user input from command-line arguments or prompt if not provided
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("💬 Enter your query: ").strip()
    if not user_query:
        print("No input provided. Exiting.")
        return
    # 1. Generate Bloom manifest from the user query
    manifest = generate_bloom_manifest(user_query)
    # 2. Send manifest to Phi LLM and get a response
    try:
        phi_response = query_phi_coder(manifest)
    except Exception as e:
        phi_response = f"(Error during Phi model query: {e})"
    # 3. Output the response
    print(phi_response)

if __name__ == "__main__":
    main()
