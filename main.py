import ollama
def main():
    print("Hello from llmtest-needleinahaystack!")


if __name__ == "__main__":

    # Use the generate function for a one-off prompt
    result = ollama.generate(model='qwen3:0.6b', prompt='Why is the sky blue?',think = True)
    print(result['response'])
