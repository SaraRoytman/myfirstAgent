import os
from groq import Groq


MODEL = "llama-3.3-70b-versatile"

def main():
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Set the API_KEY environment variable first.")
        return

    
    client = Groq(api_key=api_key)
    
    
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]

    print("Groq Agent ready. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        
        messages.append({"role": "user", "content": user_input})
        
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        
        
        reply = response.choices[0].message.content
        print(f"\nAgent: {reply}\n")
        
        
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
