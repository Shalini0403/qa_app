
# generator.py
from openai import OpenAI

from config import load_config


def init_openai_client():
    """Initialize OpenAI client with credentials from config file."""
    config = load_config()
    return OpenAI(
        api_key=config.get('OPENAI_API_KEY'),
        #organization=config.get('OPENAI_ORG_ID')  # Optional
    )

# Initialize the OpenAI client
client = init_openai_client()

def generate_text(query: str, documents: list) -> str:
    """
    Generate an answer based on the query and provided documents using OpenAI's API.
    """
    context = "\n".join(documents)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based on the context provided."}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."