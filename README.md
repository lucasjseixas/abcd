# Flask AI App

Este projeto é uma aplicação Python com Flask que consome APIs externas como Hugging Face, SerpAPI e Google Gemini. Ele utiliza variáveis de ambiente para armazenar chaves de API com segurança usando `python-dotenv`.

## Requisitos

- Python 3.8+
- pip
- Acesso à internet

## Instalação

### 1. Clone o repositório

git clone https://github.com/lucasjseixas/abcd.git
cd abcd


2. Crie um ambiente virtual

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

3. Instale as dependências
Crie um arquivo <strong>requirements.txt</strong> com o seguinte conteúdo:

flask
python-dotenv
requests
huggingface_hub

4. Crie um arquivo .env

Crie um arquivo .env na <strong>raiz</strong> do projeto com as suas chaves:

HF_TOKEN_KEY=your_huggingface_api_key
SERPAPI_API_KEY=your_serpapi_api_key
GEMINI_API_KEY=your_gemini_api_key

Como executar

python app.py

Abrir o browser no endereço http://127.0.0.1:5000

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Como obter suas chaves de API:

🔹Hugging Face

    Vá até: https://huggingface.co/settings/tokens

    Clique em "New token" e selecione o escopo necessário.

    Copie o token gerado e cole no .env como HF_TOKEN_KEY.

🔹 SerpAPI (Google Search API)

    Acesse: https://serpapi.com/users/sign_up

    Após se registrar, acesse: https://serpapi.com/dashboard

    Copie sua API Key e cole no .env como SERPAPI_API_KEY.

🔹 Gemini (Google AI / Generative Language)

    Vá para: https://aistudio.google.com/app/apikey

    Faça login com sua conta Google.

    Clique em "Create API key".

    Copie o token gerado e cole no .env como GEMINI_API_KEY.