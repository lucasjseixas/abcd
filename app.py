from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify
import os
import requests
from huggingface_hub import InferenceClient

app = Flask(__name__)

HF_TOKEN = os.environ.get('HF_TOKEN_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')

# Função para consultar Gemini (Google AI)
def consultar_gemini(pergunta):
    if not GEMINI_API_KEY:
        return 'Erro: chave da API Gemini não configurada.'
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}'
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": pergunta}]}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            return 'Erro ao interpretar resposta da IA.'
    else:
        return 'Erro na API Gemini: ' + response.text

# Função para análise de sentimento usando huggingface_hub
client_hf = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def analisar_sentimento_hf(texto):
    try:
        result = client_hf.text_classification(
            texto,
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
        # O resultado é uma lista de dicts [{'label': 'POSITIVE', 'score': ...}, ...]
        if isinstance(result, list) and len(result) > 0:
            return result[0]['label'].lower()
        return 'indefinido'
    except Exception as e:
        return f'erro: {str(e)}'

# Função para buscar notícias na SerpApi
def buscar_noticias_serpapi(termo, num=5):
    if not SERPAPI_KEY:
        return []
    url = 'https://serpapi.com/search.json'
    params = {
        'q': termo,
        'tbm': 'nws',
        'api_key': SERPAPI_KEY,
        'num': num
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        noticias = response.json().get('news_results', [])
        return noticias
    return []

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para chat IA (Gemini)
@app.route('/ask_ia', methods=['POST'])
def ask_ia():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Pergunta não fornecida.'}), 400
    question = data.get('question')
    resposta = consultar_gemini(question)
    return jsonify({'answer': resposta})

@app.route('/sentimento_noticias', methods=['POST'])
def sentimento_noticias():
    data = request.get_json()
    termo = data.get('termo') if data else None
    if not termo:
        return jsonify({'error': 'Termo não fornecido.'}), 400
    noticias = buscar_noticias_serpapi(termo, num=5)
    resultados = []
    for noticia in noticias:
        titulo = noticia.get('title', '')
        snippet = noticia.get('snippet', '')
        sentimento = analisar_sentimento_hf(f"{titulo}. {snippet}")
        resultados.append({
            'title': titulo,
            'snippet': snippet,
            'sentimento': sentimento
        })
    return jsonify({'resultados': resultados})

if __name__ == '__main__':
    app.run(debug=True) 