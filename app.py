from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify
import os
import requests
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

HF_TOKEN = os.environ.get('HF_TOKEN_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')

# Função para consultar Gemini (Google AI)
def consultar_gemini(pergunta):
    if not GEMINI_API_KEY:
        return 'Chave da API Gemini não configurada. Para usar o assistente IA, configure a variável GEMINI_API_KEY no arquivo .env. Por enquanto, aqui está uma resposta simulada sobre agricultura: A agricultura moderna pode ser melhorada através de técnicas como irrigação inteligente, monitoramento de solo, e uso de fertilizantes balanceados. O pH do solo ideal varia entre 6.0 e 7.0 para a maioria das culturas.'
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
    if not HF_TOKEN:
        # Retorna sentimento simulado se não houver chave configurada
        import random
        sentimentos = ['positive', 'negative', 'neutral']
        return random.choice(sentimentos)
    
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
        # Retorna notícias simuladas se não houver chave configurada
        return [
            {
                'title': f'Notícia sobre {termo} - Preços em alta',
                'snippet': f'Os preços de {termo} mostraram tendência de alta no mercado internacional devido a fatores climáticos e demanda crescente.'
            },
            {
                'title': f'Produção de {termo} aumenta no Brasil',
                'snippet': f'A produção brasileira de {termo} registrou crescimento significativo na última safra, impulsionada por condições climáticas favoráveis.'
            },
            {
                'title': f'Exportações de {termo} batem recorde',
                'snippet': f'As exportações de {termo} atingiram volume recorde este ano, com destaque para o mercado asiático.'
            }
        ]
    
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

@app.route('/analisar_csv', methods=['POST'])
def analisar_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not file.filename or not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Arquivo deve ser CSV'}), 400
        
        # Ler o arquivo CSV
        df = pd.read_csv(file.stream)
        
        # Informações básicas do dataset
        dataset_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum())
        }
        
        # Gerar gráficos
        charts = []
        
        # 1. Histograma da primeira coluna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            df[numeric_cols[0]].hist(ax=ax, bins=20, color='#6b9f6b', alpha=0.7)
            ax.set_title(f'Distribuição de {numeric_cols[0]}', fontsize=14, fontweight='bold')
            ax.set_xlabel(str(numeric_cols[0]))
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Converter para base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            img.seek(0)
            chart1_b64 = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            charts.append({
                'title': f'Distribuição de {numeric_cols[0]}',
                'data': f'data:image/png;base64,{chart1_b64}'
            })
        
        # 2. Matriz de correlação se houver múltiplas colunas numéricas
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax, fmt='.2f')
            ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')
            
            # Converter para base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            img.seek(0)
            chart2_b64 = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            charts.append({
                'title': 'Matriz de Correlação',
                'data': f'data:image/png;base64,{chart2_b64}'
            })
        
        # Estatísticas descritivas
        statistics = {}
        for col in numeric_cols:
            stats = df[col].describe()
            statistics[col] = {
                'mean': f"{stats['mean']:.2f}",
                'std': f"{stats['std']:.2f}",
                'min': f"{stats['min']:.2f}",
                'max': f"{stats['max']:.2f}"
            }
        
        return jsonify({
            'success': True,
            'dataset_info': dataset_info,
            'charts': charts,
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 