# Site de predição YOLO (FastAPI)

## Rodar local
1) Crie/ative um ambiente virtual e instale dependências:
```bash
pip install -r requirements.txt
```
2) Inicie a API com o modelo padrão:
```bash
uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
```
3) Acesse http://localhost:8000 e envie uma imagem. O endpoint principal é `POST /predict`.

Variáveis úteis:
- `MODEL_PATH`: caminho alternativo para o peso (.pt). Padrão: `yolo_dataset/train_chassi_detect2/weights/best.pt`.

## Deploy na Vercel (serverless Python)
1) Instale a CLI: `npm i -g vercel`.
2) Faça login: `vercel login`.
3) A partir da raiz do projeto, rode:
```bash
vercel --prod
```
O `vercel.json` já redireciona todas as rotas para `api/index.py`, que serve tanto o frontend (HTML em `api/static`) quanto a API. A Vercel instalará `requirements.txt` automaticamente.

### Observações
- O peso `best.pt` (caminho padrão) é pequeno, mas Torch e Ultralytics aumentam o tamanho do bundle; use `.vercelignore` para evitar enviar datasets enormes.
- A execução será em CPU; para cargas maiores considere um serviço com GPU ou um deploy em contêiner.
