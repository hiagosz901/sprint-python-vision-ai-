# Vision Assist AI — Reconhecimento de objetos em Python

Módulo independente que recebe uma imagem, identifica objetos usando YOLOv8n e apresenta o resultado **somente no terminal**.

- Não utiliza API externa de inteligência artificial.
- Não abre janela gráfica.
- Não utiliza câmera automaticamente.
- Não reproduz áudio.
- Processa uma imagem por execução.

## Estrutura

```text
vision-assist-ai/
├── reconhecer.py
├── requirements.txt
├── README.md
└── yolov8n.pt
```

## Requisitos

- Python 3.9 ou superior (recomendado Python 3.10 ou 3.11).
- Windows, Linux ou macOS.
- Conexão com a internet para instalar as dependências, caso ainda não estejam instaladas.

## Instalação no Windows

Abra o terminal na pasta do projeto e execute:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Como executar

Coloque a imagem na pasta do projeto ou informe o caminho completo dela.

```bash
python reconhecer.py foto.jpg
```

Outros exemplos:

```bash
python reconhecer.py imagens/ambiente.png
python reconhecer.py foto.jpg --confianca 0.5
python reconhecer.py foto.jpg --json
```

A saída padrão é um relatório organizado no terminal. A opção `--json` apresenta os dados em JSON, também no terminal.

## Uso em outro código Python

```python
from reconhecer import DetectorDeObjetos

detector = DetectorDeObjetos()  # carrega o modelo uma única vez
resultado = detector.analisar("foto.jpg")

print(resultado["descricao"])
print(resultado["objetos"])
```

Também é possível analisar bytes de uma imagem que já esteja carregada em memória:

```python
with open("foto.jpg", "rb") as arquivo:
    dados = arquivo.read()

resultado = detector.analisar_bytes(dados)
print(resultado["descricao"])
```

## Sobre o modelo

O arquivo `yolov8n.pt` acompanha este projeto. O programa procura esse arquivo na mesma pasta de `reconhecer.py`, mesmo quando o comando é executado a partir de outra pasta.

A inferência é feita localmente, sem enviar imagens para serviços externos. A instalação inicial das bibliotecas pode exigir internet.

O modelo YOLOv8n pré-treinado no COCO reconhece 80 categorias conhecidas, como pessoas, carros, cadeiras, mesas e celulares. Ele não reconhece todos os objetos existentes nem compreende completamente o contexto de uma cena.

## Limitações e segurança

Este módulo é experimental. As detecções podem estar incorretas ou deixar de identificar objetos. Não deve ser usado como única fonte para navegação, travessia de ruas ou decisões de segurança.

Formatos aceitos: JPG, JPEG, PNG, BMP e WEBP. Tamanho máximo do arquivo: 10 MB.
# sprint-python-vision-ai-
