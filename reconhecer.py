"""
Vision Assist AI - Reconhecimento de objetos em imagens.

Processa imagens localmente com YOLOv8n e apresenta os resultados somente no terminal.
Não utiliza APIs externas, não abre janelas e não reproduz áudio.

Uso:
    python reconhecer.py caminho/da/imagem.jpg
    python reconhecer.py foto.png --confianca 0.5
    python reconhecer.py foto.jpg --json

Uso em outro código:
    from reconhecer import DetectorDeObjetos
    detector = DetectorDeObjetos()
    resultado = detector.analisar("foto.jpg")
    print(resultado["descricao"])
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TAMANHO_MAXIMO_MB = 10
TAMANHO_MAXIMO_BYTES = TAMANHO_MAXIMO_MB * 1024 * 1024

AVISO = (
    "Sistema experimental. As identificações podem conter erros e não devem ser "
    "usadas como garantia de que um caminho esteja livre ou seja seguro."
)

# Tradução das classes do COCO: nome em inglês -> (singular, plural, gênero).
TRADUCAO = {
    "person": ("pessoa", "pessoas", "f"),
    "bicycle": ("bicicleta", "bicicletas", "f"),
    "car": ("carro", "carros", "m"),
    "motorcycle": ("moto", "motos", "f"),
    "airplane": ("avião", "aviões", "m"),
    "bus": ("ônibus", "ônibus", "m"),
    "train": ("trem", "trens", "m"),
    "truck": ("caminhão", "caminhões", "m"),
    "boat": ("barco", "barcos", "m"),
    "traffic light": ("semáforo", "semáforos", "m"),
    "fire hydrant": ("hidrante", "hidrantes", "m"),
    "stop sign": ("placa de pare", "placas de pare", "f"),
    "parking meter": ("parquímetro", "parquímetros", "m"),
    "bench": ("banco", "bancos", "m"),
    "bird": ("pássaro", "pássaros", "m"),
    "cat": ("gato", "gatos", "m"),
    "dog": ("cachorro", "cachorros", "m"),
    "horse": ("cavalo", "cavalos", "m"),
    "sheep": ("ovelha", "ovelhas", "f"),
    "cow": ("vaca", "vacas", "f"),
    "elephant": ("elefante", "elefantes", "m"),
    "bear": ("urso", "ursos", "m"),
    "zebra": ("zebra", "zebras", "f"),
    "giraffe": ("girafa", "girafas", "f"),
    "backpack": ("mochila", "mochilas", "f"),
    "umbrella": ("guarda-chuva", "guarda-chuvas", "m"),
    "handbag": ("bolsa", "bolsas", "f"),
    "tie": ("gravata", "gravatas", "f"),
    "suitcase": ("mala", "malas", "f"),
    "frisbee": ("frisbee", "frisbees", "m"),
    "skis": ("esqui", "esquis", "m"),
    "snowboard": ("snowboard", "snowboards", "m"),
    "sports ball": ("bola", "bolas", "f"),
    "kite": ("pipa", "pipas", "f"),
    "baseball bat": ("taco de beisebol", "tacos de beisebol", "m"),
    "baseball glove": ("luva de beisebol", "luvas de beisebol", "f"),
    "skateboard": ("skate", "skates", "m"),
    "surfboard": ("prancha de surfe", "pranchas de surfe", "f"),
    "tennis racket": ("raquete de tênis", "raquetes de tênis", "f"),
    "bottle": ("garrafa", "garrafas", "f"),
    "wine glass": ("taça de vinho", "taças de vinho", "f"),
    "cup": ("copo", "copos", "m"),
    "fork": ("garfo", "garfos", "m"),
    "knife": ("faca", "facas", "f"),
    "spoon": ("colher", "colheres", "f"),
    "bowl": ("tigela", "tigelas", "f"),
    "banana": ("banana", "bananas", "f"),
    "apple": ("maçã", "maçãs", "f"),
    "sandwich": ("sanduíche", "sanduíches", "m"),
    "orange": ("laranja", "laranjas", "f"),
    "broccoli": ("brócolis", "brócolis", "m"),
    "carrot": ("cenoura", "cenouras", "f"),
    "hot dog": ("cachorro-quente", "cachorros-quentes", "m"),
    "pizza": ("pizza", "pizzas", "f"),
    "donut": ("rosquinha", "rosquinhas", "f"),
    "cake": ("bolo", "bolos", "m"),
    "chair": ("cadeira", "cadeiras", "f"),
    "couch": ("sofá", "sofás", "m"),
    "potted plant": ("planta em vaso", "plantas em vaso", "f"),
    "bed": ("cama", "camas", "f"),
    "dining table": ("mesa", "mesas", "f"),
    "toilet": ("vaso sanitário", "vasos sanitários", "m"),
    "tv": ("televisão", "televisões", "f"),
    "laptop": ("notebook", "notebooks", "m"),
    "mouse": ("mouse", "mouses", "m"),
    "remote": ("controle remoto", "controles remotos", "m"),
    "keyboard": ("teclado", "teclados", "m"),
    "cell phone": ("celular", "celulares", "m"),
    "microwave": ("micro-ondas", "micro-ondas", "m"),
    "oven": ("forno", "fornos", "m"),
    "toaster": ("torradeira", "torradeiras", "f"),
    "sink": ("pia", "pias", "f"),
    "refrigerator": ("geladeira", "geladeiras", "f"),
    "book": ("livro", "livros", "m"),
    "clock": ("relógio", "relógios", "m"),
    "vase": ("vaso", "vasos", "m"),
    "scissors": ("tesoura", "tesouras", "f"),
    "teddy bear": ("urso de pelúcia", "ursos de pelúcia", "m"),
    "hair drier": ("secador de cabelo", "secadores de cabelo", "m"),
    "toothbrush": ("escova de dentes", "escovas de dentes", "f"),
}

_NUMEROS_COMUNS = {
    3: "três", 4: "quatro", 5: "cinco", 6: "seis", 7: "sete",
    8: "oito", 9: "nove", 10: "dez",
}
NUMEROS = {
    "m": {1: "um", 2: "dois", **_NUMEROS_COMUNS},
    "f": {1: "uma", 2: "duas", **_NUMEROS_COMUNS},
}


class ErroDeImagem(Exception):
    """Erro de validação ou leitura da imagem."""


def _traduzir(nome_en):
    return TRADUCAO.get(nome_en, (nome_en, nome_en, "m"))


def _frase_quantidade(nome_en, quantidade):
    singular, plural, genero = _traduzir(nome_en)
    if quantidade == 1:
        return f"{NUMEROS[genero][1]} {singular}"
    numero = NUMEROS[genero].get(quantidade, str(quantidade))
    return f"{numero} {plural}"


def _juntar_frases(frases):
    if len(frases) == 1:
        return frases[0]
    return ", ".join(frases[:-1]) + " e " + frases[-1]


class DetectorDeObjetos:
    """Carrega o YOLO uma vez e reutiliza o modelo nas próximas análises."""

    def __init__(self, modelo="yolov8n.pt", confianca_minima=0.40):
        if not 0 < confianca_minima <= 1:
            raise ValueError("A confiança mínima deve estar entre 0 e 1.")

        self.confianca_minima = confianca_minima

        # Procura o modelo junto deste script, independentemente da pasta atual.
        caminho_modelo = Path(modelo).expanduser()
        if not caminho_modelo.is_absolute():
            modelo_ao_lado_do_script = Path(__file__).resolve().parent / caminho_modelo
            if modelo_ao_lado_do_script.is_file():
                caminho_modelo = modelo_ao_lado_do_script

        if caminho_modelo.suffix.lower() == ".pt" and not caminho_modelo.is_file():
            raise FileNotFoundError(
                f"Pesos do modelo não encontrados: {caminho_modelo}. "
                "Coloque yolov8n.pt na mesma pasta de reconhecer.py."
            )

        self.modelo = YOLO(str(caminho_modelo))

    def _ler_arquivo(self, caminho):
        caminho = Path(caminho).expanduser()

        if not caminho.is_file():
            raise ErroDeImagem(f"Arquivo não encontrado: {caminho}")

        if caminho.suffix.lower() not in EXTENSOES_VALIDAS:
            validas = ", ".join(sorted(EXTENSOES_VALIDAS))
            raise ErroDeImagem(f"Formato não suportado ({caminho.suffix}). Use: {validas}")

        tamanho = caminho.stat().st_size
        if tamanho == 0:
            raise ErroDeImagem("O arquivo está vazio.")
        if tamanho > TAMANHO_MAXIMO_BYTES:
            raise ErroDeImagem(f"Imagem muito grande. Máximo: {TAMANHO_MAXIMO_MB} MB.")

        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise ErroDeImagem("Não foi possível abrir a imagem. Verifique se o arquivo está válido.")
        return imagem

    def _ler_bytes(self, dados):
        if not isinstance(dados, (bytes, bytearray, memoryview)):
            raise ErroDeImagem("A imagem precisa ser fornecida em bytes.")
        if not dados:
            raise ErroDeImagem("A imagem recebida está vazia.")
        if len(dados) > TAMANHO_MAXIMO_BYTES:
            raise ErroDeImagem(f"Imagem muito grande. Máximo: {TAMANHO_MAXIMO_MB} MB.")

        imagem = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
        if imagem is None:
            raise ErroDeImagem("Não foi possível decodificar a imagem recebida.")
        return imagem

    def _detectar(self, imagem):
        try:
            resultados = self.modelo.predict(
                source=imagem,
                conf=self.confianca_minima,
                verbose=False,
            )
        except Exception as erro:
            raise RuntimeError(f"Falha ao executar o modelo YOLO: {erro}") from erro

        resultado = resultados[0]
        agrupado = defaultdict(lambda: {"quantidade": 0, "confianca": 0.0})

        if resultado.boxes is not None:
            for caixa in resultado.boxes:
                classe = int(caixa.cls[0].item())
                nome_en = resultado.names[classe]
                confianca = float(caixa.conf[0].item())
                agrupado[nome_en]["quantidade"] += 1
                agrupado[nome_en]["confianca"] = max(
                    agrupado[nome_en]["confianca"], confianca
                )

        objetos = []
        for nome_en, info in agrupado.items():
            singular, plural, _ = _traduzir(nome_en)
            objetos.append({
                "nome": singular if info["quantidade"] == 1 else plural,
                "nome_original": nome_en,
                "quantidade": info["quantidade"],
                "confianca": round(info["confianca"], 2),
            })

        objetos.sort(key=lambda objeto: (-objeto["quantidade"], -objeto["confianca"]))
        return objetos

    def _montar_descricao(self, objetos):
        if not objetos:
            return (
                "Nenhum objeto conhecido foi identificado na imagem. "
                "Tente melhorar a iluminação, aproximar a câmera ou mudar o ângulo."
            )

        frases = [
            _frase_quantidade(objeto["nome_original"], objeto["quantidade"])
            for objeto in objetos
        ]
        total = sum(objeto["quantidade"] for objeto in objetos)
        titulo = "OBJETO IDENTIFICADO" if total == 1 else "OBJETOS IDENTIFICADOS"
        return f"{titulo}: {_juntar_frases(frases)}."

    def _resposta(self, imagem):
        objetos = self._detectar(imagem)
        return {
            "sucesso": True,
            "total_objetos": sum(objeto["quantidade"] for objeto in objetos),
            "objetos": objetos,
            "descricao": self._montar_descricao(objetos),
            "aviso": AVISO,
        }

    def analisar(self, caminho):
        """Analisa uma imagem salva no computador e retorna os resultados."""
        return self._resposta(self._ler_arquivo(caminho))

    def analisar_bytes(self, dados):
        """Analisa uma imagem já carregada em memória e retorna os resultados."""
        return self._resposta(self._ler_bytes(dados))


def _imprimir_resultado(resultado, exibir_json=False):
    if exibir_json:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 52)
    print("             VISION ASSIST AI")
    print("          RECONHECIMENTO VISUAL")
    print("=" * 52)

    if not resultado.get("sucesso"):
        print("\n[ERRO] A análise não foi concluída.")
        print(resultado.get("erro", "Erro desconhecido."))
        print("=" * 52)
        return

    print("\n[SUCESSO] Imagem analisada.\n")
    print("OBJETOS IDENTIFICADOS:")

    objetos = resultado["objetos"]
    if not objetos:
        print("Nenhum objeto conhecido foi identificado.")
    else:
        for indice, objeto in enumerate(objetos, start=1):
            print(f"\n{indice}. {objeto['nome'].capitalize()}")
            print(f"   Quantidade: {objeto['quantidade']}")
            print(f"   Confiança: {objeto['confianca']:.0%}")

    print("\n" + "-" * 52)
    print("DESCRIÇÃO:")
    print(resultado["descricao"])
    print("\nAVISO:")
    print(resultado["aviso"])
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(
        description="Reconhece objetos em uma imagem e mostra o resultado no terminal."
    )
    parser.add_argument(
        "imagem",
        help="Caminho da imagem (jpg, jpeg, png, bmp ou webp).",
    )
    parser.add_argument(
        "--confianca",
        type=float,
        default=0.40,
        help="Confiança mínima entre 0 e 1 (padrão: 0.40).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Mostra o resultado em formato JSON no terminal.",
    )
    args = parser.parse_args()

    if not 0 < args.confianca <= 1:
        print("Erro: --confianca deve ser maior que 0 e menor ou igual a 1.", file=sys.stderr)
        return 2

    try:
        detector = DetectorDeObjetos(confianca_minima=args.confianca)
        resultado = detector.analisar(args.imagem)
    except ErroDeImagem as erro:
        resultado = {"sucesso": False, "erro": str(erro)}
    except FileNotFoundError as erro:
        resultado = {"sucesso": False, "erro": str(erro)}
    except Exception as erro:
        resultado = {"sucesso": False, "erro": f"Erro de processamento: {erro}"}

    _imprimir_resultado(resultado, exibir_json=args.json)
    return 0 if resultado.get("sucesso") else 1


if __name__ == "__main__":
    sys.exit(main())
