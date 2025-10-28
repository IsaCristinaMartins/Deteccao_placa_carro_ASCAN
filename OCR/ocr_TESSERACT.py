import os, re, csv, random
from pathlib import Path
from typing import Dict, List
import cv2
import pytesseract

# ======================
# CONFIGURAÇÕES
# ======================
BASE_EXEC = Path(r"C:\Users\isabel_martins\Documents\ASCAN\Comp_Cognitiva\Desafio\runs\detect\placas_varios_recortes")
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



# =====================
#   Básica
# =====================
# TESS_BASE = "--oem 1 --psm 7"
# TESS_CHARS = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# TESS_CONFIG = f"{TESS_BASE} {TESS_CHARS}"
# Configuração básica e genérica, usada quando o texto está bem isolado e limpo.


# ====================
#  Específica
# ====================
# TESS_BASE = "--oem 1 --psm 8 -c load_system_dawg=F -c load_freq_dawg=F"
# TESS_CHARS = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# TESS_CONFIG = f"{TESS_BASE} {TESS_CHARS}"
# melhorar a precisão mantendo certa flexibilidade. Boa quando há ruído, corte imperfeito ou variação no tamanho da placa.

# ====================
#  Agressíva
# ====================
# TESS_BASE = "--oem 1 --psm 8 -c load_system_dawg=F -c load_freq_dawg=F -c classify_bln_numeric_mode=1"
# TESS_CHARS = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_zero_rejection=1 -c tessedit_reject_bad_qualities=0"
# máxima precisão em letras e números típicos de placas. Ele ignora tudo que não pareça placa e aceita caracteres mesmo com qualidade ruim.


TESS_BASE = "--oem 1 --psm 8 -c load_system_dawg=F -c load_freq_dawg=F"
TESS_CHARS = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ==========================
# CAMINHOS / LISTA DE GT   runs\detect\placas_varios_recortes\recorte_justoLimpoOtsu
# ==========================
BASE_EXEC = Path(r"C:\Users\isabel_martins\Documents\ASCAN\Comp_Cognitiva\Desafio\runs\detect\placas_varios_recortes")
PASTA = BASE_EXEC / "recorte_justoLimpoOtsu"   # pasta com as imagens recortadas (frame_XXXXXX...recorte_justo.png/jpg) 
CSV_SAIDA = BASE_EXEC / "resul_TESS_DOIS_JUSTO_OTSU.csv"

PLACAS_GT = [
    {"ini": 0,   "fim": 19,  "placa": "HONDUH"},
    {"ini": 21,  "fim": 42,  "placa": "TIMELESS"},
    {"ini": 43,  "fim": 60,  "placa": "EEWABUG"},
    {"ini": 61,  "fim": 79,  "placa": "PIKACHU"},
    {"ini": 80,  "fim": 100, "placa": "OHIFIT"},
    {"ini": 101, "fim": 117, "placa": "EWGAS"},
    {"ini": 118, "fim": 145, "placa": "OFZELDA"},
    {"ini": 146, "fim": 160, "placa": "BYE4O1K"},
    {"ini": 161, "fim": 183, "placa": "MUAHAHA"},
    {"ini": 184, "fim": 202, "placa": "ORINNIE"},
    {"ini": 203, "fim": 224, "placa": "NBEYOND"},
    {"ini": 225, "fim": 243, "placa": "TOYYODA"},
    {"ini": 244, "fim": 258, "placa": "INEED2P"},
    {"ini": 259, "fim": 307, "placa": "88B88BB"},
]

# ==========================
# MÉTRICAS 
# ==========================
def limpar(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', (s or "").upper())

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[lb]

def acc_por_letra(pred: str, gt: str) -> float:
    pred, gt = limpar(pred), limpar(gt)
    N = max(len(pred), len(gt))
    if N == 0:
        return 0.0
    dist = levenshtein(pred, gt)
    return (1.0 - dist / N) * 100.0

# ==========================
# INDEXAÇÃO POR FRAME
# ==========================
def index_por_arquivo(pasta: Path) -> Dict[int, List[Path]]:
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não existe: {pasta.resolve()}")
    mapa: Dict[int, List[Path]] = {}
    # padrões com e sem bloco intermediário
    for padrao in ["frame_*_*recorte_justo.*", "frame_*recorte_justo.*"]:
        for arq in pasta.glob(padrao):
            m = re.search(r"frame_(\d{1,6})", arq.name)
            if m:
                idx = int(m.group(1))
                mapa.setdefault(idx, []).append(arq)
    return mapa

# ==========================
# OCR COM TESSERACT (retorna texto_bruto e confiança média 0–1)
# ==========================
def ler_tesseract(img_bgr):
    # Usamos image_to_data para extrair confs palavra a palavra
    data = pytesseract.image_to_data(img_bgr, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
    palavras = [w for w in data.get("text", []) if w and w.strip()]
    texto_bruto = " ".join(palavras) if palavras else ""

    confs_validas = []
    for c in data.get("conf", []):
        try:
            ic = int(c)
            if ic >= 0:          # Tesseract usa -1 para "sem confiança"
                confs_validas.append(ic)
        except:
            pass

    conf_media_0a1 = (sum(confs_validas)/len(confs_validas)/100.0) if confs_validas else 0.0
    return texto_bruto, conf_media_0a1

# ==========================
# PROCESSAMENTO POR INTERVALO (2 imagens/intervalo)
# ==========================
def processar_intervalo(writer: csv.DictWriter,
                        mapa: Dict[int, List[Path]],
                        ini: int, fim: int, placa_gt: str):
    frames_no_range = sorted([i for i in mapa.keys() if ini <= i <= fim])
    if len(frames_no_range) == 0:
        return

    # Seleciona exatamente 2 frames quando possível
    escolhidos = frames_no_range if len(frames_no_range) <= 2 else random.sample(frames_no_range, 2)

    for idx in escolhidos:
        arquivo = sorted(mapa[idx])[0]  # se houver mais de um arquivo pro mesmo frame, pega o primeiro
        img = cv2.imread(str(arquivo))
        if img is None:
            writer.writerow({
                "intervalo": f"{ini}-{fim}",
                "texto_bruto": "",
                "texto_limpo": "",
                "placa": placa_gt,
                "confianca_ocr": 0.0,
                "confianca_letras": 0.0,
                "confianca_placa": 0.0
            })
            continue

        texto_bruto, conf_ocr_0a1 = ler_tesseract(img)
        texto_limpo = limpar(texto_bruto) if texto_bruto else ""
        conf_letras = acc_por_letra(texto_limpo, placa_gt)
        conf_placa  = 100.0 if (texto_limpo and texto_limpo == limpar(placa_gt)) else 0.0

        writer.writerow({
            "intervalo": f"{ini}-{fim}",
            "texto_bruto": texto_bruto,
            "texto_limpo": texto_limpo,
            "placa": placa_gt,
            "confianca_ocr": round(conf_ocr_0a1, 4),      # 0–1
            "confianca_letras": round(conf_letras, 2),    # %
            "confianca_placa": round(conf_placa, 2),      # %
        })

# ==========================
# MAIN
# ==========================
def main():
    random.seed()  # defina um inteiro aqui se quiser reprodutibilidade
    mapa = index_por_arquivo(PASTA)

    with open(CSV_SAIDA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "intervalo",
                "texto_bruto",
                "texto_limpo",
                "placa",
                "confianca_ocr",      # 0–1
                "confianca_letras",   # %
                "confianca_placa",    # %
            ],
        )
        writer.writeheader()

        for bloco in PLACAS_GT:
            processar_intervalo(
                writer=writer,
                mapa=mapa,
                ini=bloco["ini"],
                fim=bloco["fim"],
                placa_gt=bloco["placa"]
            )

    print(f"\n✅ Concluído! CSV salvo em: {CSV_SAIDA}")

if __name__ == "__main__":
    main()
