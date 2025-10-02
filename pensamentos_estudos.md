## Checklists
# - Extrair frames (10 fps)
#  - Gerar imagens únicas a partir do vídeo.
- Pré-processar
  - Remover ruído (blur leve ou filtro bilateral).
  - Aumentar contraste/bordas (equalização de histograma/CLAHE e Canny/Sobel).
- Localizar regiões candidatas à placa
  - A partir do mapa de bordas + contornos.
- Detectar contornos retangulares
  - Placas têm formato retangular padrão (aproximação poligonal).
- Filtrar por aspect ratio e tamanho
  - Largura ≈ 2 a 5× a altura.
  - Descartar regiões muito pequenas/grandes.
- Outra opção (em vez de contornos)
  - Usar classificador pré-treinado:
    - Haar Cascade (trabalha em cinza) (NÃO FOI A ESCOLHIDA) ou


    - YOLO/SSD (detecção moderna; pode usar imagem colorida; ideal treinar/avaliar com o dataset do Kaggle).
- Cortar a região da placa
  - Isolar o “retângulo” que parece placa (com pequena margem).
  - Gera uma nova imagem menor que vai para o OCR.
- Pós-processar a placa cortada
  - Binarizar (preto e branco, ex.: Otsu).
  - Corrigir inclinação (deskew/retificação de perspectiva).
  - Reduzir ruído extra (morfologia: cv2.morphologyEx).
- OCR (Tesseract ou outro)
  - Só aqui entra a leitura de caracteres.
  - A qualidade depende do corte e do contraste (usar psm de linha única e whitelist A–Z/0–9 ajuda).
- Pós-processamento do texto
  - Normalizar (maiúsculas, remover espaços).
  - Validar por regex de placas BR (AAA-0000 / AAA0A00).
  - Corrigir confusões comuns (O↔0, I↔1, B↔8, S↔5).
  - (Opcional) Voto majoritário entre frames consecutivos do mesmo carro.
- Saídas
  - CSV: frame, bbox, texto_ocr_bruto, texto_placa_final, confianca_ocr, metodo_detecao.
  - (Opcional) Frames anotados com bbox + texto.
- Uso do dataset do Kaggle (exigência)
  - Treinar YOLO/SSD ou avaliar o pipeline nas imagens do dataset e reportar métricas.
- Métricas mínimas
  - Detecção: recall@IoU≥0.5 ou taxa de detecção por imagem.
  - OCR: plate accuracy (match completo) e, opcionalmente, CER.

Nota rápida: nos passos com Canny/Sobel/Haar, trabalhe em escala de cinza; YOLO/SSD pode operar em colorido.


## Anotações
- Dúvidas:



- Decisões e pequenos resultados do dia:
        - converter_para_yolo.py → converte as anotações do Kaggle para formato YOLO
        -  treinar_yolo.py

        - extraindo_frames.py

        - inferencia_placas.py

        - ocr_placas.py


