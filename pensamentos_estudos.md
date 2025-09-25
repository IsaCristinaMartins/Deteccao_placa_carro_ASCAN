## Checklists
- Pré-processamento do frame, corte do vídeo para "foto"
- Converter para escala de cinza (cv2.cvtColor)
- Remover ruído (blur leve ou filtro bilateral)
- Aumentar contraste/bordas (Canny, Sobel ou equalização de histograma)
- Localizar regiões candidatas à placa
- Detectar contornos retangulares (placas têm formato retangular padrão).
- Filtrar pelo aspect ratio (largura ≈ 2 a 5 vezes maior que altura).
- Descartar regiões muito pequenas/grandes.
- Outra opção: usar classificador pré-treinado (Haar Cascade para placas ou YOLO/SSD para detecção mais moderna).
- Cortar a região da placa
- Depois do passo anterior, você isola só o “retângulo” que parece uma placa.
- Isso gera uma nova imagem, bem menor, que vai para o OCR.
- Pós-processamento da placa cortada
- Binarizar (preto e branco)
- Corrigir inclinação (deskew, se a placa estiver torta)
- Reduzir ruído extra (morfologia: cv2.morphologyEx)
- OCR (Tesseract ou outro)
- Só aqui entra a leitura de caracteres.
- A qualidade depende muito do corte bem feito e do contraste.

## Anotações
- Dúvidas:



- Decisões e pequenos resultados do dia:
        o código até rodou mas, só saiu um frame no vídeo de 16 segundo, foi preciso ajustar para ser retirado mais frames. 


