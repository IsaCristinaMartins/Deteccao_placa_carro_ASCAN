## Antes de rodar o código é preciso analisar os metadados do vídeo:
    - vá no vídeo, clique com o botão direito do mouse e em "detalhes" veja os metadados. 
        - Comprimento: é a duração do vídeo. Impacto: junto com o FPS, definirá o número total de quadros disponível (total_quadros =  duração x fps)
        - Altura do quadro:  é a resolução vertical. Impacto: imagens maiores = mais detalhes, porém mais necessidade de processamento. Para a OCR, mais pixels = placa mais nítida (bom) mas, pode ser necessário redimensionar par anão ficar mais pesado. 
        - Largura: é a resolução horizontal. Impacto: junto com a altura definirá o tamanho do frame. (altura x largura = tam de pixels por frame)
        - Taxa de dados: indica quantidade de dados de vídeo processado por segundo (sem áudio). Impacto: quanto maior, mais qualidade. Para OCR, taxa de dados muito baixa interfere nos artefatos (imagem borrada), oque atrapalha na leitura da placa. 
        - Taxa de bits total: inclui vídeo + áudio. Para esse projeto só interessa a parte do vídeo, mas é bom saber que tem a parte do ádio também. 
        - Taxa de quadros: quantos quadros por segundo cada vídeo tem. Impacto: quanto mais quadros, mais chances de pegar uma placa nítida, porém é mais arquivos para serem processados. 


