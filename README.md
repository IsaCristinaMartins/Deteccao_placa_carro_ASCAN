# Reconhecimento e Visualização de Placas Automotivas

![Visualização de Placa](https://sensoreng.com.br/wp-content/uploads/2021/03/lpr-leitura-de-placas-de-veiculos-monitoramento-de-portaria-online-armanzenamento-em-nuvem-300x169.jpg)

## Objetivo
Este projeto tem como objetivo estudar e implementar um sistema de **detecção e reconhecimento automático de placas de veículos (OCR)**, utilizando visão computacional e técnicas de aprendizado de máquina, como requisito parcial para a formatura no programa de aperfeiçoamento de estagiários do Instituto Atlântico AsCan. 


1. **Coleta de Dados**
   - Dataset utilizado para treinamento do Yolo foi disponibilizado pela equipe Atlântico e obtido atraves do link: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection   
   - Dataset utilizado para leitura de placa foi obtido de cortes de um vídeo obtido na web.


2. **Pré-processamento**
   - Extração de regiões de interesse (ROI) das placas, recortadas no padrão Yolo, com +8px, 16px e 32px. 
   - Utilização de frames em cor original, trabalhadas com dois tipos de binarização (Adapt e Otsu).

3. **Visualização Inicial**
   - Exibição de exemplos de placas originais e pré-processadas.


4. **Reconhecimento de Caracteres (OCR)**
   - Aplicação de bibliotecas como **Tesseract OCR**, **PaddleOCR** e **EasyOCR**
   - Teste com diferentes abordagens de pré-processamento para melhorar acurácia.

5. **Visualização dos Resultados**
   - Exibição dos resultados em imagens com as placas detectadas e reconhecidas.
   - Gráficos comparando diferentes abordagens.


## Estrutura do Banco de Imagens (Organograma)

![Organograma Banco de Imagens](./Desafio/imagens_README/organograma_readme_.png)

## Passo a passo:
   1 - Crie um ambiente virtual e rode `requitements.txt` 
   2 - Selecione um vídeo a seu critério e corrija o caminho a ser identificado pelo '1_extraindo_frames.py'
   3 - Selecione um dataset - a seu critério - para o treinamento do modelo Yolo para identificação de placa (local)
   4 - Rode '2_converter_para_yolo.py' caso seu dataset de treinamento do modelo não esteja nos padrões aceitados pelo Yolo
   5 - Rode '3_treinar_yolo.py'.
   6 - Rode '4_inferencia_placas.py'. para trabalhar as imagens das placas e seus respectivos cortes
   7 - Vá em `OCR` e escolha qual OCR você deseja analisar. 






