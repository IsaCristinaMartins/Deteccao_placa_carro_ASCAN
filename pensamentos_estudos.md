## Checklists
- [ ] Passo 1: Estrutura criada + venv pronto
- [ ] Passo 2: Leitura guiada do `main.py`
- [ ] Passo 3: Instalar libs mínimas e extrair frames
              matplotlib: para visualização de imagens e outras coisas
              opencv-python: para visão computacional, usamos para abrir vídeos, extrair frames e manipular imagens. 
              EasyOCR: ferramenta de OCR, vai tentar "ler" o que tme escrito - se tiver - na imagem. 
              TQDM: gera barra de tarefas. útil quando se processa muitos frames pra você ir acompanhando. 
- [ ] Passo 4: Rodar OCR (EasyOCR) com *greedy* e *beamsearch*
- [ ] Passo 5: Avaliar JSON de saída e comparar abordagens

## Anotações
- Dúvidas:



- Decisões e pequenos resultados do dia:
        o código até rodou mas, só saiu um frame no vídeo de 16 segundo, foi preciso ajustar para ser retirado mais frames. 