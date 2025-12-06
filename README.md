# AtividadeCompGraf9_1

Uso:
    python motion_direction_monitor.py                    # webcam
    python motion_direction_monitor.py --source video.mp4 # arquivo

Parâmetros (exemplos):
    --source: caminho para vídeo ou índice da câmera (default 0)
    --min-area: área mínima do contorno para considerar um objeto (default 1500)
    --alpha: taxa de aprendizado do background (0-1), menor -> background mais estável (default 0.01)
    --history: número de frames para usar no cálculo de tendência (default 8)
    --area-threshold: variação relativa mínima da área para considerar aproximação/recuo (default 0.08)

Descrição das classes:
    - BackgroundModel: cria e atualiza um background por média acumulada com correção de iluminação.
    - MotionDetector: calcula máscara de movimento a partir do background e limpa ruído.
    - ObjectTracker: encontra o maior contorno válido, calcula área e bounding box, mantém histórico para inferência.
    - EdgeProcessor: aplica Sobel e Canny no ROI do objeto.
    - Visualizer: desenha janelas e sobreposições com status (área, direção, velocidade aproximada).