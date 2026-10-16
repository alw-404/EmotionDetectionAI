# EmotionDetectionAI

Riconoscimento delle emozioni dal volto in tempo reale, tramite webcam.

Il programma estrae i landmark facciali con **MediaPipe Face Mesh**, li usa come feature per un classificatore **Random Forest** (scikit-learn) e mostra a video l'emozione prevista.

Emozioni riconosciute: `ANGRY`, `HAPPY`, `SAD`.

## Come funziona

1. **Estrazione delle feature**: per ogni immagine del dataset, MediaPipe individua 468 landmark del volto. Ogni landmark ha 3 coordinate (x, y, z), quindi ogni immagine diventa un vettore di 1404 numeri.
2. **Addestramento**: i vettori, insieme all'etichetta dell'emozione, vengono divisi in training (80%) e test (20%) e usati per addestrare un Random Forest da 100 alberi.
3. **Test in tempo reale**: il modello salvato viene caricato e applicato ai frame della webcam. L'emozione prevista compare in basso a sinistra nella finestra.

## Struttura del progetto

```
EmotionDetectionAI/
├── dataset/
│   └── faces/
│       ├── ANGRY/      # immagini di volti arrabbiati
│       ├── HAPPY/      # immagini di volti felici
│       └── SAD/        # immagini di volti tristi
└── code/
    ├── util.py            # funzione get_face_landmarks (MediaPipe)
    ├── prepare_data.py    # crea data.txt dalle immagini del dataset
    ├── train_model.py     # addestra il modello e crea code.pkl
    └── test_model.py      # test in tempo reale con la webcam
```

Le cartelle in `dataset/faces/` devono chiamarsi come le emozioni. L'indice di ogni emozione dipende dall'ordine alfabetico delle cartelle (`ANGRY` = 0, `HAPPY` = 1, `SAD` = 2), e deve coincidere con la lista `emotions` in `test_model.py`.

## Requisiti

- Ubuntu (o un'altra distribuzione Linux) con una webcam
- Python 3.9 – 3.12 (le versioni più recenti possono non essere supportate da MediaPipe)
- Pacchetti Python:
  - `opencv-python` (non la versione `headless`, serve `cv2.imshow`)
  - `mediapipe` (in una versione che include `mp.solutions`, per esempio la 0.10.14)
  - `scikit-learn`
  - `numpy`

## Installazione

```bash
cd ~/EmotionDetectionAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python "mediapipe==0.10.14" scikit-learn numpy
```

Per verificare che tutto si importi:

```bash
python -c "import cv2, mediapipe as mp, sklearn, numpy; print(cv2.__version__, mp.solutions.face_mesh)"
```

## Utilizzo

Esegui gli script **in ordine**, dalla cartella `code/` e con il venv attivo:

```bash
cd ~/EmotionDetectionAI/code
source ../.venv/bin/activate

python prepare_data.py   # 1. estrae i landmark e crea data.txt
python train_model.py    # 2. addestra il modello, stampa accuratezza e matrice di confusione, crea code.pkl
python test_model.py     # 3. apre la webcam (premi "q" per uscire)
```

Passi 1 e 2 vanno rifatti solo se cambia il dataset. Il passo 3 richiede che `code.pkl` esista già.

## Risoluzione dei problemi

**`ModuleNotFoundError: No module named 'cv2'` (o `mediapipe`)**
Python sta usando un interprete diverso da quello in cui hai installato i pacchetti. Attiva il venv e installa con `python -m pip install ...`, usando lo stesso `python` con cui lanci gli script.

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Controlla che nella cartella del progetto non ci sia un file o una cartella chiamata `mediapipe`. Se non c'è, installa una versione di MediaPipe che include l'API `solutions`: `python -m pip install "mediapipe==0.10.14"`.

**Conflitto `tensorflow ... requires numpy<2.0.0`**
È un avviso di pip: il progetto non usa TensorFlow. Il modo più semplice per evitarlo è usare un ambiente virtuale dedicato, come descritto sopra.

**La webcam non si apre (`can't open camera by index`, `Camera index out of range`)**
- Controlla che esista il dispositivo: `ls -l /dev/video*`.
- Verifica che sia il nodo video giusto: `v4l2-ctl --list-devices` (richiede `sudo apt install v4l-utils`).
- Apri la camera con il backend esplicito `cv2.VideoCapture(0, cv2.CAP_V4L2)` e assicurati che nello script ci sia **una sola** apertura della camera.
- Chiudi gli altri programmi che usano la webcam (browser, Zoom, Cheese).
- Se `ls -l /dev/video*` mostra solo `root` e il gruppo `video` come autorizzati, aggiungi il tuo utente al gruppo con `sudo usermod -aG video $USER` e poi fai logout/login.

**`NameError: name 'output' is not defined`**
La chiamata a `cv2.putText` deve stare dentro `if face_landmarks:`, così viene eseguita solo quando è stato rilevato un volto.

**Nessuna scritta con il volto davanti alla camera**
Stampa `len(face_landmarks)`: deve valere `1404`. Se vale `0` il volto non viene rilevato; se è un altro numero (per esempio 1434) il modello è stato addestrato con un numero di feature diverso.

**Messaggi di TensorFlow su CUDA, cuDNN o TensorRT**
Sono avvisi informativi: indicano solo che non viene usata la GPU. Puoi nasconderli con `TF_CPP_MIN_LOG_LEVEL=2 python test_model.py`.

## Limiti

- Il modello distingue solo tre emozioni e dipende dalla qualità e dalla varietà del dataset.
- Le prestazioni in tempo reale cambiano con illuminazione, angolo del volto e occlusioni (occhiali, mascherina).
- L'accuratezza stampata da `train_model.py` è misurata sul 20% del dataset e non garantisce lo stesso risultato su persone o condizioni diverse da quelle del training.