# 1. Immagine base Python
FROM python:3.9-slim

# 2. Imposta la directory di lavoro all'interno del container
WORKDIR /app

# 3. Copia SOLO i requisiti e installali
# Questo ottimizza la cache di Docker
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia tutto il resto del repository (app.py, cartella Models_APP, cartella assets)
COPY . .

# 5. Imposta la variabile d'ambiente PORTA
# Hugging Face Spaces si aspetta che l'app giri sulla porta 7860
ENV PORT=7860
EXPOSE 7860

# 6. Comando di avvio
# Esegue gunicorn, 4 worker, bindato su tutte le interfacce alla porta $PORT
# 'app:server' dice a Gunicorn di cercare il file 'app.py' e usare la variabile 'server'
# (La variabile 'server = app.server' nel tuo script è corretta e necessaria)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:server"]