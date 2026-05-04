#!/bin/bash
set -e

# Verificar se existe uma pasta de virtual environment
if [ ! -d "venv" ]; then
    echo "A criar ambiente virtual..."
    python3 -m venv venv
fi

echo "A instalar/verificar dependências (aguarda um pouco, não canceles!)..."
venv/bin/python3 -m pip install -r requirements.txt > /dev/null 2>&1

echo "A iniciar o Simulador de Titulação Ácido-Base..."
venv/bin/python3 -m streamlit run app.py

