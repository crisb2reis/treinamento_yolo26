import os
import subprocess
import sys
from pathlib import Path

def instalar_bibliotecas():
    print("📦 Verificando e instalando bibliotecas...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", 
        "ultralytics", "torch", "torchvision", "huggingface_hub"
    ])

instalar_bibliotecas()

import torch
from huggingface_hub import snapshot_download
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        print(f"🚀 GPU Detectada: {torch.cuda.get_device_name(0)}")
        device_id = 0
    else:
        print("⚠️ Nenhuma GPU encontrada. Usando CPU.")
        device_id = "cpu"

    # 1. Diretório local absoluto para garantir que o YOLO encontre os caminhos
    base_dir = os.path.abspath("./SeaDronesSee_Dataset")

    # 2. Baixa o dataset do Hugging Face se não existir
    if not os.path.exists(base_dir) or not list(Path(base_dir).rglob("data.yaml")):
        print("📥 Baixando o dataset SeaDronesSee...")
        snapshot_download(
            repo_id="dronefreak/SeaDronesSee",
            repo_type="dataset",
            local_dir=base_dir
        )
    else:
        print(f"📂 Dataset já presente em: {base_dir}")

    # 3. Localiza e corrige o arquivo data.yaml automaticamente
    yaml_files = list(Path(base_dir).rglob("data.yaml"))
    if not yaml_files:
        print("❌ Erro: Arquivo data.yaml não encontrado.")
        return
    
    DATA_YAML = str(yaml_files[0])
    
    # Reescreve o data.yaml para garantir que a raiz aponte para a pasta atual correta
    print(fal := f"🔧 Atualizando caminhos no arquivo: {DATA_YAML}")
    with open(DATA_YAML, "r") as f:
        conteudo = f.read()
    
    # Substitui a raiz antiga pela pasta absoluta correta no seu PC
    # Geralmente os datasets do Hugging Face trazem a chave 'path:'
    dir_dataset_pai = str(Path(DATA_YAML).parent)
    
    with open(DATA_YAML, "w") as f:
        # Garante que o caminho base aponte para o diretório correto
        f.write(f"path: {dir_dataset_pai}\n")
        f.write(conteudo)

    checkpoint_path = "./runs_local/yolo26s_seadrones_pc/weights/last.pt"

    # 4. Retomada ou Início do Treinamento
    if os.path.exists(checkpoint_path):
        print(f"🔄 Retomando treinamento do checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
    else:
        print("🚀 Iniciando novo treinamento...")
        model = YOLO("yolo26s.pt")
        results = model.train(
            data=DATA_YAML,
            epochs=100,
            imgsz=640,
            batch=-1,
            device=device_id,
            workers=4,
            patience=20,
            project="./runs_local",
            name="yolo26s_seadrones_pc"
        )

    print("✅ Processo finalizado com sucesso!")

if __name__ == "__main__":
    main()