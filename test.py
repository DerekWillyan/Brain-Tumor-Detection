import torch
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Definindo transformações para as imagens (mesmas usadas em 'train.py')
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Carregando o modelo treinado
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('brain_tumor_classifier.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Função de predição para nova imagem
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Converte para RGB se for necessário
    image = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return "Sim (tumor)" if predicted.item() == 1 else "Não (sem tumor)"

# Função para carregar e classificar a imagem
def classify_image():
    # Abrir uma caixa de diálogo para escolher o arquivo
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")]
    )
    if file_path:
        # Carregar a imagem
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Redimensiona para caber na janela
        img_tk = ImageTk.PhotoImage(img)

        # Atualiza a imagem mostrada na GUI
        panel.configure(image=img_tk)
        panel.image = img_tk

        # Classificar a imagem
        result = predict_image(file_path)
        result_label.config(text=f"Resultado: {result}")

# Criando a janela principal da GUI
root = tk.Tk()
root.title("Detecção de Tumor Cerebral")
root.geometry("400x500")

# Label para exibir a imagem
panel = tk.Label(root)
panel.pack(pady=20)

# Botão para selecionar e classificar a imagem
btn = tk.Button(root, text="Selecionar Imagem", command=classify_image)
btn.pack(pady=20)

# Label para exibir o resultado da classificação
result_label = tk.Label(root, text="Resultado: ", font=("Arial", 14))
result_label.pack(pady=20)

# Iniciar a interface
root.mainloop()
