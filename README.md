# Brain Tumor Detection

Este é um projeto de detecção de tumores cerebrais em imagens de raio-X usando PyTorch e uma interface gráfica com `tkinter`. O modelo utiliza uma CNN pré-treinada (ResNet18) para classificar imagens em duas categorias: **com tumor** e **sem tumor**.

## Estrutura do Projeto

- **`train.py`**: Script para treinar o modelo de detecção de tumores cerebrais.
- **`test.py`**: Interface gráfica (`tkinter`) para classificar novas imagens de raio-X usando o modelo treinado.
- **`brain_tumor_classifier.pth`**: Arquivo do modelo treinado, salvo após o treinamento.

## Pré-requisitos

Certifique-se de ter os seguintes pacotes instalados:

- Python 3.7+
- PyTorch
- Torchvision
- PIL (Pillow)
- Tkinter (geralmente vem com Python)
- Scikit-learn

Para instalar os pacotes, você pode usar o seguinte comando:

```bash
pip install torch torchvision pillow scikit-learn
```

## Preparando o Dataset

Organize as imagens de raio-X em uma pasta chamada `Brain Tumor` no seguinte formato:

```
Brain Tumor/
├── No/        # Imagens de raio-X sem tumor
└── Yes/       # Imagens de raio-X com tumor
```

## Treinamento do Modelo

Para treinar o modelo, execute o script `train.py`:

```bash
python train.py
```

O modelo treinado será salvo como `brain_tumor_classifier.pth`.

## Classificação de Imagens

Para classificar novas imagens e verificar se há tumor ou não, execute o arquivo `test.py`:

```bash
python test.py
```

A interface gráfica permitirá que você selecione uma imagem, e o modelo exibirá o resultado da classificação na tela.

## Exemplo de Uso

1. Clique em **"Selecionar Imagem"** na GUI.
2. Escolha a imagem de raio-X para classificação.
3. A interface exibirá o resultado: **"Yes (tumor)"** ou **"No (no tumor)"**.

![Captura de tela de 2024-11-04 12-14-19](https://github.com/user-attachments/assets/5bcd632f-abe4-4d7e-b381-0cf6616cbbdd)


## Estrutura do Código

- **`train.py`**: Contém o código para carregar e treinar o modelo ResNet18, ajustado para detectar tumores cerebrais.
- **`test.py`**: Contém a interface `tkinter` que permite a seleção de uma imagem para classificação.
- **`brain_tumor_classifier.pth`**: O modelo treinado que `test.py` usa para classificar novas imagens.

## Contribuição

Sinta-se à vontade para contribuir com melhorias no código, na documentação, ou sugerir novos recursos. Forks e pull requests são bem-vindos!

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

Criado por [Derek Willyan](https://github.com/DerekWillyan/)
