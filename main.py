import torch
from torch import nn
from torchvision.utils import save_image
import gradio as gr


latent_size=64


# задаем класс модели 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)            
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)            
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)    
        )

        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)

        return x


# функция генерации: на вход число картинок
# на выходе имя файла
def generate(number):
    global gen
    gen.eval()
    noise = torch.randn((number, 64, 1, 1))
    tensors = gen(noise)
    save_image(tensors, 'cats.jpg', normalize=True)
    return 'cats.jpg'


# инициализация модели: архитектура + веса    
def init_model():
    global gen 
    gen = Generator()
    gen.load_state_dict(torch.load('cats_generator.pt', map_location=torch.device('cpu')))
    return gen

# запуск gradio
def run(share=False):    
    gr.Interface(
        generate,
        inputs=[gr.inputs.Slider(label='Number of cats', minimum=8, maximum=32, step=8, default=8)],
        outputs="image",
    ).launch(share=share)

    
if __name__ == '__main__':
    init_model()
    run()