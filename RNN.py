import numpy as np

ao = 0.1  

class RNN:
    def __init__(self):
        self.e = None
        self.Weo = None
        self.o = None
        self.Wos = None
        self.s = None
        self.Ho = None

    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def gerar(self):
        self.e = np.random.uniform(-0.1, 0.1, 1)
        self.Weo = np.random.uniform(-0.1, 0.1, 6)
        self.o = np.random.uniform(-0.1, 0.1, 6)
        self.Wos = np.random.uniform(-0.1, 0.1, (9, 6))
        self.s = np.zeros((9))

    def calcular_perda(self, predicoes, verdade):
        epsilon = 1e-15
        predicoes = np.clip(predicoes, epsilon, 1 - epsilon)  
        perda = -np.sum(verdade * np.log(predicoes))  
        return perda

    def treinar(self, numero):
        verdade = np.zeros(9)  
        indice = int(numero) - 1  
        verdade[indice] = 1  

        
        perda = self.calcular_perda(self.Hs, verdade)
        print(f'Perda de Entropia Cruzada: {perda:.4f}')

        
        grad_Hs = self.Hs - verdade  
        grad_Wos = np.outer(grad_Hs, self.Ho)  
        grad_Ho = np.dot(self.Wos.T, grad_Hs) * self.Ho * (1 - self.Ho)  
        grad_Weo = grad_Ho * self.e  

        
        self.Weo -= ao * grad_Weo
        self.Wos -= ao * grad_Wos

    def iniciar(self):
        while True:
            numero = float(input('Escolha um número: '))
            self.e = np.array((numero))
            self.o = np.dot(self.Weo, self.e)
            self.Ho = self.sigmoide(self.o)
            self.s = np.dot(self.Wos, self.Ho)
            self.Hs = self.softmax(self.s)
            
            indice_max = np.argmax(self.Hs)
            mensagem = f"Eu escolho o {indice_max + 1}!"
            print(mensagem)
            
            if indice_max + 1 != numero:
                self.treinar(numero)

rnn = RNN()
rnn.gerar()
rnn.iniciar()
# made by SemOpraque ;)