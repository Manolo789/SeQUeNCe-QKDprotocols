import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Dados de exemplo
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
z = [100, 80, 60, 50, 30]


# Gráfico Y (lado esquerdo)
plt.plot(x, y)
plt.set_xlabel("Eixo X")          # legenda do eixo X (inferior)
plt.set_ylabel("Dado Y")          # legenda do eixo Y esquerdo

# Cria o segundo eixo Y (lado direito)
plt2 = plt.twinx()

# Gráfico Z (lado direito)
plt2.plot(x, z)
plt2.set_ylabel("Dado Z")          # legenda do eixo Y direito

# Mostra o gráfico
plt.show()
