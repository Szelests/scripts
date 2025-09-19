import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# CLASSE PARA DEFINIR O PROBLEMA DE VALOR INICIAL (PVI)
# (Esta classe não precisa de alterações)
class ProblemaValorInicial:
    """
    Encapsula os dados de um Problema de Valor Inicial (PVI).
    y' = f(t, y), y(t0) = y0
    """
    def __init__(self, f, y0, t0, t_final, solucao_exata, descricao):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t_final = t_final
        self.solucao_exata = solucao_exata
        self.descricao = descricao

# CLASSE PARA IMPLEMENTAR OS MÉTODOS NUMÉRICOS
class ResolvedorNumerico:
    """
    Contém implementações de métodos numéricos para resolver PVIs.
    """
    def metodo_euler(self, pvi, h):
        t_vals = np.arange(pvi.t0, pvi.t_final + h, h)
        y_vals = np.zeros(len(t_vals))
        y_vals[0] = pvi.y0
        for i in range(len(t_vals) - 1):
            y_vals[i+1] = y_vals[i] + h * pvi.f(t_vals[i], y_vals[i])
        return t_vals, y_vals

    def metodo_euler_aprimorado(self, pvi, h):
        t_vals = np.arange(pvi.t0, pvi.t_final + h, h)
        y_vals = np.zeros(len(t_vals))
        y_vals[0] = pvi.y0
        for i in range(len(t_vals) - 1):
            k1 = pvi.f(t_vals[i], y_vals[i])
            y_preditor = y_vals[i] + h * k1
            k2 = pvi.f(t_vals[i+1], y_preditor)
            y_vals[i+1] = y_vals[i] + (h / 2.0) * (k1 + k2)
        return t_vals, y_vals

# CLASSE PARA A INTERFACE GRÁFICA (GUI)
class AppGUI:
    """
    Cria e gerencia a interface gráfica para resolver e visualizar os PVIs.
    AGORA COM VISUALIZAÇÃO EM SUBPLOTS 2x2.
    """
    def __init__(self, root, problemas):
        self.root = root
        self.problemas = problemas
        self.resolvedor = ResolvedorNumerico()

        self.root.title("Análise Comparativa de Métodos de Euler")
        self.root.geometry("1200x850") # Janela maior para acomodar os 4 gráficos

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Selecione o Problema de Valor Inicial:").pack(side=tk.LEFT, padx=(0, 10))

        self.descricoes_problemas = [p.descricao for p in self.problemas]
        self.problema_selecionado = tk.StringVar(value=self.descricoes_problemas[0])
        
        self.combo_problemas = ttk.Combobox(
            control_frame, textvariable=self.problema_selecionado, 
            values=self.descricoes_problemas, state="readonly", width=70
        )
        self.combo_problemas.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.solve_button = ttk.Button(control_frame, text="Resolver e Plotar", command=self.resolver_e_plotar)
        self.solve_button.pack(side=tk.LEFT, padx=10)

        self.figura, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=100)
        
        self.canvas = FigureCanvasTkAgg(self.figura, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.resolver_e_plotar()

    def resolver_e_plotar(self):
        descricao = self.problema_selecionado.get()
        pvi_atual = next((p for p in self.problemas if p.descricao == descricao), None)
        if not pvi_atual:
            return

        # Limpa todos os subplots antes de desenhar
        for ax in self.axes.flat:
            ax.clear()

        # Cálculos (feitos uma única vez)
        t_e1, y_e1 = self.resolvedor.metodo_euler(pvi_atual, 0.1)
        t_e2, y_e2 = self.resolvedor.metodo_euler(pvi_atual, 0.05)
        t_e3, y_e3 = self.resolvedor.metodo_euler(pvi_atual, 0.01)
        t_ie, y_ie = self.resolvedor.metodo_euler_aprimorado(pvi_atual, 0.1)
        t_exato = np.linspace(pvi_atual.t0, pvi_atual.t_final, 200)
        y_exato = pvi_atual.solucao_exata(t_exato)

        # --- Plotagem no Subplot 1 (Superior Esquerdo): Euler h=0.1 ---
        ax1 = self.axes[0, 0]
        ax1.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=2)
        ax1.plot(t_e1, y_e1, 'o--', color='blue', label='Aprox.')
        ax1.set_title("Euler (h=0.1)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("y(t)")
        ax1.legend()
        ax1.grid(True)

        # --- Plotagem no Subplot 2 (Superior Direito): Euler h=0.05 ---
        ax2 = self.axes[0, 1]
        ax2.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=2)
        ax2.plot(t_e2, y_e2, 's--', color='green', label='Aprox.')
        ax2.set_title("Euler (h=0.05)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("y(t)")
        ax2.legend()
        ax2.grid(True)

        # --- Plotagem no Subplot 3 (Inferior Esquerdo): Euler h=0.01 ---
        ax3 = self.axes[1, 0]
        ax3.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=2)
        ax3.plot(t_e3, y_e3, '^--', color='red', label='Aprox.')
        ax3.set_title("Euler (h=0.01)")
        ax3.set_xlabel("t")
        ax3.set_ylabel("y(t)")
        ax3.legend()
        ax3.grid(True)

        # --- Plotagem no Subplot 4 (Inferior Direito): Euler Aprimorado h=0.1 ---
        ax4 = self.axes[1, 1]
        ax4.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=2)
        ax4.plot(t_ie, y_ie, 'd-', color='purple', label='Aprox.')
        ax4.set_title("Euler Aprimorado (h=0.1)")
        ax4.set_xlabel("t")
        ax4.set_ylabel("y(t)")
        ax4.legend()
        ax4.grid(True)

        # Adiciona um título geral para a figura inteira
        self.figura.suptitle(f"Análise Comparativa: {pvi_atual.descricao}", fontsize=14)
        
        # Ajusta o layout para evitar sobreposição de títulos e eixos
        self.figura.tight_layout(rect=[0, 0, 1, 0.96])

        # Atualiza o canvas da interface
        self.canvas.draw()

# PONTO DE ENTRADA DO SCRIPT
if __name__ == "__main__":
    # Lista dos PVIs originais 
    lista_de_problemas = [
        ProblemaValorInicial(
            f=lambda t, y: 3 + t - y, y0=1, t0=0, t_final=0.5,
            solucao_exata=lambda t: 2 + t - np.exp(-t),
            descricao="(a) y' = 3 + t - y, y(0) = 1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 + t - y, y0=-1, t0=0, t_final=0.5,
            solucao_exata=lambda t: 2 + t - 3 * np.exp(-t),
            descricao="(b) y' = 3 + t - y, y(0) = -1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 2 * y - 1, y0=2, t0=0, t_final=0.5,
            solucao_exata=lambda t: 0.5 + 1.5 * np.exp(2*t),
            descricao="(c) y' = 2y - 1, y(0) = 2"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 2 * y - 1, y0=-2, t0=0, t_final=0.5,
            solucao_exata=lambda t: 0.5 - 2.5 * np.exp(2*t),
            descricao="(d) y' = 2y - 1, y(0) = -2"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 0.5 - t + 2*y, y0=1, t0=0, t_final=0.5,
            solucao_exata=lambda t: 0.5 * t + np.exp(2*t),
            descricao="(e) y' = 0.5 - t + 2y, y(0) = 1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 0.5 - t + 2*y, y0=-1, t0=0, t_final=0.5,
            solucao_exata=lambda t: 0.5 * t - np.exp(2*t),
            descricao="(f) y' = 0.5 - t + 2y, y(0) = -1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 * np.cos(t) - 2*y, y0=0, t0=0, t_final=0.5,
            solucao_exata=lambda t: 1.2 * np.cos(t) + 0.6 * np.sin(t) - 1.2 * np.exp(-2*t),
            descricao="(g) y' = 3cos(t) - 2y, y(0) = 0"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 * np.sin(t) - 2*y, y0=0, t0=0, t_final=0.5,
            solucao_exata=lambda t: 1.2 * np.sin(t) - 0.6 * np.cos(t) + 0.6 * np.exp(-2*t),
            descricao="(h) y' = 3sin(t) - 2y, y(0) = 0"
        )
    ]

    root = tk.Tk()
    app = AppGUI(root, lista_de_problemas)
    root.mainloop()