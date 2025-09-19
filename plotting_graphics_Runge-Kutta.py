import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# CLASSE PARA DEFINIR O PROBLEMA DE VALOR INICIAL (PVI)
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
    def metodo_runge_kutta_4(self, pvi, h):
        """
        Resolve um PVI usando o método de Runge-Kutta de 4ª Ordem (RK4).
        """
        # Garante que o último ponto não ultrapasse t_final
        num_steps = int(round((pvi.t_final - pvi.t0) / h))
        t_vals = pvi.t0 + np.arange(num_steps + 1) * h
        
        y_vals = np.zeros(len(t_vals))
        y_vals[0] = pvi.y0

        for i in range(len(t_vals) - 1):
            t_i = t_vals[i]
            y_i = y_vals[i]
            
            k1 = pvi.f(t_i, y_i)
            k2 = pvi.f(t_i + 0.5 * h, y_i + 0.5 * h * k1)
            k3 = pvi.f(t_i + 0.5 * h, y_i + 0.5 * h * k2)
            k4 = pvi.f(t_i + h, y_i + h * k3)
            
            y_vals[i+1] = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        return t_vals, y_vals

# CLASSE PARA A INTERFACE GRÁFICA (GUI)
class AppGUI:
    """
    Cria e gerencia a interface gráfica para resolver e visualizar os PVIs.
    AGORA COM GRÁFICO DA SOLUÇÃO EXATA SEPARADO DOS DEMAIS.
    """
    def __init__(self, root, problemas):
        self.root = root
        self.problemas = problemas
        self.resolvedor = ResolvedorNumerico()

        self.root.title("Análise de Runge-Kutta com Solução Exata Separada")
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

        # MODIFICAÇÃO: Criar uma figura com uma grade 2x2 de subplots
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
        t_rk1, y_rk1 = self.resolvedor.metodo_runge_kutta_4(pvi_atual, 0.5)
        t_rk2, y_rk2 = self.resolvedor.metodo_runge_kutta_4(pvi_atual, 0.25)
        t_rk3, y_rk3 = self.resolvedor.metodo_runge_kutta_4(pvi_atual, 0.1)
        t_exato = np.linspace(pvi_atual.t0, pvi_atual.t_final, 400)
        y_exato = pvi_atual.solucao_exata(t_exato)
        
        # --- Configurações de plotagem comuns ---
        plot_configs = {'linewidth': 2}
        approx_configs = {'marker': 'o', 'linestyle': '--', 'markersize': 5, 'alpha': 0.8}

        # --- Plotagem no Subplot 1 (Superior Esquerdo): APENAS Solução Exata ---
        ax1 = self.axes[0, 0]
        ax1.plot(t_exato, y_exato, 'k-', label='Solução Exata', **plot_configs)
        ax1.set_title("Solução Exata")
        ax1.set_xlabel("t"); ax1.set_ylabel("y(t)"); ax1.legend(); ax1.grid(True)
        ax1.set_xlim([pvi_atual.t0, pvi_atual.t_final])


        # --- Plotagem no Subplot 2 (Superior Direito): RK4 h=0.5 com Exata como Ref ---
        ax2 = self.axes[0, 1]
        ax2.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=1.5, alpha=0.7) # Exata como referência
        ax2.plot(t_rk1, y_rk1, color='red', label='RK4 (h=0.5)', **approx_configs)
        ax2.set_title("Runge-Kutta (h=0.5)")
        ax2.set_xlabel("t"); ax2.set_ylabel("y(t)"); ax2.legend(); ax2.grid(True)
        ax2.set_xlim([pvi_atual.t0, pvi_atual.t_final])


        # --- Plotagem no Subplot 3 (Inferior Esquerdo): RK4 h=0.25 com Exata como Ref ---
        ax3 = self.axes[1, 0]
        ax3.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=1.5, alpha=0.7) # Exata como referência
        ax3.plot(t_rk2, y_rk2, color='blue', label='RK4 (h=0.25)', **approx_configs)
        ax3.set_title("Runge-Kutta (h=0.25)")
        ax3.set_xlabel("t"); ax3.set_ylabel("y(t)"); ax3.legend(); ax3.grid(True)
        ax3.set_xlim([pvi_atual.t0, pvi_atual.t_final])


        # --- Plotagem no Subplot 4 (Inferior Direito): RK4 h=0.1 com Exata como Ref ---
        ax4 = self.axes[1, 1]
        ax4.plot(t_exato, y_exato, 'k-', label='Exata', linewidth=1.5, alpha=0.7) # Exata como referência
        ax4.plot(t_rk3, y_rk3, color='green', label='RK4 (h=0.1)', **approx_configs)
        ax4.set_title("Runge-Kutta (h=0.1)")
        ax4.set_xlabel("t"); ax4.set_ylabel("y(t)"); ax4.legend(); ax4.grid(True)
        ax4.set_xlim([pvi_atual.t0, pvi_atual.t_final])


        # Adiciona um título geral para a figura inteira
        self.figura.suptitle(f"Análise de Runge-Kutta para: {pvi_atual.descricao}", fontsize=14)
        
        # Ajusta o layout para evitar sobreposição de títulos e eixos
        self.figura.tight_layout(rect=[0, 0, 1, 0.96])

        self.canvas.draw()

# PONTO DE ENTRADA DO SCRIPT
if __name__ == "__main__":
    # Lista dos PVIs do exercício 2 com intervalo [0, 1.0]
    lista_de_problemas = [
        ProblemaValorInicial(
            f=lambda t, y: 3 + t - y, y0=1, t0=0, t_final=1.0,
            solucao_exata=lambda t: 2 + t - np.exp(-t),
            descricao="(a) y' = 3 + t - y, y(0) = 1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 + t - y, y0=-1, t0=0, t_final=1.0,
            solucao_exata=lambda t: 2 + t - 3 * np.exp(-t),
            descricao="(b) y' = 3 + t - y, y(0) = -1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 2 * y - 1, y0=2, t0=0, t_final=1.0,
            solucao_exata=lambda t: 0.5 + 1.5 * np.exp(2*t),
            descricao="(c) y' = 2y - 1, y(0) = 2"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 2 * y - 1, y0=-2, t0=0, t_final=1.0,
            solucao_exata=lambda t: 0.5 - 2.5 * np.exp(2*t),
            descricao="(d) y' = 2y - 1, y(0) = -2"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 0.5 - t + 2*y, y0=1, t0=0, t_final=1.0,
            solucao_exata=lambda t: 0.5 * t + np.exp(2*t),
            descricao="(e) y' = 0.5 - t + 2y, y(0) = 1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 0.5 - t + 2*y, y0=-1, t0=0, t_final=1.0,
            solucao_exata=lambda t: 0.5 * t - np.exp(2*t),
            descricao="(f) y' = 0.5 - t + 2y, y(0) = -1"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 * np.cos(t) - 2*y, y0=0, t0=0, t_final=1.0,
            solucao_exata=lambda t: 1.2 * np.cos(t) + 0.6 * np.sin(t) - 1.2 * np.exp(-2*t),
            descricao="(g) y' = 3cos(t) - 2y, y(0) = 0"
        ),
        ProblemaValorInicial(
            f=lambda t, y: 3 * np.sin(t) - 2*y, y0=0, t0=0, t_final=1.0,
            solucao_exata=lambda t: 1.2 * np.sin(t) - 0.6 * np.cos(t) + 0.6 * np.exp(-2*t),
            descricao="(h) y' = 3sin(t) - 2y, y(0) = 0"
        )
    ]

    root = tk.Tk()
    app = AppGUI(root, lista_de_problemas)
    root.mainloop()