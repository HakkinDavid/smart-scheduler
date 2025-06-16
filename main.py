# ============================== CONTEXTO COMPLETO DEL PROYECTO ==============================
"""
Este programa llamado "Smart Scheduler" genera todas las combinaciones posibles de horarios
semanales para un conjunto de clases universitarias, eligiendo por cada clase una configuración
válida (par de huecos semanales) sin colisiones. Cada clase tiene un conjunto predefinido de 
configuraciones-clase que indican en qué días y huecos puede impartirse.

Cada hueco está definido por un día (Rango personalizable) y una posición dentro del día (A, B, C, etc.),
formando una cuadrícula.

La aplicación:
- Evalúa cada configuración global según un sistema de puntuación por "comodidad" del horario.
- Permite al usuario definir clases, configuraciones y ver todas las soluciones posibles.
- Usa una interfaz gráfica basada en `tkinter` y visualización con `matplotlib`.
"""
# ============================================================================================

import itertools
from typing import List, Tuple, Dict, NamedTuple, Any, Optional
import datetime

def generar_nombres_huecos(n):
    """Genera nombres tipo A, B, ..., Z, AA, AB, ..., AZ, ..., ZZ, etc."""
    nombres = []
    for i in range(n):
        s = ""
        x = i
        while True:
            s = chr(ord('A') + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        nombres.append(s)
    return nombres

def sumar_minutos(hora_str, minutos):
    """Suma minutos a una hora tipo '08:00' y retorna 'HH:MM'."""
    h, m = map(int, hora_str.split(":"))
    t = datetime.datetime(2000, 1, 1, h, m) + datetime.timedelta(minutes=minutos)
    return t.strftime("%H:%M")

def parse_duracion(duracion):
    """
    Convierte una cadena tipo '1h', '1h 20min', '80m', '90', '90min', '1h20', etc. a minutos.
    Lanza ValueError si el formato es inválido.
    """
    duracion = duracion.strip().lower().replace(' ', '')
    if not duracion:
        raise ValueError("Duración vacía")
    minutos = 0
    if 'h' in duracion:
        partes = duracion.split('h')
        horas = int(partes[0]) if partes[0] else 0
        minutos += horas * 60
        resto = partes[1] if len(partes) > 1 else ''
        if resto:
            if 'm' in resto:
                minutos += int(resto.replace('min', '').replace('m', ''))
            else:
                minutos += int(resto)
    elif 'm' in duracion:
        minutos += int(duracion.replace('min', '').replace('m', ''))
    else:
        minutos += int(duracion)
    if minutos <= 0:
        raise ValueError("Duración debe ser mayor a cero")
    return minutos

def calcular_etiquetas_huecos(huecos, inicio, horas, minutos):
    """Genera etiquetas de tiempo para cada hueco dado el inicio y duración."""
    etiquetas = {}
    total_min = int(horas) * 60 + int(minutos)
    minutos_acum = 0
    for hq in huecos:
        ini = sumar_minutos(inicio, minutos_acum)
        fin = sumar_minutos(inicio, minutos_acum + total_min)
        etiquetas[hq] = f"{ini}–{fin}"
        minutos_acum += total_min
    return etiquetas

# ================== NUEVAS CONFIGURACIONES DE HORARIO ==================
class ConfigHorario(NamedTuple):
    dias: List[str]  # Ej: ['Lunes', 'Martes', ...]
    huecos: List[str]  # Ej: ['A', 'B', 'C']
    etiquetas_huecos: Dict[str, str]  # Ej: {'A': '08:00–09:20', ...}
    etiquetas_dias: Dict[str, str]  # Ej: {'L': 'Lunes', ...}
    inicio: str  # '08:00'
    duracion: str  # '1h 20min'

# Configuración por defecto
DEFAULT_DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
DEFAULT_HUECOS = ['A', 'B', 'C']
DEFAULT_ETIQUETAS_HUECOS = {'A': '16:00–18:00', 'B': '18:00–20:00', 'C': '20:00–22:00'}
DEFAULT_ETIQUETAS_DIAS = {'L': 'Lunes', 'M': 'Martes', 'X': 'Miércoles', 'J': 'Jueves', 'V': 'Viernes'}

config_horario = ConfigHorario(
    dias=DEFAULT_DIAS,
    huecos=DEFAULT_HUECOS,
    etiquetas_huecos=DEFAULT_ETIQUETAS_HUECOS,
    etiquetas_dias=DEFAULT_ETIQUETAS_DIAS,
    inicio='16:00',
    duracion='2h'
)

# ================== MODELOS DE DATOS EXTENDIDOS ==================
class ConfiguracionClase(NamedTuple):
    nombre: str
    huecos: Tuple[Tuple[str, str], ...]  # Soporta N huecos
    maestro: Optional[str] = None
    nombre_curso: Optional[str] = None

class Clase(NamedTuple):
    nombre: str
    siglas: str
    maestro: str
    configuraciones: List[ConfiguracionClase]
    maestro_por_cfg: bool = False
    nombre_por_cfg: bool = False
    n_huecos: int = 2

class Solucion(NamedTuple):
    asignacion: Dict[str, ConfiguracionClase]
    puntuacion: int
    detalle: Dict[str, Any]

# Generador de combinaciones sin colisiones

def generar_globales(clases: List[Clase]) -> List[Dict[str, ConfiguracionClase]]:
    """
    Genera todas las configuraciones globales (una configuración por clase)
    sin solapamientos de huecos.
    """
    soluciones = []
    asign = {}
    ocupados = set()

    def backtrack(idx: int):
        if idx == len(clases):
            soluciones.append(asign.copy())
            return
        curso = clases[idx]
        for cfg in curso.configuraciones:
            huecos = cfg.huecos
            if all(h not in ocupados for h in huecos):
                ocupados.update(huecos)
                asign[curso.nombre] = cfg
                backtrack(idx + 1)
                ocupados.difference_update(huecos)
                asign.pop(curso.nombre)
    backtrack(0)
    return soluciones

# Evaluación de comodidad

def evaluar(solucion: Dict[str, ConfiguracionClase], config: ConfigHorario = config_horario) -> Solucion:
    """
    Evalúa una solución global y retorna puntuación y análisis detallado.
    Usa la configuración dinámica de días y huecos.
    """
    dias_cortos = list(config.etiquetas_dias.keys())
    dias: Dict[str, set] = {d: set() for d in dias_cortos}
    for cfg in solucion.values():
        for dia, hueco in cfg.huecos:
            if dia in dias:
                dias[dia].add(hueco)

    puntos = 0
    detalle = { 'dias_libres': [], 'intermedios_ok': True, 'dias_unico': [],
                'entrada_tarde': [], 'salida_temprano': [], 'carga_contigua': False,
                'unico_en_C': [] }

    # a) día libre
    for d, h in dias.items():
        if not h:
            puntos += 1
            detalle['dias_libres'].append(d)

    # b) evitar hueco intermedio (A y C sin B)
    for d, h in dias.items():
        if 'A' in h and 'C' in h and 'B' not in h:
            detalle['intermedios_ok'] = False
    if detalle['intermedios_ok']:
        puntos += 1

    # c) no hay días con una sola clase
    for d, h in dias.items():
        if len(h) == 1:
            detalle['dias_unico'].append(d)
    if not detalle['dias_unico']:
        puntos += 1

    # d) día con entrada tardía: ¬A ∧ B ∧ C
    for d, h in dias.items():
        if 'A' not in h and {'B','C'}.issubset(h):
            puntos += 1
            detalle['entrada_tarde'].append(d)

    # e) día con salida temprana: A ∧ B ∧ ¬C
    for d, h in dias.items():
        if {'A','B'}.issubset(h) and 'C' not in h:
            puntos += 1
            detalle['salida_temprano'].append(d)

    # f) días de mayor carga juntos (dos días con 2+ clases contiguos)
    dias_carga = [d for d, h in dias.items() if len(h) >= 2]
    orden = dias_cortos
    indices = sorted(orden.index(d) for d in dias_carga if d in orden)
    for i in range(len(indices)-1):
        if indices[i+1] == indices[i] + 1:
            puntos += 1
            detalle['carga_contigua'] = True
            break

    # g) día único en C
    for d, h in dias.items():
        if len(h) == 1 and 'C' in h:
            puntos += 1
            detalle['unico_en_C'].append(d)

    return Solucion(asignacion=solucion, puntuacion=puntos, detalle=detalle)

# Representación matricial (sin visualización gráfica)
def representar_matriz(solucion: Dict[str, ConfiguracionClase], config: ConfigHorario) -> List[List[str]]:
    """
    Genera una matriz NxM con etiquetas de configuración-clase o cadenas vacías.
    Filas = huecos, columnas = días, ambos según la configuración activa.
    """
    matriz = [['' for _ in config.dias] for _ in config.huecos]
    # Mapear clave corta de día a índice de columna
    etiqueta_a_clave = {v: k for k, v in config.etiquetas_dias.items()}
    dia_idx = {d: i for i, d in enumerate(config.dias)}
    hueco_idx = {h: i for i, h in enumerate(config.huecos)}
    for cfg in solucion.values():
        for dia_corto, hueco in cfg.huecos:
            dia_etiqueta = config.etiquetas_dias.get(dia_corto, dia_corto)
            if dia_etiqueta in dia_idx and hueco in hueco_idx:
                i = hueco_idx[hueco]
                j = dia_idx[dia_etiqueta]
                matriz[i][j] = cfg.nombre
    return matriz

# ================== VISUALIZACIÓN Y EXPORTACIÓN CON LEYENDA ==================
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def representar_matriz_con_leyenda(solucion: Dict[str, ConfiguracionClase], config: ConfigHorario):
    """
    Genera la matriz y la leyenda usando la configuración personalizada.
    """
    matriz = [['' for _ in config.dias] for _ in config.huecos]
    leyenda = [[{} for _ in config.dias] for _ in config.huecos]
    # Mapear clave corta de día a etiqueta personalizada
    dia_idx = {d: i for i, d in enumerate(config.dias)}
    hueco_idx = {h: i for i, h in enumerate(config.huecos)}
    for clase, cfg in solucion.items():
        for dia_corto, hueco in cfg.huecos:
            dia_etiqueta = config.etiquetas_dias.get(dia_corto, dia_corto)
            if dia_etiqueta in dia_idx and hueco in hueco_idx:
                i = hueco_idx[hueco]
                j = dia_idx[dia_etiqueta]
                matriz[i][j] = f"{cfg.nombre}"
                leyenda[i][j] = {'clase': clase, 'cfg': cfg.nombre}
    return matriz, leyenda

def mostrar_matriz_color(solucion: Solucion, config: ConfigHorario, parent_frame=None):
    """
    Renderiza la matriz de horarios en un FigureCanvasTkAgg dentro de parent_frame si se provee,
    o muestra con plt.show() si no se provee (modo legacy).
    """
    matriz, leyenda = representar_matriz_con_leyenda(solucion.asignacion, config)
    fig, ax = plt.subplots(figsize=(2+len(config.dias)*1.5, 2+len(config.huecos)))
    clases = list(solucion.asignacion.keys())
    random.seed(42)
    colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases)}
    dia_labels = config.dias
    hueco_labels = [config.etiquetas_huecos.get(h, h) for h in config.huecos]
    def get_color(i, j):
        ley = leyenda[i][j]
        if isinstance(ley, dict) and 'clase' in ley:
            return colores_asignados.get(ley['clase'], 'white')
        return 'white'
    table = ax.table(
        cellText=matriz,
        cellColours=[
            [get_color(i, j) for j in range(len(config.dias))]
            for i in range(len(config.huecos))
        ],
        rowLabels=hueco_labels, colLabels=dia_labels, loc='center', cellLoc='center'
    )
    table.scale(1, 2)
    for key, cell in table.get_celld().items():
        cell.set_fontsize(14)
    ax.axis('off')
    plt.title(f'Configuración global - Puntuación: {solucion.puntuacion}')
    # Leyenda mejorada
    legend_handles = []
    for clase, color in colores_asignados.items():
        cfg = solucion.asignacion[clase]
        nombre_curso = cfg.nombre_curso if cfg.nombre_curso else clase
        siglas = clase
        curso_str = nombre_curso if nombre_curso == siglas else f"{nombre_curso} ({siglas})"
        # Maestro: usa el de la configuración, si no, el global del curso
        maestro = cfg.maestro
        if not maestro:
            # Buscar el objeto Clase correspondiente
            clase_obj = None
            for c in solucion.asignacion:
                if c == clase:
                    # Buscar en la lista de clases global
                    # self.clases no está disponible aquí, así que buscar en asignación
                    break
            # Buscar en la lista de clases global
            # Se asume que el nombre de la clase es único
            global_maestro = None
            # Buscar en la lista de clases global (hack: usar variable global si existe)
            try:
                app = tk._default_root
                if hasattr(app, "clases"):
                    for cobj in app.clases:
                        if cobj.nombre == clase:
                            global_maestro = cobj.maestro
                            break
            except Exception:
                pass
            maestro = global_maestro if global_maestro else "(sin maestro)"
        maestro_str = maestro if maestro else "(sin maestro)"
        label = f"{curso_str}\n{maestro_str}\n{cfg.nombre}"
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label=label, markerfacecolor=color, markersize=15))
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if parent_frame is not None:
        # Limpia el frame antes de insertar el canvas
        for widget in parent_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    else:
        plt.show()

# === Funciones para guardar y cargar datos en JSON ===
import json

def guardar_entrada(clases: List['Clase'], ruta: str):
    """
    Guarda la lista de clases en un archivo JSON.
    """
    data = []
    for c in clases:
        data.append({
            'nombre': c.nombre,
            'siglas': c.siglas,
            'maestro': c.maestro,
            'maestro_por_cfg': c.maestro_por_cfg,
            'nombre_por_cfg': c.nombre_por_cfg,
            'n_huecos': c.n_huecos,
            'configuraciones': [
                {
                    'nombre': cfg.nombre,
                    'huecos': list(cfg.huecos),
                    'maestro': cfg.maestro,
                    'nombre_curso': cfg.nombre_curso
                } for cfg in c.configuraciones
            ]
        })
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def cargar_entrada(ruta: str) -> List['Clase']:
    """
    Carga la lista de clases desde un archivo JSON.
    """
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)
    clases = []
    for c in data:
        configuraciones = [
            ConfiguracionClase(
                nombre=cfg['nombre'],
                huecos=tuple(tuple(h) for h in cfg['huecos']),
                maestro=cfg.get('maestro'),
                nombre_curso=cfg.get('nombre_curso')
            )
            for cfg in c['configuraciones']
        ]
        clases.append(
            Clase(
                nombre=c['nombre'],
                siglas=c.get('siglas', c['nombre'][:3].upper()),
                maestro=c['maestro'],
                configuraciones=configuraciones,
                maestro_por_cfg=c.get('maestro_por_cfg', False),
                nombre_por_cfg=c.get('nombre_por_cfg', False),
                n_huecos=c.get('n_huecos', 2)
            )
        )
    return clases

def guardar_salida(soluciones: List[Solucion], ruta: str):
    data = []
    for sol in soluciones:
        data.append({
            'puntuacion': sol.puntuacion,
            'detalle': sol.detalle,
            'asignacion': {
                clase: {
                    'nombre': cfg.nombre,
                    'huecos': list(cfg.huecos)
                } for clase, cfg in sol.asignacion.items()
            }
        })
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ================== UI MODERNA Y SECCIONES NAVEGABLES ==================
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class SmartSchedulerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Scheduler")
        self.geometry("1100x700")
        # --- Cambia el estilo visual aquí ---
        self.configure(bg="#f5f6fa")
        self.style = ttk.Style(self)
        # Usa un tema más moderno si está disponible
        if "clam" in self.style.theme_names():
            self.style.theme_use('clam')
        # Colores personalizados para widgets ttk
        self.style.configure('.', background="#f5f6fa", foreground="#222", font=("Segoe UI", 11))
        self.style.configure('TFrame', background="#f5f6fa")
        self.style.configure('TLabel', background="#f5f6fa", foreground="#222", font=("Segoe UI", 12))
        self.style.configure('TButton', background="#4078c0", foreground="#fff", font=("Segoe UI", 11, "bold"), borderwidth=0, focusthickness=2, focuscolor="#4078c0")
        self.style.map('TButton',
            background=[('active', '#305080'), ('!active', '#4078c0')],
            foreground=[('active', '#fff'), ('!active', '#fff')]
        )
        self.style.configure('Treeview', background="#fff", fieldbackground="#fff", foreground="#222", font=("Segoe UI", 11))
        self.style.configure('Treeview.Heading', background="#4078c0", foreground="#fff", font=("Segoe UI", 11, "bold"))
        self.style.map('Treeview.Heading', background=[('active', '#305080')])
        self.style.configure('TCheckbutton', background="#f5f6fa", font=("Segoe UI", 11))
        self.style.configure('TCombobox', fieldbackground="#fff", background="#fff", font=("Segoe UI", 11))
        # Estado global
        self.clases: List[Clase] = []
        self.config_horario = config_horario
        self.soluciones: List[Solucion] = []
        self.n_huecos = len(self.config_horario.huecos)
        # Secciones
        self.frames = {}
        self.init_ui()

    def init_ui(self):
        # Navegador de secciones
        nav = ttk.Frame(self, style='TFrame')
        nav.pack(side='left', fill='y', padx=5, pady=5)
        self.main_frame = ttk.Frame(self, style='TFrame')
        self.main_frame.pack(side='right', fill='both', expand=True)
        # Botones de navegación
        secciones = [
            ("Clases", self.show_clases),
            ("Configuración Horario", self.show_config_horario),
            ("Soluciones", self.show_soluciones)
        ]
        for i, (nombre, fn) in enumerate(secciones):
            btn = ttk.Button(nav, text=nombre, command=fn, style='TButton')
            btn.pack(fill='x', pady=6, padx=4, ipady=6)
        self.show_clases()

    def clear_main(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def refrescar(self, tree):
        """
        Actualiza el Treeview de clases con los datos actuales.
        """
        tree.delete(*tree.get_children())
        for clase in self.clases:
            tree.insert(
                "", "end",
                values=(
                    clase.siglas,
                    clase.maestro,
                    ", ".join(f"{cfg.nombre}:{cfg.huecos}" for cfg in clase.configuraciones),
                    clase.maestro_por_cfg,
                    clase.nombre_por_cfg,
                    clase.n_huecos
                ),
                text=clase.nombre
            )

    # ========== SECCIÓN CLASES ==========
    def show_clases(self):
        self.clear_main()
        frm = ttk.Frame(self.main_frame, style='TFrame')
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Clases", font=("Segoe UI", 18, "bold"), style='TLabel').pack(anchor='w', pady=(0, 8))
        # Tabla de clases
        tree = ttk.Treeview(frm, columns=("Siglas", "Maestro", "Configuraciones", "Varía Maestro", "Varía Nombre", "N Huecos"), show="headings", style='Treeview')
        for col in tree["columns"]:
            tree.heading(col, text=col)
        tree.pack(fill='both', expand=True, pady=10)
        self.refrescar(tree)
        # Botones
        btns = ttk.Frame(frm, style='TFrame')
        btns.pack(pady=8)
        ttk.Button(btns, text="Agregar Clase", command=lambda: self.agregar_clase(tree), style='TButton').pack(side='left', padx=6, ipadx=8)
        ttk.Button(btns, text="Editar Clase", command=lambda: self.editar_clase(tree), style='TButton').pack(side='left', padx=6, ipadx=8)
        ttk.Button(btns, text="Eliminar Clase", command=lambda: self.eliminar_clase(tree), style='TButton').pack(side='left', padx=6, ipadx=8)
        ttk.Button(btns, text="Guardar Clases", command=lambda: self.guardar_clases(), style='TButton').pack(side='left', padx=6, ipadx=8)
        ttk.Button(btns, text="Cargar Clases", command=lambda: self.cargar_clases(tree), style='TButton').pack(side='left', padx=6, ipadx=8)
        # Elimina el botón "Generar Horarios"
        # ttk.Button(btns, text="Generar Horarios", command=self.generar_soluciones).pack(side='left', padx=5)

    def guardar_clases(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if ruta:
            guardar_entrada(self.clases, ruta)
            messagebox.showinfo("Guardado", "Clases guardadas exitosamente.")

    def cargar_clases(self, tree):
        ruta = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if ruta:
            try:
                self.clases = cargar_entrada(ruta)
                self.refrescar(tree)
                messagebox.showinfo("Carga exitosa", f"Se cargaron {len(self.clases)} clases.")
            except Exception as e:
                messagebox.showerror("Error al cargar", str(e))

    def clase_form(self, tree, clase: Optional[Clase] = None):
        # Reemplaza el contenido principal en vez de abrir ventana nueva
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        # Etiquetas para campos de texto
        tk.Label(frm, text="Nombre del curso:").grid(row=0, column=0, sticky='w')
        tk.Label(frm, text="Siglas:").grid(row=1, column=0, sticky='w')
        tk.Label(frm, text="Maestro global:").grid(row=2, column=0, sticky='w')
        tk.Label(frm, text="N° de huecos por configuración:").grid(row=3, column=0, sticky='w')
        e_nombre = tk.Entry(frm)
        e_siglas = tk.Entry(frm)
        e_maestro = tk.Entry(frm)
        e_nhuecos = ttk.Combobox(frm, values=[str(i) for i in range(1, 6)], width=3, state="readonly")
        if clase:
            e_nombre.insert(0, clase.nombre)
            e_siglas.insert(0, clase.siglas)
            e_maestro.insert(0, clase.maestro)
            e_nhuecos.set(str(clase.n_huecos))
        else:
            e_nhuecos.set("2")
        e_nombre.grid(row=0, column=1)
        e_siglas.grid(row=1, column=1)
        e_maestro.grid(row=2, column=1)
        e_nhuecos.grid(row=3, column=1)

        var_maestro_cfg = tk.BooleanVar(value=clase.maestro_por_cfg if clase else False)
        var_nombre_cfg = tk.BooleanVar(value=clase.nombre_por_cfg if clase else False)
        chk_maestro_cfg = tk.Checkbutton(frm, text="¿El maestro varía por configuración?", variable=var_maestro_cfg)
        chk_nombre_cfg = tk.Checkbutton(frm, text="¿El nombre del curso varía por configuración?", variable=var_nombre_cfg)
        chk_maestro_cfg.grid(row=4, column=0, columnspan=2, sticky='w')
        chk_nombre_cfg.grid(row=5, column=0, columnspan=2, sticky='w')

        encabezado = tk.Frame(frm)
        encabezado.grid(row=6, column=0, columnspan=2, sticky='w')
        tk.Label(encabezado, text="Nombre CFG", width=12).pack(side='left')
        encabezado_dia_hueco = []
        def actualizar_encabezado():
            for w in encabezado_dia_hueco:
                w.destroy()
            encabezado_dia_hueco.clear()
            try:
                n_huecos = int(e_nhuecos.get())
            except Exception:
                n_huecos = 2
            for i in range(n_huecos):
                l1 = tk.Label(encabezado, text=f"Día {i+1}", width=10)
                l2 = tk.Label(encabezado, text=f"Hueco {i+1}", width=8)
                l1.pack(side='left')
                l2.pack(side='left')
                encabezado_dia_hueco.extend([l1, l2])
            if var_maestro_cfg.get():
                maestro_col.pack(side='left')
            else:
                maestro_col.pack_forget()
            if var_nombre_cfg.get():
                nombre_col.pack(side='left')
            else:
                nombre_col.pack_forget()
        maestro_col = tk.Label(encabezado, text="Maestro CFG", width=14)
        nombre_col = tk.Label(encabezado, text="Nombre Curso CFG", width=18)

        campos_cfg = []
        cfg_frame = tk.Frame(frm)
        cfg_frame.grid(row=7, column=0, columnspan=2, sticky='w')
        dias_full = list(self.config_horario.dias)
        huecos_full = list(self.config_horario.huecos)
        etiquetas_huecos = self.config_horario.etiquetas_huecos

        def actualizar_opciones_dias_huecos():
            nonlocal dias_full, huecos_full, etiquetas_huecos
            dias_full = list(self.config_horario.dias)
            huecos_full = list(self.config_horario.huecos)
            etiquetas_huecos = self.config_horario.etiquetas_huecos

        def añadir_fila_cfg(c=None):
            actualizar_opciones_dias_huecos()
            fila = tk.Frame(cfg_frame)
            # Nombre CFG primero
            e_nombre_cfg = tk.Entry(fila, width=12)
            e_nombre_cfg.pack(side='left')
            dia_boxes = []
            hueco_boxes = []
            try:
                n_huecos = int(e_nhuecos.get())
            except Exception:
                n_huecos = 2
            for i in range(n_huecos):
                dia = ttk.Combobox(fila, values=dias_full, width=10, state="readonly")
                hueco = ttk.Combobox(
                    fila,
                    values=[f"{h} ({etiquetas_huecos.get(h, h)})" for h in huecos_full],
                    width=18,
                    state="readonly"
                )
                dia.pack(side='left')
                hueco.pack(side='left')
                dia_boxes.append(dia)
                hueco_boxes.append(hueco)
            e_maestro_cfg = tk.Entry(fila, width=14) if var_maestro_cfg.get() else None
            e_nombre_curso_cfg = tk.Entry(fila, width=18) if var_nombre_cfg.get() else None
            if e_maestro_cfg:
                e_maestro_cfg.pack(side='left')
            if e_nombre_curso_cfg:
                e_nombre_curso_cfg.pack(side='left')
            btn_del = tk.Button(fila, text="❌", command=lambda: (fila.destroy(), campos_cfg.remove(campo)))
            btn_del.pack(side='left')
            fila.pack(anchor='w', pady=2)
            campo = (e_nombre_cfg, dia_boxes, hueco_boxes, e_maestro_cfg, e_nombre_curso_cfg)
            campos_cfg.append(campo)
            if c:
                e_nombre_cfg.insert(0, c.nombre)
                for i, (d, h) in enumerate(c.huecos):
                    dia_boxes[i].set(self.config_horario.etiquetas_dias.get(d, d) if d in self.config_horario.etiquetas_dias else d)
                    hueco_boxes[i].set(f"{h} ({etiquetas_huecos.get(h, h)})")
                if e_maestro_cfg and c.maestro:
                    e_maestro_cfg.insert(0, c.maestro)
                if e_nombre_curso_cfg and c.nombre_curso:
                    e_nombre_curso_cfg.insert(0, c.nombre_curso)

        def on_var_cfg(*_):
            actualizar_encabezado()
            for child in cfg_frame.winfo_children():
                child.destroy()
            campos_cfg.clear()
            if clase:
                for cfg in clase.configuraciones:
                    añadir_fila_cfg(cfg)
            else:
                añadir_fila_cfg()

        var_maestro_cfg.trace_add('write', on_var_cfg)
        var_nombre_cfg.trace_add('write', on_var_cfg)
        e_nhuecos.bind('<<ComboboxSelected>>', lambda e: on_var_cfg())

        def on_config_horario_change(*_):
            actualizar_opciones_dias_huecos()
            for child in cfg_frame.winfo_children():
                child.destroy()
            campos_cfg.clear()
            if clase:
                for cfg in clase.configuraciones:
                    añadir_fila_cfg(cfg)
            else:
                añadir_fila_cfg()

        self.bind("<<ConfigHorarioChanged>>", lambda e: on_config_horario_change())

        btn_add = tk.Button(frm, text="+ Añadir Configuración", command=añadir_fila_cfg)
        btn_add.grid(row=8, column=0, columnspan=2, sticky='w')
        if clase:
            for cfg in clase.configuraciones:
                añadir_fila_cfg(cfg)
        else:
            añadir_fila_cfg()
        actualizar_encabezado()

        # --- Validación y botón Guardar ---
        btn_confirmar = tk.Button(frm, text="Confirmar")
        btn_confirmar.grid(row=9, column=0, columnspan=2)
        btn_cancelar = tk.Button(frm, text="Cancelar", command=self.show_clases)
        btn_cancelar.grid(row=10, column=0, columnspan=2)

        def validar():
            nombre = e_nombre.get().strip()
            siglas = e_siglas.get().strip() or nombre[:3].upper()
            maestro = e_maestro.get().strip()
            try:
                n_huecos = int(e_nhuecos.get())
            except Exception:
                n_huecos = 2
            # Validar campos principales
            if not nombre or not siglas or not maestro:
                return False
            # Validar al menos una configuración válida
            alguna_cfg_valida = False
            for e_nombre_cfg, dia_boxes, hueco_boxes, e_maestro_cfg, e_nombre_curso_cfg in campos_cfg:
                nombre_cfg = e_nombre_cfg.get().strip()
                huecos = []
                for dia_box, hueco_box in zip(dia_boxes, hueco_boxes):
                    dia = dia_box.get().strip()
                    hueco = hueco_box.get().strip()
                    if not dia or not hueco:
                        continue
                    huecos.append((dia, hueco))
                if len(huecos) == n_huecos and nombre_cfg:
                    alguna_cfg_valida = True
                # Si varía maestro/nombre, validar que no estén vacíos si el campo existe
                if var_maestro_cfg.get() and e_maestro_cfg and not e_maestro_cfg.get().strip():
                    return False
                if var_nombre_cfg.get() and e_nombre_curso_cfg and not e_nombre_curso_cfg.get().strip():
                    return False
            return alguna_cfg_valida

        def on_validate_change(*_):
            if validar():
                btn_confirmar.config(state="normal")
            else:
                btn_confirmar.config(state="disabled")

        # Bindings para validación en tiempo real
        e_nombre.bind("<KeyRelease>", lambda e: on_validate_change())
        e_siglas.bind("<KeyRelease>", lambda e: on_validate_change())
        e_maestro.bind("<KeyRelease>", lambda e: on_validate_change())
        e_nhuecos.bind("<<ComboboxSelected>>", lambda e: on_validate_change())
        var_maestro_cfg.trace_add('write', lambda *_: on_validate_change())
        var_nombre_cfg.trace_add('write', lambda *_: on_validate_change())
        # También para cada campo de configuración
        def bind_cfg_fields():
            for campo in campos_cfg:
                e_nombre_cfg, dia_boxes, hueco_boxes, e_maestro_cfg, e_nombre_curso_cfg = campo
                e_nombre_cfg.bind("<KeyRelease>", lambda e: on_validate_change())
                for dia_box in dia_boxes:
                    dia_box.bind("<<ComboboxSelected>>", lambda e: on_validate_change())
                for hueco_box in hueco_boxes:
                    hueco_box.bind("<<ComboboxSelected>>", lambda e: on_validate_change())
                if e_maestro_cfg:
                    e_maestro_cfg.bind("<KeyRelease>", lambda e: on_validate_change())
                if e_nombre_curso_cfg:
                    e_nombre_curso_cfg.bind("<KeyRelease>", lambda e: on_validate_change())
        # Llamar después de añadir cada fila
        old_añadir_fila_cfg = añadir_fila_cfg
        def añadir_fila_cfg_y_bind(c=None):
            old_añadir_fila_cfg(c)
            bind_cfg_fields()
            on_validate_change()
        añadir_fila_cfg = añadir_fila_cfg_y_bind
        # Reemplazar el botón para usar la nueva función
        btn_add.config(command=añadir_fila_cfg)

        # Inicializar validación
        bind_cfg_fields()
        on_validate_change()

        def confirmar():
            if not validar():
                messagebox.showerror("Datos incompletos", "Por favor completa todos los campos obligatorios y al menos una configuración válida.")
                return
            nombre = e_nombre.get().strip()
            siglas = e_siglas.get().strip() or nombre[:3].upper()
            maestro = e_maestro.get().strip()
            try:
                n_huecos = int(e_nhuecos.get())
            except Exception:
                n_huecos = 2
            configuraciones = []
            for e_nombre_cfg, dia_boxes, hueco_boxes, e_maestro_cfg, e_nombre_curso_cfg in campos_cfg:
                nombre_cfg = e_nombre_cfg.get().strip()
                huecos = []
                for dia_box, hueco_box in zip(dia_boxes, hueco_boxes):
                    dia = dia_box.get().strip()
                    dia_key = next((k for k, v in DEFAULT_ETIQUETAS_DIAS.items() if v == dia), dia)
                    hueco = hueco_box.get().strip()
                    if " " in hueco:
                        hueco = hueco.split(" ")[0]
                    if dia_key and hueco:
                        huecos.append((dia_key, hueco))
                if len(huecos) == n_huecos and nombre_cfg:
                    maestro_cfg = e_maestro_cfg.get().strip() if e_maestro_cfg else None
                    nombre_curso_cfg = e_nombre_curso_cfg.get().strip() if e_nombre_curso_cfg else None
                    configuraciones.append(ConfiguracionClase(
                        nombre=nombre_cfg,
                        huecos=tuple(huecos),
                        maestro=maestro_cfg,
                        nombre_curso=nombre_curso_cfg
                    ))
            nueva = Clase(
                nombre=nombre,
                siglas=siglas,
                maestro=maestro,
                configuraciones=configuraciones,
                maestro_por_cfg=var_maestro_cfg.get(),
                nombre_por_cfg=var_nombre_cfg.get(),
                n_huecos=n_huecos
            )
            if clase:
                i = self.clases.index(clase)
                self.clases[i] = nueva
            else:
                self.clases.append(nueva)
            self.refrescar(tree)
            self.show_clases()

        btn_confirmar.config(command=confirmar)
    def agregar_clase(self, tree):
        self.clase_form(tree, clase=None)

    def editar_clase(self, tree):
        sel = tree.selection()
        if not sel:
            return
        nombre = tree.item(sel[0], 'text')
        clase = next((c for c in self.clases if c.nombre == nombre), None)
        if not clase:
            return
        self.clase_form(tree, clase=clase)

    def eliminar_clase(self, tree):
        sel = tree.selection()
        if not sel:
            return
        nombre = tree.item(sel[0], 'text')
        self.clases = [c for c in self.clases if c.nombre != nombre]
        self.refrescar(tree)

    # ========== SECCIÓN CONFIGURACIÓN HORARIO ==========
    def show_config_horario(self):
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Configuración Visual del Horario", font=("Helvetica", 16)).pack(anchor='w')

        opciones_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        opciones_n_huecos = [str(i) for i in range(1, 73)]  # Hasta 72 huecos

        # Días activos (checkboxes)
        tk.Label(frm, text="Días activos:").pack(anchor='w')
        dias_vars = []
        dias_frame = ttk.Frame(frm)
        dias_frame.pack(anchor='w', pady=2)
        for d in opciones_dias:
            var = tk.BooleanVar(value=d in self.config_horario.dias)
            chk = tk.Checkbutton(dias_frame, text=d, variable=var)
            chk.pack(side='left')
            dias_vars.append((d, var))

        # Número de huecos
        tk.Label(frm, text="Cantidad de huecos por día:").pack(anchor='w')
        n_huecos_var = tk.StringVar(value=str(len(self.config_horario.huecos)))
        n_huecos_combo = ttk.Combobox(frm, values=opciones_n_huecos, width=4, state="readonly", textvariable=n_huecos_var)
        n_huecos_combo.pack(anchor='w', pady=2)

        # Hora de inicio del primer hueco (dos dropdowns)
        tk.Label(frm, text="Hora de inicio del primer hueco:").pack(anchor='w')
        inicio_frame = ttk.Frame(frm)
        inicio_frame.pack(anchor='w', pady=2)
        # Extraer hora y minuto de la configuración actual
        try:
            h_ini, m_ini = map(int, self.config_horario.inicio.split(":"))
        except Exception:
            h_ini, m_ini = 16, 0
        inicio_hora_var = tk.StringVar(value=str(h_ini))
        inicio_min_var = tk.StringVar(value=str(m_ini))
        inicio_hora_combo = ttk.Combobox(inicio_frame, values=[str(i) for i in range(0, 24)], width=3, state="readonly", textvariable=inicio_hora_var)
        inicio_min_combo = ttk.Combobox(inicio_frame, values=[str(i) for i in range(0, 60)], width=3, state="readonly", textvariable=inicio_min_var)
        inicio_hora_combo.pack(side='left')
        tk.Label(inicio_frame, text=":").pack(side='left')
        inicio_min_combo.pack(side='left')

        # Duración de cada hueco (dos dropdowns, opciones dinámicas)
        tk.Label(frm, text="Duración de cada hueco:").pack(anchor='w')
        duracion_frame = ttk.Frame(frm)
        duracion_frame.pack(anchor='w', pady=2)
        horas_var = tk.StringVar()
        minutos_var = tk.StringVar()
        horas_combo = ttk.Combobox(duracion_frame, width=3, state="readonly", textvariable=horas_var)
        minutos_combo = ttk.Combobox(duracion_frame, width=3, state="readonly", textvariable=minutos_var)
        tk.Label(duracion_frame, text="h").pack(side='left')
        horas_combo.pack(side='left')
        tk.Label(duracion_frame, text="m").pack(side='left')
        minutos_combo.pack(side='left')

        # Etiquetas de tiempo generadas automáticamente (con wrap)
        etiquetas_huecos_frame = ttk.Frame(frm)
        etiquetas_huecos_frame.pack(fill='x', pady=2)
        etiquetas_huecos_labels = []

        def actualizar_duracion_dropdowns(*_):
            # Calcula las opciones válidas para horas y minutos según inicio y n_huecos
            try:
                h_ini = int(inicio_hora_var.get())
                m_ini = int(inicio_min_var.get())
                inicio_min = h_ini * 60 + m_ini
            except Exception:
                h_ini, m_ini, inicio_min = 16, 0, 16*60
            try:
                n = int(n_huecos_var.get())
            except Exception:
                n = 1
            minutos_totales = 24*60 - inicio_min
            if n < 1:
                n = 1
            max_bloque = minutos_totales // n
            # Opciones para horas y minutos (0 siempre debe estar)
            horas_opciones = [str(h) for h in range(0, 25) if h*60 <= max_bloque]
            if not horas_opciones:
                horas_opciones = ['0']
            # Para minutos, depende de la hora seleccionada
            sel_h = int(horas_var.get() or 0)
            minutos_opciones = [str(m) for m in range(0, 60) if (sel_h*60 + m) <= max_bloque]
            if '0' not in minutos_opciones:
                minutos_opciones.insert(0, '0')
            # Si la opción actual no es válida, poner la primera válida
            if horas_var.get() not in horas_opciones:
                horas_var.set(horas_opciones[0])
            if minutos_var.get() not in minutos_opciones:
                minutos_var.set(minutos_opciones[0])
            horas_combo['values'] = horas_opciones
            minutos_combo['values'] = minutos_opciones
            # Por defecto, al iniciar, la duración de un bloque debe ser el máximo permitido
            if not hasattr(actualizar_duracion_dropdowns, "initialized"):
                if horas_opciones:
                    horas_var.set(horas_opciones[-1])
                if minutos_opciones:
                    minutos_var.set(minutos_opciones[0])
                actualizar_duracion_dropdowns.initialized = True

        def actualizar_configuracion(*_):
            actualizar_duracion_dropdowns()
            # Limpiar etiquetas previas
            for w in etiquetas_huecos_labels:
                w.destroy()
            etiquetas_huecos_labels.clear()
            for w in etiquetas_huecos_frame.winfo_children():
                w.destroy()
            # Días seleccionados
            dias = [d for d, var in dias_vars if var.get()]
            try:
                n = int(n_huecos_var.get())
            except Exception:
                n = 1
            huecos = generar_nombres_huecos(n)
            try:
                h_ini = int(inicio_hora_var.get())
                m_ini = int(inicio_min_var.get())
                inicio = f"{h_ini:02d}:{m_ini:02d}"
            except Exception:
                inicio = "16:00"
            try:
                h = int(horas_var.get())
                m = int(minutos_var.get())
            except Exception:
                h, m = 1, 0
            etiquetas = calcular_etiquetas_huecos(huecos, inicio, h, m)
            etiquetas_dias = {k: v for k, v in DEFAULT_ETIQUETAS_DIAS.items() if v in dias}
            self.config_horario = ConfigHorario(
                dias=dias,
                huecos=huecos,
                etiquetas_huecos=etiquetas,
                etiquetas_dias=etiquetas_dias,
                inicio=inicio,
                duracion=f"{h}h {m}min"
            )
            # Etiquetas de tiempo para cada hueco (wrap)
            ltitle = tk.Label(etiquetas_huecos_frame, text="Etiquetas de tiempo para cada hueco:")
            ltitle.grid(row=0, column=0, sticky='w')
            etiquetas_huecos_labels.append(ltitle)
            max_por_fila = 6
            for idx, hq in enumerate(huecos):
                l = tk.Label(etiquetas_huecos_frame, text=f"{hq}: {etiquetas[hq]}", relief='groove', padx=4)
                l.grid(row=1 + idx // max_por_fila, column=idx % max_por_fila, sticky='w', padx=2, pady=2)
                etiquetas_huecos_labels.append(l)

        # Bindings para actualización dinámica
        for _, var in dias_vars:
            var.trace_add('write', actualizar_configuracion)
        n_huecos_combo.bind('<<ComboboxSelected>>', actualizar_configuracion)
        inicio_hora_combo.bind('<<ComboboxSelected>>', actualizar_configuracion)
        inicio_min_combo.bind('<<ComboboxSelected>>', actualizar_configuracion)
        horas_combo.bind('<<ComboboxSelected>>', actualizar_configuracion)
        minutos_combo.bind('<<ComboboxSelected>>', actualizar_configuracion)
        # Inicialización
        actualizar_configuracion()

    # ========== SECCIÓN GENERAR HORARIOS ==========
    def show_generar(self):
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Generar Horarios", font=("Helvetica", 16)).pack(anchor='w')
        ttk.Button(frm, text="Generar", command=self.generar_soluciones).pack(pady=10)
        # Mostrar resumen de soluciones
        self.resumen_soluciones(frm)

    def generar_soluciones(self):
        self.soluciones = [evaluar(sol, self.config_horario) for sol in generar_globales(self.clases)]
        self.soluciones.sort(key=lambda x: x.puntuacion, reverse=True)
        messagebox.showinfo("Generación", f"Se generaron {len(self.soluciones)} soluciones.")

    def resumen_soluciones(self, frm):
        for widget in frm.winfo_children():
            widget.destroy()
        ttk.Label(frm, text="Resumen de Soluciones", font=("Helvetica", 14)).pack(anchor='w')
        if not self.soluciones:
            ttk.Label(frm, text="No hay soluciones generadas.").pack(anchor='w')
            return
        mejor = self.soluciones[0]
        ttk.Label(frm, text=f"Mejor Puntuación: {mejor.puntuacion}").pack(anchor='w')
        detalles = "\n".join(f"- {k}: {v}" for k, v in mejor.detalle.items())
        ttk.Label(frm, text=f"Detalles:\n{detalles}").pack(anchor='w', pady=5)
        ttk.Button(frm, text="Ver Solución", command=lambda: self.ver_solucion(mejor)).pack(pady=10)

    def ver_solucion(self, solucion):
        # Ahora muestra en la ventana principal, no en ventana emergente
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Solución Generada", font=("Helvetica", 16)).pack(anchor='w')
        canvas_frame = ttk.Frame(frm)
        canvas_frame.pack(fill='both', pady=5, expand=True)
        mostrar_matriz_color(solucion, self.config_horario, parent_frame=canvas_frame)
        ttk.Button(frm, text="Volver", command=self.show_soluciones).pack(pady=10)

    # ========== SECCIÓN SOLUCIONES ==========
    def show_soluciones(self):
        # Al entrar a esta sección, siempre genera soluciones y muestra la vista
        self.soluciones = [evaluar(sol, self.config_horario) for sol in generar_globales(self.clases)]
        self.soluciones.sort(key=lambda x: x.puntuacion, reverse=True)
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Soluciones", font=("Helvetica", 16)).pack(anchor='w')
        if not self.soluciones:
            ttk.Label(frm, text="No hay soluciones generadas.").pack(anchor='w')
            return

        # Estado de índice de solución
        self.idx_solucion = getattr(self, "idx_solucion", 0)
        total = len(self.soluciones)

        # Frame para la solución y detalles
        self.sol_frame = ttk.Frame(frm)
        self.sol_frame.pack(fill='both', expand=True, pady=10)

        # Navegación y exportación
        btns = ttk.Frame(frm)
        btns.pack(pady=10)
        ttk.Button(btns, text="Anterior", command=lambda: self.cambiar_solucion(-1)).pack(side='left', padx=5)
        ttk.Button(btns, text="Siguiente", command=lambda: self.cambiar_solucion(1)).pack(side='left', padx=5)
        ttk.Button(btns, text="Exportar PNG + TXT", command=self.exportar_solucion).pack(side='left', padx=5)
        ttk.Button(btns, text="Exportar TODO", command=self.exportar_todo_en_lote).pack(side='left', padx=5)

        self.mostrar_solucion(self.sol_frame, self.soluciones[self.idx_solucion])

    def mostrar_solucion(self, frm, solucion):
        # Limpia el frame pero no los botones de navegación
        for widget in frm.winfo_children():
            widget.destroy()
        # Info de índice
        idx = getattr(self, "idx_solucion", 0)
        total = len(self.soluciones)
        ttk.Label(frm, text=f"Solución {idx+1} / {total} - Puntuación: {solucion.puntuacion}", font=("Helvetica", 14)).pack(anchor='w')
        # Matriz visual en canvas embebido
        canvas_frame = ttk.Frame(frm)
        canvas_frame.pack(fill='both', pady=5, expand=True)
        mostrar_matriz_color(solucion, self.config_horario, parent_frame=canvas_frame)
        # Detalles de comodidad
        detalles = solucion.detalle
        detalles_str = ""
        for k, v in detalles.items():
            if isinstance(v, list):
                detalles_str += f"- {k}: {', '.join(str(x) for x in v) if v else '✔️'}\n"
            else:
                detalles_str += f"- {k}: {'✔️' if v or v is True else '❌'}\n"
        ttk.Label(frm, text="Análisis de comodidad:", font=("Helvetica", 12, "bold")).pack(anchor='w', pady=(10,0))
        ttk.Label(frm, text=detalles_str, justify='left').pack(anchor='w')

    def cambiar_solucion(self, direccion):
        total = len(self.soluciones)
        nueva_idx = getattr(self, "idx_solucion", 0) + direccion
        if 0 <= nueva_idx < total:
            self.idx_solucion = nueva_idx
            self.mostrar_solucion(self.sol_frame, self.soluciones[self.idx_solucion])

    def exportar_solucion(self):
        solucion = self.soluciones[self.idx_solucion]
        matriz, leyenda = representar_matriz_con_leyenda(solucion.asignacion, self.config_horario)
        fig, ax = plt.subplots(figsize=(2+len(self.config_horario.dias)*1.5, 2+len(self.config_horario.huecos)))
        clases_ = list(solucion.asignacion.keys())
        random.seed(42)
        colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases_)}
        dia_labels = self.config_horario.dias
        hueco_labels = [self.config_horario.etiquetas_huecos.get(h, h) for h in self.config_horario.huecos]
        def get_color(i, j):
            ley = leyenda[i][j]
            if isinstance(ley, dict) and 'clase' in ley:
                return colores_asignados.get(ley['clase'], 'white')
            return 'white'
        table = ax.table(
            cellText=matriz,
            cellColours=[
                [get_color(i, j) for j in range(len(self.config_horario.dias))]
                for i in range(len(self.config_horario.huecos))
            ],
            rowLabels=hueco_labels, colLabels=dia_labels,
            loc='center', cellLoc='center'
        )
        table.scale(1, 2)
        for key, cell in table.get_celld().items():
            cell.set_fontsize(18)
        ax.axis('off')
        plt.title(f'Solución {self.idx_solucion+1} - Puntuación: {solucion.puntuacion}')
        # Leyenda igual que en mostrar_matriz_color
        legend_handles = []
        for clase, color in colores_asignados.items():
            cfg = solucion.asignacion[clase]
            nombre_curso = cfg.nombre_curso if cfg.nombre_curso else clase
            siglas = clase
            curso_str = nombre_curso if nombre_curso == siglas else f"{nombre_curso} ({siglas})"
            maestro = cfg.maestro if cfg.maestro else None
            if not maestro:
                clase_obj = next((c for c in solucion.asignacion.values() if c.nombre == cfg.nombre), None)
                if clase_obj and hasattr(clase_obj, 'maestro') and clase_obj.maestro:
                    maestro = clase_obj.maestro
                else:
                    maestro = "(sin maestro)"
            maestro_str = maestro if maestro else "(sin maestro)"
            label = f"{curso_str}\n{maestro_str}\n{cfg.nombre}"
            legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label=label, markerfacecolor=color, markersize=15))
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            fig.savefig(path)
            txt_path = path.rsplit('.', 1)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write(f"Puntuación: {solucion.puntuacion}\n\n")
                for k, v in solucion.detalle.items():
                    f.write(f"{k}: {v}\n")
            messagebox.showinfo("Exportado", f"PNG y análisis TXT guardados.")

    def exportar_todo_en_lote(self):
        carpeta = filedialog.askdirectory()
        if not carpeta:
            return
        for i, sol in enumerate(self.soluciones, 1):
            matriz, leyenda = representar_matriz_con_leyenda(sol.asignacion, self.config_horario)
            fig, ax = plt.subplots(figsize=(2+len(self.config_horario.dias)*1.5, 2+len(self.config_horario.huecos)))
            clases_ = list(sol.asignacion.keys())
            random.seed(42)
            colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            colores_asignados = {clase: colores[j % len(colores)] for j, clase in enumerate(clases_)}
            dia_labels = self.config_horario.dias
            hueco_labels = [self.config_horario.etiquetas_huecos.get(h, h) for h in self.config_horario.huecos]
            def get_color(ii, jj):
                ley = leyenda[ii][jj]
                if isinstance(ley, dict) and 'clase' in ley:
                    return colores_asignados.get(ley['clase'], 'white')
                return 'white'
            table = ax.table(
                cellText=matriz,
                cellColours=[
                    [get_color(ii, jj) for jj in range(len(self.config_horario.dias))]
                    for ii in range(len(self.config_horario.huecos))
                ],
                rowLabels=hueco_labels, colLabels=dia_labels,
                loc='center', cellLoc='center'
            )
            table.scale(1, 2)
            for key, cell in table.get_celld().items():
                cell.set_fontsize(18)
            ax.axis('off')
            plt.title(f'Solución {i} - Puntuación: {sol.puntuacion}')
            legend_handles = []
            for clase, color in colores_asignados.items():
                cfg = sol.asignacion[clase]
                nombre_curso = cfg.nombre_curso if cfg.nombre_curso else clase
                siglas = clase
                curso_str = nombre_curso if nombre_curso == siglas else f"{nombre_curso} ({siglas})"
                maestro = cfg.maestro if cfg.maestro else None
                if not maestro:
                    clase_obj = next((c for c in sol.asignacion.values() if c.nombre == cfg.nombre), None)
                    if clase_obj and hasattr(clase_obj, 'maestro') and clase_obj.maestra:
                        maestro = clase_obj.maestra
                    else:
                        maestro = "(sin maestro)"
                maestro_str = maestro if maestro else "(sin maestro)"
                label = f"{curso_str}\n{maestro_str}\n{cfg.nombre}"
                legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label=label, markerfacecolor=color, markersize=15))
            plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            nombre_base = f"solucion_{i}_p{sol.puntuacion}"
            path_img = f"{carpeta}/{nombre_base}.png"
            path_txt = f"{carpeta}/{nombre_base}.txt"
            fig.savefig(path_img)
            with open(path_txt, "w") as f:
                f.write(f"Puntuación: {sol.puntuacion}\n\n")
                for k, v in sol.detalle.items():
                    f.write(f"{k}: {v}\n")
        messagebox.showinfo("Exportado", f"Se guardaron {len(self.soluciones)} soluciones en {carpeta}")

if __name__ == '__main__':
    app = SmartSchedulerApp()
    app.mainloop()
