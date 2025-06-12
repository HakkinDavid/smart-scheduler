# ============================== CONTEXTO COMPLETO DEL PROYECTO ==============================
"""
Este programa llamado "Smart Scheduler" genera todas las combinaciones posibles de horarios
semanales para un conjunto de clases universitarias, eligiendo por cada clase una configuración
válida (par de huecos semanales) sin colisiones. Cada clase tiene un conjunto predefinido de 
configuraciones-clase que indican en qué días y huecos puede impartirse.

Cada hueco está definido por un día (Lunes a Viernes) y una posición dentro del día (A, B o C),
formando una cuadrícula de 5 columnas (días) por 3 filas (huecos).

La aplicación:
- Evalúa cada configuración global según un sistema de puntuación por "comodidad" del horario.
- Permite al usuario definir clases, configuraciones y ver todas las soluciones posibles.
- Usa una interfaz gráfica basada en `tkinter` y visualización con `matplotlib`.
"""
# ============================================================================================

import itertools
from typing import List, Tuple, Dict, NamedTuple, Any, Optional

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
DEFAULT_ETIQUETAS_HUECOS = {'A': '08:00–09:20', 'B': '09:30–10:50', 'C': '11:00–12:20'}
DEFAULT_ETIQUETAS_DIAS = {'L': 'Lunes', 'M': 'Martes', 'X': 'Miércoles', 'J': 'Jueves', 'V': 'Viernes'}

config_horario = ConfigHorario(
    dias=DEFAULT_DIAS,
    huecos=DEFAULT_HUECOS,
    etiquetas_huecos=DEFAULT_ETIQUETAS_HUECOS,
    etiquetas_dias=DEFAULT_ETIQUETAS_DIAS,
    inicio='08:00',
    duracion='1h 20min'
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
    # backtracking recursivo
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

def evaluar(solucion: Dict[str, ConfiguracionClase]) -> Solucion:
    """
    Evalúa una solución global y retorna puntuación y análisis detallado.
    """
    # Mapa día -> set de huecos ocupados
    dias: Dict[str, set] = {d: set() for d in ['L','M','X','J','V']}
    for cfg in solucion.values():
        for dia, hueco in cfg.huecos:
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
            points = 1
            puntos += 1
            detalle['salida_temprano'].append(d)

    # f) días de mayor carga juntos (dos días con 2+ clases contiguos)
    dias_carga = [d for d, h in dias.items() if len(h) >= 2]
    orden = ['L','M','X','J','V']
    indices = sorted(orden.index(d) for d in dias_carga)
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
def representar_matriz(solucion: Dict[str, ConfiguracionClase]) -> List[List[str]]:
    """
    Genera una matriz 5x3 con etiquetas de configuración-clase o cadenas vacías.
    Filas en orden A, B, C; columnas en orden L,M,X,J,V.
    """
    matriz = [['' for _ in range(5)] for _ in range(3)]
    dia_idx = {'L':0,'M':1,'X':2,'J':3,'V':4}
    hueco_idx = {'A':0,'B':1,'C':2}
    for cfg in solucion.values():
        for dia, hueco in cfg.huecos:
            i = hueco_idx[hueco]
            j = dia_idx[DEFAULT_ETIQUETAS_DIAS[dia]]
            matriz[i][j] = cfg.nombre
    return matriz

# ================== VISUALIZACIÓN Y EXPORTACIÓN CON LEYENDA ==================
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def mostrar_matriz_color(solucion: Solucion, config: ConfigHorario = config_horario):
    matriz, leyenda = representar_matriz_con_leyenda(solucion.asignacion, config)
    fig, ax = plt.subplots(figsize=(10, 3))
    clases = list(solucion.asignacion.keys())
    random.seed(42)
    colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases)}
    dia_labels = config.dias
    hueco_labels = [config.etiquetas_huecos[h] for h in config.huecos]
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
    ax.axis('off')
    plt.title(f'Configuración global - Puntuación: {solucion.puntuacion}')
    # Leyenda mejorada
    legend_handles = []
    for clase, color in colores_asignados.items():
        cfg = solucion.asignacion[clase]
        # Determinar nombre_curso y siglas
        nombre_curso = cfg.nombre_curso if cfg.nombre_curso else clase
        siglas = clase
        # Mostrar solo uno si son iguales
        if nombre_curso == siglas:
            curso_str = nombre_curso
        else:
            curso_str = f"{nombre_curso} ({siglas})"
        # Maestro: si no hay maestro por configuración, usar el global
        maestro = cfg.maestro if cfg.maestro else None
        if not maestro:
            # Buscar el objeto Clase correspondiente
            clase_obj = next((c for c in solucion.asignacion.values() if c.nombre == cfg.nombre), None)
            if clase_obj and hasattr(clase_obj, 'maestro') and clase_obj.maestro:
                maestro = clase_obj.maestro
            else:
                # fallback: buscar en la lista de clases global
                from inspect import currentframe
                frame = currentframe()
                while frame:
                    if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'clases'):
                        clases_list = frame.f_locals['self'].clases
                        clase_data = next((c for c in clases_list if c.nombre == clase), None)
                        if clase_data:
                            maestro = clase_data.maestro
                        break
                    frame = frame.f_back
        maestro_str = maestro if maestro else "(sin maestro)"
        label = f"{curso_str}\n{maestro_str}\n{cfg.nombre}"
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label=label, markerfacecolor=color, markersize=15))
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def representar_matriz_con_leyenda(solucion: Dict[str, ConfiguracionClase], config: ConfigHorario):
    matriz = [['' for _ in config.dias] for _ in config.huecos]
    leyenda = [[{} for _ in config.dias] for _ in config.huecos]
    dia_idx = {d: i for i, d in enumerate(config.dias)}
    hueco_idx = {h: i for i, h in enumerate(config.huecos)}
    for clase, cfg in solucion.items():
        siglas = clase  # Asumimos que el nombre de la clase es la sigla
        if hasattr(cfg, 'nombre_curso') and cfg.nombre_curso:
            siglas = cfg.nombre_curso
        for dia, hueco in cfg.huecos:
            i = hueco_idx[hueco]
            j = dia_idx[DEFAULT_ETIQUETAS_DIAS[dia]]
            matriz[i][j] = f"{siglas} @ {cfg.nombre}"
            leyenda[i][j] = {'clase': clase, 'cfg': cfg.nombre}
    return matriz, leyenda

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
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        # Estado global
        self.clases: List[Clase] = []
        self.config_horario = config_horario
        self.soluciones: List[Solucion] = []
        # Secciones
        self.frames = {}
        self.init_ui()

    def init_ui(self):
        # Navegador de secciones
        nav = ttk.Frame(self)
        nav.pack(side='left', fill='y', padx=5, pady=5)
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(side='right', fill='both', expand=True)
        # Botones de navegación
        secciones = [
            ("Clases", self.show_clases),
            ("Configuración Horario", self.show_config_horario),
            ("Soluciones", self.show_soluciones)
        ]
        for i, (nombre, fn) in enumerate(secciones):
            btn = ttk.Button(nav, text=nombre, command=fn)
            btn.pack(fill='x', pady=2)
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
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Clases", font=("Helvetica", 16)).pack(anchor='w')
        # Tabla de clases
        tree = ttk.Treeview(frm, columns=("Siglas", "Maestro", "Configuraciones", "Varía Maestro", "Varía Nombre", "N Huecos"), show="headings")
        for col in tree["columns"]:
            tree.heading(col, text=col)
        tree.pack(fill='both', expand=True, pady=10)
        self.refrescar(tree)
        # Botones
        btns = ttk.Frame(frm)
        btns.pack()
        ttk.Button(btns, text="Agregar Clase", command=lambda: self.agregar_clase(tree)).pack(side='left', padx=5)
        ttk.Button(btns, text="Editar Clase", command=lambda: self.editar_clase(tree)).pack(side='left', padx=5)
        ttk.Button(btns, text="Eliminar Clase", command=lambda: self.eliminar_clase(tree)).pack(side='left', padx=5)
        ttk.Button(btns, text="Guardar Clases", command=lambda: self.guardar_clases()).pack(side='left', padx=5)
        ttk.Button(btns, text="Cargar Clases", command=lambda: self.cargar_clases(tree)).pack(side='left', padx=5)
        # Botón para generar horarios
        ttk.Button(btns, text="Generar Horarios", command=self.generar_soluciones).pack(side='left', padx=5)

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

    def agregar_clase(self, tree):
        win = tk.Toplevel(self)
        win.title("Nueva Clase")
        # Etiquetas para campos de texto
        tk.Label(win, text="Nombre del curso:").grid(row=0, column=0, sticky='w')
        tk.Label(win, text="Siglas:").grid(row=1, column=0, sticky='w')
        tk.Label(win, text="Maestro global:").grid(row=2, column=0, sticky='w')
        tk.Label(win, text="N° de huecos por configuración:").grid(row=3, column=0, sticky='w')
        e_nombre = tk.Entry(win)
        e_siglas = tk.Entry(win)
        e_maestro = tk.Entry(win)
        e_nhuecos = ttk.Combobox(win, values=[str(i) for i in range(1, 6)], width=3, state="readonly")
        e_nhuecos.set("2")
        e_nombre.grid(row=0, column=1)
        e_siglas.grid(row=1, column=1)
        e_maestro.grid(row=2, column=1)
        e_nhuecos.grid(row=3, column=1)

        var_maestro_cfg = tk.BooleanVar()
        var_nombre_cfg = tk.BooleanVar()
        chk_maestro_cfg = tk.Checkbutton(win, text="¿El maestro varía por configuración?", variable=var_maestro_cfg)
        chk_nombre_cfg = tk.Checkbutton(win, text="¿El nombre del curso varía por configuración?", variable=var_nombre_cfg)
        chk_maestro_cfg.grid(row=4, column=0, columnspan=2, sticky='w')
        chk_nombre_cfg.grid(row=5, column=0, columnspan=2, sticky='w')

        # Encabezados de configuración
        encabezado = tk.Frame(win)
        encabezado.grid(row=6, column=0, columnspan=2, sticky='w')
        tk.Label(encabezado, text="Nombre CFG", width=12).pack(side='left')
        for i in range(1, 6):
            tk.Label(encabezado, text=f"Día {i}", width=10).pack(side='left')
            tk.Label(encabezado, text=f"Hueco {i}", width=8).pack(side='left')
        maestro_col = tk.Label(encabezado, text="Maestro CFG", width=14)
        nombre_col = tk.Label(encabezado, text="Nombre Curso CFG", width=18)
        maestro_col.pack(side='left')
        nombre_col.pack(side='left')
        maestro_col.pack_forget()
        nombre_col.pack_forget()

        campos_cfg = []

        cfg_frame = tk.Frame(win)
        cfg_frame.grid(row=7, column=0, columnspan=2, sticky='w')

        dias_full = list(DEFAULT_ETIQUETAS_DIAS.values())
        huecos_full = list(DEFAULT_ETIQUETAS_HUECOS.keys())

        def actualizar_encabezado():
            if var_maestro_cfg.get():
                maestro_col.pack(side='left')
            else:
                maestro_col.pack_forget()
            if var_nombre_cfg.get():
                nombre_col.pack(side='left')
            else:
                nombre_col.pack_forget()

        def añadir_fila_cfg(c=None):
            fila = tk.Frame(cfg_frame)
            e_nombre_cfg = tk.Entry(fila, width=12)
            dia_boxes = []
            hueco_boxes = []
            try:
                n_huecos = int(e_nhuecos.get())
            except Exception:
                n_huecos = 2
            for i in range(n_huecos):
                dia = ttk.Combobox(fila, values=dias_full, width=10, state="readonly")
                hueco = ttk.Combobox(fila, values=huecos_full, width=8, state="readonly")
                dia.pack(side='left')
                hueco.pack(side='left')
                dia_boxes.append(dia)
                hueco_boxes.append(hueco)
            e_maestro_cfg = tk.Entry(fila, width=14) if var_maestro_cfg.get() else None
            e_nombre_curso_cfg = tk.Entry(fila, width=18) if var_nombre_cfg.get() else None
            e_nombre_cfg.pack(side='left')
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
                    dia_boxes[i].set(DEFAULT_ETIQUETAS_DIAS.get(d, d))
                    hueco_boxes[i].set(h)
                if e_maestro_cfg and c.maestro:
                    e_maestro_cfg.insert(0, c.maestro)
                if e_nombre_curso_cfg and c.nombre_curso:
                    e_nombre_curso_cfg.insert(0, c.nombre_curso)

        def on_var_cfg(*_):
            actualizar_encabezado()
            for child in cfg_frame.winfo_children():
                child.destroy()
            campos_cfg.clear()
            añadir_fila_cfg()

        var_maestro_cfg.trace_add('write', on_var_cfg)
        var_nombre_cfg.trace_add('write', on_var_cfg)
        e_nhuecos.bind('<<ComboboxSelected>>', lambda e: on_var_cfg())

        btn_add = tk.Button(win, text="+ Añadir Configuración", command=añadir_fila_cfg)
        btn_add.grid(row=8, column=0, columnspan=2, sticky='w')
        añadir_fila_cfg()

        def confirmar():
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
            self.clases.append(Clase(
                nombre=nombre,
                siglas=siglas,
                maestro=maestro,
                configuraciones=configuraciones,
                maestro_por_cfg=var_maestro_cfg.get(),
                nombre_por_cfg=var_nombre_cfg.get(),
                n_huecos=n_huecos
            ))
            self.refrescar(tree)
            win.destroy()

        tk.Button(win, text="Confirmar", command=confirmar).grid(row=9, column=0, columnspan=2)

    def editar_clase(self, tree):
        sel = tree.selection()
        if not sel:
            return
        nombre = tree.item(sel[0], 'text')
        clase = next((c for c in self.clases if c.nombre == nombre), None)
        if not clase:
            return

        win = tk.Toplevel(self)
        win.title("Editar Clase")
        tk.Label(win, text="Nombre del curso:").grid(row=0, column=0, sticky='w')
        tk.Label(win, text="Siglas:").grid(row=1, column=0, sticky='w')
        tk.Label(win, text="Maestro global:").grid(row=2, column=0, sticky='w')
        tk.Label(win, text="N° de huecos por configuración:").grid(row=3, column=0, sticky='w')
        e_nombre = tk.Entry(win)
        e_siglas = tk.Entry(win)
        e_maestro = tk.Entry(win)
        e_nhuecos = ttk.Combobox(win, values=[str(i) for i in range(1, 6)], width=3, state="readonly")
        e_nombre.insert(0, clase.nombre)
        e_siglas.insert(0, clase.siglas)
        e_maestro.insert(0, clase.maestro)
        e_nhuecos.set(str(clase.n_huecos))
        e_nombre.grid(row=0, column=1)
        e_siglas.grid(row=1, column=1)
        e_maestro.grid(row=2, column=1)
        e_nhuecos.grid(row=3, column=1)

        var_maestro_cfg = tk.BooleanVar(value=clase.maestro_por_cfg)
        var_nombre_cfg = tk.BooleanVar(value=clase.nombre_por_cfg)
        chk_maestro_cfg = tk.Checkbutton(win, text="¿El maestro varía por configuración?", variable=var_maestro_cfg)
        chk_nombre_cfg = tk.Checkbutton(win, text="¿El nombre del curso varía por configuración?", variable=var_nombre_cfg)
        chk_maestro_cfg.grid(row=4, column=0, columnspan=2, sticky='w')
        chk_nombre_cfg.grid(row=5, column=0, columnspan=2, sticky='w')

        encabezado = tk.Frame(win)
        encabezado.grid(row=6, column=0, columnspan=2, sticky='w')
        tk.Label(encabezado, text="Nombre CFG", width=12).pack(side='left')
        for i in range(1, 6):
            tk.Label(encabezado, text=f"Día {i}", width=10).pack(side='left')
            tk.Label(encabezado, text=f"Hueco {i}", width=8).pack(side='left')
        maestro_col = tk.Label(encabezado, text="Maestro CFG", width=14)
        nombre_col = tk.Label(encabezado, text="Nombre Curso CFG", width=18)
        maestro_col.pack(side='left')
        nombre_col.pack(side='left')
        maestro_col.pack_forget()
        nombre_col.pack_forget()

        campos_cfg = []

        cfg_frame = tk.Frame(win)
        cfg_frame.grid(row=7, column=0, columnspan=2, sticky='w')

        dias_full = list(DEFAULT_ETIQUETAS_DIAS.values())
        huecos_full = list(DEFAULT_ETIQUETAS_HUECOS.keys())

        def actualizar_encabezado():
            if var_maestro_cfg.get():
                maestro_col.pack(side='left')
            else:
                maestro_col.pack_forget()
            if var_nombre_cfg.get():
                nombre_col.pack(side='left')
            else:
                nombre_col.pack_forget()

        def añadir_fila_cfg(c=None):
            fila = tk.Frame(cfg_frame)
            e_nombre_cfg = tk.Entry(fila, width=12)
            dia_boxes = []
            hueco_boxes = []
            n_huecos = int(e_nhuecos.get())
            for i in range(n_huecos):
                dia = ttk.Combobox(fila, values=dias_full, width=10, state="readonly")
                hueco = ttk.Combobox(fila, values=huecos_full, width=8, state="readonly")
                dia.pack(side='left')
                hueco.pack(side='left')
                dia_boxes.append(dia)
                hueco_boxes.append(hueco)
            e_maestro_cfg = tk.Entry(fila, width=14) if var_maestro_cfg.get() else None
            e_nombre_curso_cfg = tk.Entry(fila, width=18) if var_nombre_cfg.get() else None
            e_nombre_cfg.pack(side='left')
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
                    dia_boxes[i].set(DEFAULT_ETIQUETAS_DIAS.get(d, d))
                    hueco_boxes[i].set(h)
                if e_maestro_cfg and c.maestro:
                    e_maestro_cfg.insert(0, c.maestro)
                if e_nombre_curso_cfg and c.nombre_curso:
                    e_nombre_curso_cfg.insert(0, c.nombre_curso)

        def on_var_cfg(*_):
            actualizar_encabezado()
            for child in cfg_frame.winfo_children():
                child.destroy()
            campos_cfg.clear()
            for cfg in clase.configuraciones:
                añadir_fila_cfg(cfg)

        var_maestro_cfg.trace_add('write', on_var_cfg)
        var_nombre_cfg.trace_add('write', on_var_cfg)
        e_nhuecos.bind('<<ComboboxSelected>>', lambda e: on_var_cfg())

        btn_add = tk.Button(win, text="+ Añadir Configuración", command=añadir_fila_cfg)
        btn_add.grid(row=8, column=0, columnspan=2, sticky='w')
        for cfg in clase.configuraciones:
            añadir_fila_cfg(cfg)

        def confirmar_edicion():
            nombre = e_nombre.get().strip()
            siglas = e_siglas.get().strip() or nombre[:3].upper()
            maestro = e_maestro.get().strip()
            n_huecos = int(e_nhuecos.get())
            configuraciones = []
            for e_nombre_cfg, dia_boxes, hueco_boxes, e_maestro_cfg, e_nombre_curso_cfg in campos_cfg:
                nombre_cfg = e_nombre_cfg.get().strip()
                huecos = []
                for dia_box, hueco_box in zip(dia_boxes, hueco_boxes):
                    dia = dia_box.get().strip()
                    dia_key = next((k for k, v in DEFAULT_ETIQUETAS_DIAS.items() if v == dia), dia)
                    hueco = hueco_box.get().strip()
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
            i = self.clases.index(clase)
            self.clases[i] = Clase(
                nombre=nombre,
                siglas=siglas,
                maestro=maestro,
                configuraciones=configuraciones,
                maestro_por_cfg=var_maestro_cfg.get(),
                nombre_por_cfg=var_nombre_cfg.get(),
                n_huecos=n_huecos
            )
            self.refrescar(tree)
            win.destroy()

        tk.Button(win, text="Confirmar", command=confirmar_edicion).grid(row=9, column=0, columnspan=2)

    # ========== SECCIÓN CONFIGURACIÓN HORARIO ==========
    def show_config_horario(self):
        self.clear_main()
        frm = ttk.Frame(self.main_frame)
        frm.pack(fill='both', expand=True, padx=10, pady=10)
        ttk.Label(frm, text="Configuración Visual del Horario", font=("Helvetica", 16)).pack(anchor='w')

        # Opciones predefinidas
        opciones_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        opciones_huecos = ['A', 'B', 'C', 'D', 'E']
        opciones_horas = [
            '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30',
            '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30',
            '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00'
        ]
        opciones_duracion = ['1h', '1h 15min', '1h 20min', '1h 30min', '2h']

        # Días activos
        tk.Label(frm, text="Días activos:").pack(anchor='w')
        dias_vars = []
        dias_frame = ttk.Frame(frm)
        dias_frame.pack(anchor='w', pady=2)
        for d in opciones_dias:
            var = tk.BooleanVar(value=d in self.config_horario.dias)
            chk = tk.Checkbutton(dias_frame, text=d, variable=var)
            chk.pack(side='left')
            dias_vars.append((d, var))

        # Huecos activos
        tk.Label(frm, text="Huecos:").pack(anchor='w')
        huecos_vars = []
        huecos_frame = ttk.Frame(frm)
        huecos_frame.pack(anchor='w', pady=2)
        for h in opciones_huecos:
            var = tk.BooleanVar(value=h in self.config_horario.huecos)
            chk = tk.Checkbutton(huecos_frame, text=h, variable=var)
            chk.pack(side='left')
            huecos_vars.append((h, var))

        # Etiquetas de tiempo para cada hueco
        tk.Label(frm, text="Etiquetas de tiempo (inicio-fin) para cada hueco:").pack(anchor='w')
        etiquetas_huecos_frame = ttk.Frame(frm)
        etiquetas_huecos_frame.pack(anchor='w', pady=2)
        hueco_etiquetas = {}
        for h in opciones_huecos:
            if h in self.config_horario.huecos:
                inicio = ttk.Combobox(etiquetas_huecos_frame, values=opciones_horas, width=6, state="readonly")
                fin = ttk.Combobox(etiquetas_huecos_frame, values=opciones_horas, width=6, state="readonly")
                # Prellenar si existe
                etiqueta = self.config_horario.etiquetas_huecos.get(h, "")
                if etiqueta and "–" in etiqueta:
                    ini, fi = etiqueta.split("–")
                    inicio.set(ini.strip())
                    fin.set(fi.strip())
                else:
                    inicio.set(opciones_horas[0])
                    fin.set(opciones_horas[1])
                tk.Label(etiquetas_huecos_frame, text=f"{h}:").pack(side='left')
                inicio.pack(side='left')
                tk.Label(etiquetas_huecos_frame, text="–").pack(side='left')
                fin.pack(side='left')
                hueco_etiquetas[h] = (inicio, fin)

        # Etiquetas de días (asociación clave corta -> nombre)
        tk.Label(frm, text="Etiquetas de días:").pack(anchor='w')
        etiquetas_dias_frame = ttk.Frame(frm)
        etiquetas_dias_frame.pack(anchor='w', pady=2)
        dia_keys = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
        dia_labels = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        dia_etiquetas = {}
        for k, label in zip(dia_keys, dia_labels):
            cb = ttk.Combobox(etiquetas_dias_frame, values=opciones_dias, width=10, state="readonly")
            cb.set(self.config_horario.etiquetas_dias.get(k, label))
            tk.Label(etiquetas_dias_frame, text=f"{k}:").pack(side='left')
            cb.pack(side='left')
            dia_etiquetas[k] = cb

        # Hora de inicio del primer hueco
        tk.Label(frm, text="Hora de inicio del primer hueco:").pack(anchor='w')
        e_inicio = ttk.Combobox(frm, values=opciones_horas, width=8, state="readonly")
        e_inicio.set(self.config_horario.inicio)
        e_inicio.pack(fill='x', pady=2)

        # Duración de cada hueco
        tk.Label(frm, text="Duración de cada hueco:").pack(anchor='w')
        e_duracion = ttk.Combobox(frm, values=opciones_duracion, width=12, state="readonly")
        e_duracion.set(self.config_horario.duracion)
        e_duracion.pack(fill='x', pady=2)

        def guardar_config():
            dias = [d for d, var in dias_vars if var.get()]
            huecos = [h for h, var in huecos_vars if var.get()]
            etiquetas_huecos = {}
            for h in huecos:
                if h in hueco_etiquetas:
                    ini = hueco_etiquetas[h][0].get()
                    fin = hueco_etiquetas[h][1].get()
                    etiquetas_huecos[h] = f"{ini}–{fin}"
            etiquetas_dias = {}
            for k in dia_keys:
                val = dia_etiquetas[k].get()
                if val:
                    etiquetas_dias[k] = val
            inicio = e_inicio.get()
            duracion = e_duracion.get()
            self.config_horario = ConfigHorario(
                dias=dias,
                huecos=huecos,
                etiquetas_huecos=etiquetas_huecos,
                etiquetas_dias=etiquetas_dias,
                inicio=inicio,
                duracion=duracion
            )
            messagebox.showinfo("Guardado", "Configuración guardada exitosamente.")

        ttk.Button(frm, text="Guardar Configuración", command=guardar_config).pack(pady=10)

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
        self.soluciones = [evaluar(sol) for sol in generar_globales(self.clases)]
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
        ventana_sol = tk.Toplevel(self)
        ventana_sol.title("Solución Generada")
        mostrar_matriz_color(solucion, self.config_horario)

    # ========== SECCIÓN SOLUCIONES ==========
    def show_soluciones(self):
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
        # Matriz visual
        canvas_frame = ttk.Frame(frm)
        canvas_frame.pack(fill='x', pady=5)
        mostrar_matriz_color(solucion, self.config_horario)
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
        fig, ax = plt.subplots(figsize=(10, 3))
        matriz = representar_matriz(solucion.asignacion)
        clases_ = list(solucion.asignacion.keys())
        random.seed(42)
        colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases_)}
        dia_labels = self.config_horario.dias
        hueco_labels = [self.config_horario.etiquetas_huecos[h] for h in self.config_horario.huecos]
        table_data = [['' for _ in range(len(dia_labels))] for _ in range(len(hueco_labels))]
        cell_colors = [['white' for _ in range(len(dia_labels))] for _ in range(len(hueco_labels))]
        for clase, cfg in solucion.asignacion.items():
            color = colores_asignados[clase]
            for dia, hueco in cfg.huecos:
                i = {'A':0, 'B':1, 'C':2}[hueco]
                j = {'L':0, 'M':1, 'X':2, 'J':3, 'V':4}[dia]
                table_data[i][j] = cfg.nombre
                cell_colors[i][j] = color
        table = ax.table(cellText=table_data, cellColours=cell_colors,
                         rowLabels=hueco_labels, colLabels=dia_labels,
                         loc='center', cellLoc='center')
        table.scale(1, 2)
        ax.axis('off')
        plt.title(f'Solución {self.idx_solucion+1} - Puntuación: {solucion.puntuacion}')
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
            fig, ax = plt.subplots(figsize=(10, 3))
            matriz = representar_matriz(sol.asignacion)
            clases_ = list(sol.asignacion.keys())
            random.seed(42)
            colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            colores_asignados = {clase: colores[j % len(colores)] for j, clase in enumerate(clases_)}
            dia_labels = self.config_horario.dias
            hueco_labels = [self.config_horario.etiquetas_huecos[h] for h in self.config_horario.huecos]
            table_data = [['' for _ in range(len(dia_labels))] for _ in range(len(hueco_labels))]
            cell_colors = [['white' for _ in range(len(dia_labels))] for _ in range(len(hueco_labels))]
            for clase, cfg in sol.asignacion.items():
                color = colores_asignados[clase]
                for dia, hueco in cfg.huecos:
                    i = {'A':0, 'B':1, 'C':2}[hueco]
                    j = {'L':0, 'M':1, 'X':2, 'J':3, 'V':4}[dia]
                    table_data[i][j] = cfg.nombre
                    cell_colors[i][j] = color
            table = ax.table(cellText=table_data, cellColours=cell_colors,
                             rowLabels=hueco_labels, colLabels=dia_labels,
                             loc='center', cellLoc='center')
            table.scale(1, 2)
            ax.axis('off')
            plt.title(f'Solución {i} - Puntuación: {sol.puntuacion}')
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
