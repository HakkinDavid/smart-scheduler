import itertools
from typing import List, Tuple, Dict, NamedTuple, Any

# Definiciones de tipos
DIA = str  # 'L','M','X','J','V'
HUECO = str  # 'A','B','C'

# Centralización de días y huecos
DIAS = [
    ('L', 'Lunes'), ('M', 'Martes'), ('X', 'Miércoles'), ('J', 'Jueves'), ('V', 'Viernes')
]
HUECOS = [
    ('A', 'Hueco 1'), ('B', 'Hueco 2'), ('C', 'Hueco 3')
]
ETIQUETAS_HORARIO = {k: f"{16+2*i:02d}:00 - {18+2*i:02d}:00" for i, (k, _) in enumerate(HUECOS)}
NOMBRES_DIAS = {k: v for k, v in DIAS}
NOMBRES_DIAS.update({v: k for k, v in DIAS})

# Diccionario para traducción de días
Coordenada = Tuple[DIA, HUECO]

class ConfiguracionClase(NamedTuple):
    nombre: str
    huecos: Tuple[Coordenada, Coordenada]
    maestro: str = ''
    curso: str = ''

class Clase(NamedTuple):
    nombre: str
    maestro: str
    siglas: str
    configuraciones: List[ConfiguracionClase]

class Solucion(NamedTuple):
    asignacion: Dict[str, ConfiguracionClase]  # curso.nombre -> configuración escogida
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
            h1, h2 = cfg.huecos
            if h1 not in ocupados and h2 not in ocupados:
                ocupados.update([h1, h2])
                asign[curso.nombre] = cfg
                backtrack(idx + 1)
                ocupados.difference_update([h1, h2])
                asign.pop(curso.nombre)
    backtrack(0)
    return soluciones

# Evaluación de comodidad

def evaluar(solucion: Dict[str, ConfiguracionClase]) -> Solucion:
    """
    Evalúa una solución global y retorna puntuación y análisis detallado.
    """
    # Mapa día -> set de huecos ocupados
    dias: Dict[DIA, set] = {d: set() for d in ['L','M','X','J','V']}
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
def representar_matriz(solucion: Dict[str, ConfiguracionClase], clases_dict: Dict[str, Clase] = None) -> List[List[str]]:
    """
    Genera una matriz 5x3 con etiquetas de siglas y configuración, o cadenas vacías.
    Filas en orden A, B, C; columnas en orden L,M,X,J,V.
    """
    matriz = [['' for _ in range(5)] for _ in range(3)]
    dia_idx = {'L':0,'M':1,'X':2,'J':3,'V':4}
    hueco_idx = {'A':0,'B':1,'C':2}
    for clase_nombre, cfg in solucion.items():
        for dia, hueco in cfg.huecos:
            j = dia_idx[dia]
            siglas = clases_dict[clase_nombre].siglas if clases_dict and clase_nombre in clases_dict else clase_nombre[:2].upper()
            etiqueta = f"{siglas} @ {cfg.nombre}"
            i = hueco_idx[hueco]
            matriz[i][j] = cfg.nombre
    return matriz

# === Visualización gráfica con matplotlib ===
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def mostrar_matriz_color(solucion: Solucion, clases_dict: Dict[str, Clase] = None):
    fig, ax = plt.subplots(figsize=(10, 3))
    matriz = representar_matriz(solucion.asignacion, clases_dict)

    # Colores únicos por clase
    clases = list(solucion.asignacion.keys())
    random.seed(42)
    colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases)}

    dia_labels = ['L', 'M', 'X', 'J', 'V']
    hueco_labels = [ETIQUETAS_HORARIO[h] for h in ['A', 'B', 'C']]

    table_data = [['' for _ in range(5)] for _ in range(3)]
    cell_colors = [['white' for _ in range(5)] for _ in range(3)]

    for clase, cfg in solucion.asignacion.items():
        color = colores_asignados[clase]
        for dia, hueco in cfg.huecos:
            i = {'A':0, 'B':1, 'C':2}[hueco]
            j = {'L':0, 'M':1, 'X':2, 'J':3, 'V':4}[dia]
            siglas = clases_dict[clase].siglas if clases_dict and clase in clases_dict else clase[:2].upper()
            etiqueta = f"{siglas} @ {cfg.nombre}"
            table_data[i][j] = etiqueta
            cell_colors[i][j] = color

    table = ax.table(cellText=table_data, cellColours=cell_colors,
                     rowLabels=hueco_labels, colLabels=dia_labels,
                     loc='center', cellLoc='center')
    table.scale(1, 2)
    ax.axis('off')
    plt.title(f'Configuración global - Puntuación: {solucion.puntuacion}')

    # Leyenda a la derecha del horario
    leyenda = []
    for clase, cfg in solucion.asignacion.items():
        base = clases_dict[clase] if clases_dict and clase in clases_dict else None
        color = colores_asignados[clase]
        siglas = base.siglas if base else clase[:2].upper()
        maestro = cfg.maestro or (base.maestro if base else '')
        curso = cfg.curso or (base.nombre if base else '')
        grupo = cfg.nombre
        leyenda.append((color, f"{siglas} ({curso}) | {maestro} | Grupo: {grupo}"))

    for i, (color, texto) in enumerate(leyenda):
        ax.text(1.02, 0.9 - 0.05 * i, texto, transform=ax.transAxes,
                fontsize=8, color='black', backgroundcolor=color, verticalalignment='top')

    plt.show()


# === Funciones para guardar y cargar datos en JSON ===
import json

def guardar_entrada(clases: List[Clase], ruta: str):
    data = []
    for c in clases:
        data.append({
            'nombre': c.nombre,
            'maestro': c.maestro,
            'siglas': getattr(c, 'siglas', c.nombre[:2].upper()),
            'configuraciones': [
                {
                    'nombre': cfg.nombre,
                    'huecos': list(cfg.huecos),
                    'maestro': getattr(cfg, 'maestro', ''),
                    'curso': getattr(cfg, 'curso', '')
                } for cfg in c.configuraciones
            ]
        })
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def cargar_entrada(ruta: str) -> List[Clase]:
    with open(ruta, 'r', encoding='utf-8') as f:
        data = json.load(f)
    clases = []
    for c in data:
        configuraciones = [
            ConfiguracionClase(
                nombre=cfg['nombre'],
                huecos=tuple(tuple(h) for h in cfg['huecos']),
                maestro=cfg.get('maestro', ''),
                curso=cfg.get('curso', '')
            )
            for cfg in c['configuraciones']
        ]
        clases.append(Clase(
            nombre=c['nombre'],
            maestro=c['maestro'],
            siglas=c.get('siglas', c['nombre'][:2].upper()),
            configuraciones=configuraciones
        ))
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


# === Interfaz gráfica básica con tkinter ===
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def lanzar_gui():
    ventana = tk.Tk()
    # --- Utilidades UI ---
    def combo_valores_dias():
        return [v for _, v in DIAS]
    def combo_valores_huecos():
        return [ETIQUETAS_HORARIO[k] for k, _ in HUECOS]
    def clave_dia(nombre):
        return NOMBRES_DIAS.get(nombre, nombre)
    def clave_hueco(etiqueta):
        for k, v in ETIQUETAS_HORARIO.items():
            if v == etiqueta: return k
        return etiqueta

    # --- Editor de clases ---
    def editar_clases():
        editor = tk.Toplevel(ventana)
        editor.title("Editar Clases")
        editor.geometry("700x500")

        tree = ttk.Treeview(editor, columns=("Nombre", "Maestro", "Siglas", "Configuraciones"), show="headings")
        for col in ("Nombre", "Maestro", "Siglas", "Configuraciones"):
            tree.heading(col, text=col)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def refrescar():
            tree.delete(*tree.get_children())
            for clase in clases:
                confs = ", ".join(f"{cfg.nombre}:{cfg.huecos}" for cfg in clase.configuraciones)
                tree.insert("", "end", values=(clase.nombre, clase.maestro, clase.siglas, confs))

        def agregar_o_editar(clase=None):
            win = tk.Toplevel(editor)
            win.title("Clase" + (" - Editar" if clase else " - Nueva"))
            # --- Encabezados generales ---
            tk.Label(win, text="Nombre:").grid(row=0, column=0)
            tk.Label(win, text="Maestro:").grid(row=1, column=0)
            tk.Label(win, text="Siglas:").grid(row=2, column=0)
            e_nombre = tk.Entry(win)
            e_maestro = tk.Entry(win)
            e_siglas = tk.Entry(win)
            e_nombre.grid(row=0, column=1)
            e_maestro.grid(row=1, column=1)
            e_siglas.grid(row=2, column=1)
            if clase:
                e_nombre.insert(0, clase.nombre)
                e_maestro.insert(0, clase.maestro)
                e_siglas.insert(0, clase.siglas)
            # --- Checkboxes para variabilidad ---
            var_maestro_varia = tk.BooleanVar(value=False)
            var_nombre_varia = tk.BooleanVar(value=False)
            def toggle_maestro():
                maestro_general.grid_remove() if var_maestro_varia.get() else maestro_general.grid()
                for campos in campos_cfg:
                    if campos[-2]: campos[-2].grid() if var_maestro_varia.get() else campos[-2].grid_remove()
            def toggle_nombre():
                nombre_general.grid_remove() if var_nombre_varia.get() else nombre_general.grid()
                for campos in campos_cfg:
                    if campos[-1]: campos[-1].grid() if var_nombre_varia.get() else campos[-1].grid_remove()
            tk.Checkbutton(win, text="Maestro varía por configuración", variable=var_maestro_varia, command=toggle_maestro).grid(row=3, column=0, columnspan=2)
            tk.Checkbutton(win, text="Nombre varía por configuración", variable=var_nombre_varia, command=toggle_nombre).grid(row=4, column=0, columnspan=2)
            maestro_general = e_maestro
            nombre_general = e_nombre

            # --- Encabezados de configuraciones ---
            cfg_frame = tk.Frame(win)
            cfg_frame.grid(row=5, column=0, columnspan=2, sticky='ew')
            encabezados = ["Nombre", "Día 1", "Hueco 1", "Día 2", "Hueco 2", "Maestro", "Curso", ""]
            for i, h in enumerate(encabezados):
                tk.Label(cfg_frame, text=h).grid(row=0, column=i, padx=2)
            campos_cfg = []
            def añadir_fila_cfg(c=None):
                fila = tk.Frame(cfg_frame)
                e_nombre_cfg = tk.Entry(fila, width=8)
                dia1 = ttk.Combobox(fila, values=combo_valores_dias(), width=10, state="readonly")
                hueco1 = ttk.Combobox(fila, values=combo_valores_huecos(), width=12, state="readonly")
                dia2 = ttk.Combobox(fila, values=combo_valores_dias(), width=10, state="readonly")
                hueco2 = ttk.Combobox(fila, values=combo_valores_huecos(), width=12, state="readonly")
                maestro_cfg = tk.Entry(fila, width=10)
                curso_cfg = tk.Entry(fila, width=10)
                # Set values if editing
                if c:
                    e_nombre_cfg.insert(0, c.nombre)
                    dia1.set(NOMBRES_DIAS.get(c.huecos[0][0], c.huecos[0][0]))
                    hueco1.set(ETIQUETAS_HORARIO.get(c.huecos[0][1], c.huecos[0][1]))
                    dia2.set(NOMBRES_DIAS.get(c.huecos[1][0], c.huecos[1][0]))
                    hueco2.set(ETIQUETAS_HORARIO.get(c.huecos[1][1], c.huecos[1][1]))
                    maestro_cfg.insert(0, getattr(c, 'maestro', ''))
                    curso_cfg.insert(0, getattr(c, 'curso', ''))
                # Packing
                widgets = [e_nombre_cfg, dia1, hueco1, dia2, hueco2, maestro_cfg, curso_cfg]
                for i, w in enumerate(widgets):
                    w.grid(row=0, column=i, padx=2)
                # Mostrar/ocultar según checkboxes
                maestro_cfg.grid_remove() if not var_maestro_varia.get() else maestro_cfg.grid()
                curso_cfg.grid_remove() if not var_nombre_varia.get() else curso_cfg.grid()
                # Botón eliminar
                def eliminar_fila():
                    fila.destroy()
                    campos_cfg.remove(entry_tuple)
                tk.Button(fila, text='❌', command=eliminar_fila, width=2).grid(row=0, column=7)
                fila.pack(anchor='w', pady=2, fill='x')
                entry_tuple = (e_nombre_cfg, dia1, hueco1, dia2, hueco2, maestro_cfg, curso_cfg)
                campos_cfg.append(entry_tuple)
            # Si editando, cargar configuraciones existentes
            if clase:
                for cfg in clase.configuraciones:
                    añadir_fila_cfg(cfg)
            else:
                añadir_fila_cfg()
            tk.Button(win, text="+ Añadir Configuración", command=añadir_fila_cfg).grid(row=6, column=0, sticky='w')
            # --- Confirmar ---
            def confirmar():
                nombre = e_nombre.get().strip()
                maestro = e_maestro.get().strip()
                siglas = e_siglas.get().strip() or nombre[:2].upper()
                configuraciones = []
                for campos in campos_cfg:
                    e_nombre_cfg, dia1, hueco1, dia2, hueco2, maestro_cfg, curso_cfg = campos
                    nombre_cfg = e_nombre_cfg.get().strip()
                    h1 = (clave_dia(dia1.get()), clave_hueco(hueco1.get()))
                    h2 = (clave_dia(dia2.get()), clave_hueco(hueco2.get()))
                    if all(h1) and all(h2) and nombre_cfg:
                        configuraciones.append(
                            ConfiguracionClase(
                                nombre=nombre_cfg,
                                huecos=(h1, h2),
                                maestro=maestro_cfg.get().strip() if var_maestro_varia.get() else maestro,
                                curso=curso_cfg.get().strip() if var_nombre_varia.get() else nombre
                            )
                        )
                if clase:
                    i = clases.index(clase)
                    clases[i] = Clase(nombre=nombre, maestro=maestro, siglas=siglas, configuraciones=configuraciones)
                else:
                    clases.append(Clase(nombre=nombre, maestro=maestro, siglas=siglas, configuraciones=configuraciones))
                refrescar()
                win.destroy()
            tk.Button(win, text="Confirmar", command=confirmar).grid(row=7, column=0, columnspan=2)
            # --- Sincronizar checkboxes ---
            toggle_maestro()
            toggle_nombre()
        # --- Botones ---
        btn_frame = ttk.Frame(editor)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Agregar Clase", command=lambda: agregar_o_editar()).pack(side="left", padx=5)
        def editar():
            sel = tree.selection()
            if not sel: return
            idx = tree.index(sel[0])
            agregar_o_editar(clases[idx])
        def eliminar():
            sel = tree.selection()
            if not sel: return
            idx = tree.index(sel[0])
            clases.pop(idx)
            refrescar()
        ttk.Button(btn_frame, text="Editar Clase", command=editar).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Eliminar Clase", command=eliminar).pack(side="left", padx=5)
        refrescar()

    # --- Configuración avanzada de días/huecos ---
    def configurar_dias_huecos():
        win = tk.Toplevel(ventana)
        win.title("Configurar Días y Huecos")
        # Días
        tk.Label(win, text="Días de la semana:").grid(row=0, column=0)
        dias_vars = [tk.BooleanVar(value=True) for _ in DIAS]
        for i, (_, nombre) in enumerate(DIAS):
            tk.Checkbutton(win, text=nombre, variable=dias_vars[i]).grid(row=0, column=i+1)
        # Huecos
        tk.Label(win, text="Huecos:").grid(row=1, column=0)
        huecos_entries = []
        for i, (k, _) in enumerate(HUECOS):
            tk.Label(win, text=f"Hueco {k}:").grid(row=1, column=i*2+1)
            e = tk.Entry(win, width=10)
            e.insert(0, ETIQUETAS_HORARIO[k])
            e.grid(row=1, column=i*2+2)
            huecos_entries.append((k, e))
        def guardar():
            # Actualizar días y huecos globales
            global DIAS, HUECOS, ETIQUETAS_HORARIO
            DIAS = [d for d, v in zip(DIAS, dias_vars) if v.get()]
            HUECOS = [(k, f"Hueco {k}") for k, _ in HUECOS]
            for k, e in huecos_entries:
                ETIQUETAS_HORARIO[k] = e.get().strip()
            messagebox.showinfo("Guardado", "Configuración actualizada.")
            win.destroy()
        tk.Button(win, text="Guardar", command=guardar).grid(row=2, column=0, columnspan=10, pady=10)

    # --- Visualización de soluciones y motivo ---
    def mostrar_motivo(sol):
        win = tk.Toplevel(ventana)
        win.title("Motivo de la Calificación")
        txt = tk.Text(win, width=60, height=15)
        txt.pack()
        txt.insert("end", f"Puntuación: {sol.puntuacion}\n\n")
        for k, v in sol.detalle.items():
            txt.insert("end", f"{k}: {v}\n")
        txt.config(state="disabled")

    # --- Generación y previsualización ---
    def generar():
        if not clases:
            messagebox.showwarning("Advertencia", "No hay clases cargadas.")
            return
        soluciones = generar_globales(clases)
        evaluadas = [evaluar(sol) for sol in soluciones]
        evaluadas.sort(key=lambda x: x.puntuacion, reverse=True)
        if not evaluadas:
            messagebox.showinfo("Resultado", "No se encontraron soluciones válidas.")
            return
        # Ventana de soluciones
        ventana_sol = tk.Toplevel(ventana)
        ventana_sol.title("Soluciones Generadas")
        idx = [0]
        def actualizar():
            mostrar_matriz_color(evaluadas[idx[0]], {c.nombre: c for c in clases})
            label.config(text=f"Solución {idx[0]+1} de {len(evaluadas)} - Puntuación: {evaluadas[idx[0]].puntuacion}")
        def anterior():
            if idx[0] > 0: idx[0] -= 1; actualizar()
        def siguiente():
            if idx[0] < len(evaluadas)-1: idx[0] += 1; actualizar()
        def ver_motivo():
            mostrar_motivo(evaluadas[idx[0]])
        label = ttk.Label(ventana_sol, text="")
        label.pack()
        btn_frame = ttk.Frame(ventana_sol)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Anterior", command=anterior).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Siguiente", command=siguiente).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Motivo Calificación", command=ver_motivo).pack(side="left", padx=5)
        actualizar()

    # --- UI principal con pestañas ---
    notebook = ttk.Notebook(ventana)
    frm_editar = ttk.Frame(notebook)
    frm_generar = ttk.Frame(notebook)
    frm_config = ttk.Frame(notebook)
    notebook.add(frm_editar, text="Editar Clases")
    notebook.add(frm_generar, text="Generar Horarios")
    notebook.add(frm_config, text="Configuración")
    notebook.pack(fill="both", expand=True)

    # --- Pestaña editar ---
    ttk.Button(frm_editar, text="Editar Clases", command=editar_clases).pack(pady=10)
    # --- Pestaña generar ---
    ttk.Button(frm_generar, text="Generar Horarios", command=generar).pack(pady=10)
    # --- Pestaña configuración ---
    ttk.Button(frm_config, text="Configurar Días y Huecos", command=configurar_dias_huecos).pack(pady=10)

    ventana.title("Smart Scheduler")
    ventana.mainloop()

if __name__ == '__main__':
    lanzar_gui()
