import itertools
from typing import List, Tuple, Dict, NamedTuple, Any

# Definiciones de tipos
DIA = str  # 'L','M','X','J','V'
HUECO = str  # 'A','B','C'
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
            i = hueco_idx[hueco]
            j = dia_idx[dia]
            siglas = clases_dict[clase_nombre].siglas if clases_dict and clase_nombre in clases_dict else clase_nombre[:2].upper()
            etiqueta = f"{siglas} @ {cfg.nombre}"
            matriz[i][j] = etiqueta
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
    hueco_labels = ['A', 'B', 'C']

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
    def cargar_archivo():
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                clases.clear()
                clases.extend(cargar_entrada(path))
                messagebox.showinfo("Carga exitosa", f"Se cargaron {len(clases)} clases.")
            except Exception as e:
                messagebox.showerror("Error al cargar", str(e))

    def guardar_archivo():
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                guardar_entrada(clases, path)
                messagebox.showinfo("Guardado", "Entrada guardada exitosamente.")
            except Exception as e:
                messagebox.showerror("Error al guardar", str(e))

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
        messagebox.showinfo("Resultado", f"Mejor puntuación: {evaluadas[0].puntuacion}")

        # Eliminar botones previos si existen
        for widget in frm.grid_slaves(row=3):
            widget.destroy()

        def ver_soluciones():
            if not evaluadas:
                return
            idx = [0]

            ventana_sol = tk.Toplevel(ventana)
            ventana_sol.title("Soluciones Generadas")

            canvas = tk.Canvas(ventana_sol)
            canvas.pack()

            def exportar_todo_en_lote():
                carpeta = filedialog.askdirectory()
                if not carpeta:
                    return
                for i, sol in enumerate(evaluadas, 1):
                    fig, ax = plt.subplots(figsize=(10, 3))
                    matriz = representar_matriz(sol.asignacion)
                    clases_ = list(sol.asignacion.keys())
                    random.seed(42)
                    colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                    colores_asignados = {clase: colores[j % len(colores)] for j, clase in enumerate(clases_)}
                    dia_labels = ['L', 'M', 'X', 'J', 'V']
                    hueco_labels = ['A', 'B', 'C']
                    table_data = [['' for _ in range(5)] for _ in range(3)]
                    cell_colors = [['white' for _ in range(5)] for _ in range(3)]
                    for clase, cfg in sol.asignacion.items():
                        color = colores_asignados[clase]
                        for dia, hueco in cfg.huecos:
                            ii = {'A':0, 'B':1, 'C':2}[hueco]
                            jj = {'L':0, 'M':1, 'X':2, 'J':3, 'V':4}[dia]
                            table_data[ii][jj] = cfg.nombre
                            cell_colors[ii][jj] = color
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
                messagebox.showinfo("Exportado", f"Se guardaron {len(evaluadas)} soluciones en {carpeta}")

            def actualizar():
                canvas.delete("all")
                mostrar_matriz_color(evaluadas[idx[0]])
                label.config(text=f"Solución {idx[0]+1} de {len(evaluadas)} - Puntuación: {evaluadas[idx[0]].puntuacion}")

            def anterior():
                if idx[0] > 0:
                    idx[0] -= 1
                    actualizar()

            def siguiente():
                if idx[0] < len(evaluadas) - 1:
                    idx[0] += 1
                    actualizar()

            def exportar_png():
                sol = evaluadas[idx[0]]
                fig, ax = plt.subplots(figsize=(10, 3))
                matriz = representar_matriz(sol.asignacion)
                clases_ = list(sol.asignacion.keys())
                random.seed(42)
                colores = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                colores_asignados = {clase: colores[i % len(colores)] for i, clase in enumerate(clases_)}
                dia_labels = ['L', 'M', 'X', 'J', 'V']
                hueco_labels = ['A', 'B', 'C']
                table_data = [['' for _ in range(5)] for _ in range(3)]
                cell_colors = [['white' for _ in range(5)] for _ in range(3)]
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
                plt.title(f'Solución {idx[0]+1} - Puntuación: {sol.puntuacion}')
                path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
                if path:
                    fig.savefig(path)
                    txt_path = path.rsplit('.', 1)[0] + ".txt"
                    with open(txt_path, "w") as f:
                        for k, v in sol.detalle.items():
                            f.write(f"{k}: {v}\n")
                    messagebox.showinfo("Exportado", f"PNG y análisis TXT guardados.")

            label = ttk.Label(ventana_sol, text="")
            label.pack()

            btn_frame = ttk.Frame(ventana_sol)
            btn_frame.pack(pady=10)

            ttk.Button(btn_frame, text="Anterior", command=anterior).pack(side="left", padx=5)
            ttk.Button(btn_frame, text="Siguiente", command=siguiente).pack(side="left", padx=5)
            ttk.Button(btn_frame, text="Exportar PNG + TXT", command=exportar_png).pack(side="left", padx=5)
            ttk.Button(btn_frame, text="Exportar TODO", command=exportar_todo_en_lote).pack(side="left", padx=5)

            actualizar()

        ttk.Button(frm, text="Ver Soluciones", command=ver_soluciones).grid(column=0, row=3, columnspan=2, pady=10)

    clases: List[Clase] = []

    ventana = tk.Tk()
    ventana.title("Generador de Horarios")

    frm = ttk.Frame(ventana, padding=10)
    frm.grid()

    ttk.Label(frm, text="Smart Scheduler", font=("Helvetica", 16)).grid(column=0, row=0, columnspan=3, pady=10)

    ttk.Button(frm, text="Cargar Clases", command=cargar_archivo).grid(column=0, row=1, padx=5, pady=5)
    ttk.Button(frm, text="Guardar Clases", command=guardar_archivo).grid(column=1, row=1, padx=5, pady=5)
    ttk.Button(frm, text="Generar Horarios", command=generar).grid(column=2, row=1, padx=5, pady=5)

    def editar_clases():
        editor = tk.Toplevel(ventana)
        editor.title("Editar Clases")
        editor.geometry("600x400")

        tree = ttk.Treeview(editor, columns=("Maestro", "Configuraciones"), show="headings")
        tree.heading("Maestro", text="Maestro")
        tree.heading("Configuraciones", text="Configuraciones")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def refrescar():
            tree.delete(*tree.get_children())
            for clase in clases:
                confs = ", ".join(f"{cfg.nombre}:{cfg.huecos}" for cfg in clase.configuraciones)
                tree.insert("", "end", values=(clase.maestro, confs), text=clase.nombre)

        def agregar():
            def confirmar():
                nombre = e_nombre.get()
                maestro = e_maestro.get()
                siglas = e_siglas.get().strip() or nombre[:2].upper()
                configuraciones = []
                for campos in campos_cfg:
                    # unpack with possible None for maestro_cfg, curso_cfg
                    e_nombre_cfg, dia1, hueco1, dia2, hueco2, maestro_cfg, curso_cfg = campos
                    nombre_cfg = e_nombre_cfg.get().strip()
                    h1 = (dia1.get().strip(), hueco1.get().strip())
                    h2 = (dia2.get().strip(), hueco2.get().strip())
                    if all(h1) and all(h2) and nombre_cfg:
                        configuraciones.append(
                            ConfiguracionClase(
                                nombre=nombre_cfg,
                                huecos=(h1, h2),
                                maestro=maestro_cfg.get().strip() if maestro_cfg else '',
                                curso=curso_cfg.get().strip() if curso_cfg else ''
                            )
                        )
                clases.append(Clase(nombre=nombre, maestro=maestro, siglas=siglas, configuraciones=configuraciones))
                refrescar()
                win.destroy()

            win = tk.Toplevel(editor)
            win.title("Nueva Clase")
            tk.Label(win, text="Nombre:").grid(row=0, column=0)
            tk.Label(win, text="Maestro:").grid(row=1, column=0)
            e_nombre = tk.Entry(win)
            e_maestro = tk.Entry(win)
            e_nombre.grid(row=0, column=1)
            e_maestro.grid(row=1, column=1)
            # Siglas
            tk.Label(win, text="Siglas:").grid(row=2, column=0)
            e_siglas = tk.Entry(win)
            e_siglas.grid(row=2, column=1)
            # Checkboxes para variabilidad
            var_maestro_varia = tk.BooleanVar()
            var_nombre_varia = tk.BooleanVar()
            tk.Checkbutton(win, text="Maestro varía por configuración", variable=var_maestro_varia).grid(row=3, column=0, columnspan=2)
            tk.Checkbutton(win, text="Nombre varía por configuración", variable=var_nombre_varia).grid(row=4, column=0, columnspan=2)

            campos_cfg = []

            cfg_frame = tk.Frame(win)
            cfg_frame.grid(row=5, column=1)

            def añadir_fila_cfg():
                fila = tk.Frame(cfg_frame)
                e_nombre_cfg = tk.Entry(fila, width=6)
                dia1 = ttk.Combobox(fila, values=['L','M','X','J','V'], width=2)
                hueco1 = ttk.Combobox(fila, values=['A','B','C'], width=2)
                dia2 = ttk.Combobox(fila, values=['L','M','X','J','V'], width=2)
                hueco2 = ttk.Combobox(fila, values=['A','B','C'], width=2)
                e_nombre_cfg.pack(side='left')
                dia1.pack(side='left')
                hueco1.pack(side='left')
                dia2.pack(side='left')
                hueco2.pack(side='left')
                # Añadir campos extra según checkbox
                campos = [e_nombre_cfg, dia1, hueco1, dia2, hueco2]
                if var_maestro_varia.get():
                    e_maestro_cfg = tk.Entry(fila, width=10)
                    e_maestro_cfg.pack(side='left')
                    campos.append(e_maestro_cfg)
                else:
                    campos.append(None)
                if var_nombre_varia.get():
                    e_curso_cfg = tk.Entry(fila, width=10)
                    e_curso_cfg.pack(side='left')
                    campos.append(e_curso_cfg)
                else:
                    campos.append(None)
                fila.pack(anchor='w', pady=2)
                campos_cfg.append(tuple(campos))

            btn_add = tk.Button(win, text="+ Añadir Configuración", command=añadir_fila_cfg)
            btn_add.grid(row=6, column=1, sticky='w')
            añadir_fila_cfg()

            tk.Button(win, text="Confirmar", command=confirmar).grid(row=7, column=0, columnspan=2)

        btn_frame = ttk.Frame(editor)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Agregar Clase", command=agregar).pack(side="left", padx=5)
        def eliminar():
            sel = tree.selection()
            if not sel:
                return
            nombre = tree.item(sel[0], 'text')
            clases[:] = [c for c in clases if c.nombre != nombre]
            refrescar()

        def editar():
            sel = tree.selection()
            if not sel:
                return
            nombre = tree.item(sel[0], 'text')
            clase = next((c for c in clases if c.nombre == nombre), None)
            if not clase:
                return

            def confirmar_edicion():
                clase_nombre = e_nombre.get()
                clase_maestro = e_maestro.get()
                nueva_cfgs = []
                for e_nombre_cfg, dia1, hueco1, dia2, hueco2 in campos_cfg:
                    nombre_cfg = e_nombre_cfg.get().strip()
                    h1 = (dia1.get().strip(), hueco1.get().strip())
                    h2 = (dia2.get().strip(), hueco2.get().strip())
                    if all(h1) and all(h2) and nombre_cfg:
                        nueva_cfgs.append(ConfiguracionClase(nombre=nombre_cfg, huecos=(h1, h2)))
                i = clases.index(clase)
                siglas = clase_nombre[:2].upper()
                clases[i] = Clase(nombre=clase_nombre, maestro=clase_maestro, siglas=siglas, configuraciones=nueva_cfgs)
                refrescar()
                win.destroy()

            win = tk.Toplevel(editor)
            win.title("Editar Clase")
            tk.Label(win, text="Nombre:").grid(row=0, column=0)
            tk.Label(win, text="Maestro:").grid(row=1, column=0)
            tk.Label(win, text="Configuraciones:\n(nombre, D1, H1, D2, H2)").grid(row=2, column=0)
            e_nombre = tk.Entry(win)
            e_maestro = tk.Entry(win)
            e_nombre.insert(0, clase.nombre)
            e_maestro.insert(0, clase.maestro)

            campos_cfg = []

            cfg_frame = tk.Frame(win)
            cfg_frame.grid(row=2, column=1)

            def añadir_fila_cfg(c=None):
                fila = tk.Frame(cfg_frame)
                e_nombre_cfg = tk.Entry(fila, width=6)
                dia1 = ttk.Combobox(fila, values=['L','M','X','J','V'], width=2)
                hueco1 = ttk.Combobox(fila, values=['A','B','C'], width=2)
                dia2 = ttk.Combobox(fila, values=['L','M','X','J','V'], width=2)
                hueco2 = ttk.Combobox(fila, values=['A','B','C'], width=2)
                if c:
                    e_nombre_cfg.insert(0, c.nombre)
                    dia1.set(c.huecos[0][0])
                    hueco1.set(c.huecos[0][1])
                    dia2.set(c.huecos[1][0])
                    hueco2.set(c.huecos[1][1])
                e_nombre_cfg.pack(side='left')
                dia1.pack(side='left')
                hueco1.pack(side='left')
                dia2.pack(side='left')
                hueco2.pack(side='left')
                fila.pack(anchor='w', pady=2)
                campos_cfg.append((e_nombre_cfg, dia1, hueco1, dia2, hueco2))

            for cfg in clase.configuraciones:
                añadir_fila_cfg(cfg)

            btn_add = tk.Button(win, text="+ Añadir Configuración", command=lambda: añadir_fila_cfg())
            btn_add.grid(row=3, column=1, sticky='w')

            e_nombre.grid(row=0, column=1)
            e_maestro.grid(row=1, column=1)
            tk.Button(win, text="Confirmar", command=confirmar_edicion).grid(row=4, column=0, columnspan=2)

        ttk.Button(btn_frame, text="Editar Clase", command=editar).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Eliminar Clase", command=eliminar).pack(side="left", padx=5)

        refrescar()

    ttk.Button(frm, text="Editar Clases", command=editar_clases).grid(column=1, row=2, columnspan=1, pady=5)

    ventana.mainloop()

if __name__ == '__main__':
    lanzar_gui()
