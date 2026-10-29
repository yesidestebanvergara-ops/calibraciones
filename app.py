import io
import os
import random
import re
import tempfile
import zipfile  # <-- Importación agregada para corregir el error
from datetime import datetime

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from fpdf import FPDF

# ==========================================
# 0. CONFIGURACIÓN INICIAL Y ESTADO (SESSION)
# ==========================================
st.set_page_config(page_title="Exalud - División de Metrología", layout="wide")

if "patrones_cfg" not in st.session_state:
    st.session_state["patrones_cfg"] = {
        "TEMPERATURA": {
            "desc": "Termómetros y Termohigrómetros",
            "equipo": "TERMOHIGROMETRO PATRON",
            "marca": "UNI-T",
            "cert": "LP8605-25",
            "fecha_calibracion": "28/04/2025",
        },
        "ACUSTICA": {
            "desc": "Audiómetros y Cabinas Sonoras",
            "equipo": "SONOMETRO PATRON",
            "marca": "UNIT",
            "cert": "NU1305-25",
            "fecha_calibracion": "02/05/2025",
        },
        "REVOLUCIONES": {
            "desc": "Centrífugas, Agitadores y Mazini",
            "equipo": "TACOMETRO DIGITAL PATRON",
            "marca": "UNI-T",
            "cert": "J18404-25",
            "fecha_calibracion": "20/02/2025",
        },
        "PRESION": {
            "desc": "Tensiómetros y Esfigmomanómetros",
            "equipo": "MANOMETRO DIGITAL PATRON",
            "marca": "UNIT",
            "cert": "SIM CUBE SC-2",
            "fecha_calibracion": "01/04/2025",
        },
        "MASA": {
            "desc": "Básculas y Balanzas",
            "equipo": "JUEGO DE PESAS M1 / F1",
            "marca": "PINZUAR",
            "cert": "M-29968-004 RO",
            "fecha_calibracion": "12/09/2025",
        },
        "SPO2": {
            "desc": "Pulsoxímetros",
            "equipo": "SIMULADOR DE PACIENTE SPO2",
            "marca": "FLUKE BIOMEDICAL",
            "cert": "OX-7712-25",
            "fecha_calibracion": "15/08/2027",
        },
        "ECG": {
            "desc": "Electrocardiógrafos",
            "equipo": "SIMULADOR DE ECG",
            "marca": "FLUKE BIOMEDICAL",
            "cert": "ECG9042-25",
            "fecha_calibracion": "10/08/2027",
        },
        "FLUJO": {
            "desc": "Espirómetros",
            "equipo": "JERINGA DE CALIBRACION 3L",
            "marca": "NDD",
            "cert": "YES10512",
            "fecha_calibracion": "02/09/2027",
        },
        "VOLUMEN": {
            "desc": "Micropipetas",
            "equipo": "BALANZA ANALITICA PATRON",
            "marca": "SARTORIUS",
            "cert": "BAL8821-25",
            "fecha_calibracion": "18/06/2027",
        },
        "OPTICA": {
            "desc": "Visiómetros y Luxómetros",
            "equipo": "LUXOMETRO PATRON",
            "marca": "EXTECH",
            "cert": "LUX5501-25",
            "fecha_calibracion": "05/11/2027",
        },
        "ALCOHOL": {
            "desc": "Alcoholímetros",
            "equipo": "SIMULADOR DE ALIENTO HUMEDO",
            "marca": "GUTH",
            "cert": "ALC1102-25",
            "fecha_calibracion": "14/07/2027",
        },
        "ELECTRICO": {
            "desc": "Multiparámetro Genérico",
            "equipo": "CALIBRADOR MULTIFUNCION",
            "marca": "FLUKE",
            "cert": "T23-635",
            "fecha_calibracion": "20/05/2024",
        },
    }


def limpiar_nombre_archivo(texto):
    """Sanitiza cadenas de texto eliminando caracteres no válidos para rutas de archivo."""
    cadena = str(texto).strip()
    cadena = re.sub(r'[\\/*?:"<>|]', "-", cadena)
    return re.sub(r"\s+", " ", cadena)


# ==========================================
# 1. MOTOR DE SIMULACIÓN Y MATRIZ TÉCNICA
# ==========================================
def simular_mediciones(datos_equipo):
    tipo = str(datos_equipo.get("equipo", "")).upper().strip()
    modelo = str(datos_equipo.get("modelo", "")).upper().strip()
    cfg = st.session_state["patrones_cfg"]

    res_dict = {
        "tipo": tipo,
        "es_multiparametro": False,
        "es_bascula": False,
        "es_ecg": False,
        "es_espirometro": False,
        "es_audiometro": False,
        "parametros": [],
        "patron": cfg["ELECTRICO"].copy(),
    }

    if "TERMOHIGRO" in tipo or "TERMOHIGO" in tipo:
        res_dict["es_multiparametro"] = True
        res_dict["patron"] = cfg["TEMPERATURA"].copy()
        datos_temp = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-0.3, 0.3), 1), 1),
                "correccion": round(-round(random.uniform(-0.3, 0.3), 1), 1),
                "incertidumbre": 0.25,
            }
            for p in [15.0, 20.0, 25.0]
        ]
        datos_hum = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-1.5, 1.5), 1), 1),
                "correccion": round(-round(random.uniform(-1.5, 1.5), 1), 1),
                "incertidumbre": 1.20,
            }
            for p in [30.0, 50.0, 70.0]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "TEMPERATURA AMBIENTAL",
                "unidad": "°C",
                "datos": datos_temp,
                "error_max": max([abs(d["correccion"]) for d in datos_temp]),
            },
            {
                "nombre": "HUMEDAD RELATIVA",
                "unidad": "%HR",
                "datos": datos_hum,
                "error_max": max([abs(d["correccion"]) for d in datos_hum]),
            },
        ]

    elif "PULSO" in tipo or "OXIMETRO" in tipo or "PULSOXIMETRO" in tipo:
        res_dict["es_multiparametro"] = True
        res_dict["patron"] = cfg["SPO2"].copy()
        datos_spo2 = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-1.0, 1.0), 1), 1),
                "correccion": round(-round(random.uniform(-1.0, 1.0), 1), 1),
                "incertidumbre": 0.50,
            }
            for p in [90.0, 95.0, 99.0]
        ]
        datos_bpm = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-1.5, 1.5), 1), 1),
                "correccion": round(-round(random.uniform(-1.5, 1.5), 1), 1),
                "incertidumbre": 0.80,
            }
            for p in [60.0, 80.0, 120.0]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "SATURACIÓN DE OXÍGENO (SpO2)",
                "unidad": "%",
                "datos": datos_spo2,
                "error_max": max([abs(d["correccion"]) for d in datos_spo2]),
            },
            {
                "nombre": "FRECUENCIA DE PULSO",
                "unidad": "BPM",
                "datos": datos_bpm,
                "error_max": max([abs(d["correccion"]) for d in datos_bpm]),
            },
        ]

    elif "BASCULA" in tipo or "BALANZA" in tipo:
        res_dict["es_bascula"] = True
        res_dict["patron"] = cfg["MASA"].copy()
        datos_lin = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(0.05, 0.20), 2), 2),
                "correccion": round(-round(random.uniform(0.05, 0.20), 2), 2),
                "incertidumbre": 0.08,
            }
            for p in [20.0, 40.0, 60.0, 80.0, 100.0]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "EXACTITUD Y LINEALIDAD",
                "unidad": "kg",
                "datos": datos_lin,
                "error_max": max([abs(d["correccion"]) for d in datos_lin]),
            }
        ]
        res_dict["excentricidad"] = {
            "carga": 50.0,
            "posiciones": [
                {"pos": "Centro (Posición 1)", "ind": 50.00, "error": 0.00},
                {"pos": "Frontal (Posición 2)", "ind": 50.05, "error": 0.05},
                {"pos": "Posterior (Posición 3)", "ind": 50.02, "error": 0.02},
                {"pos": "Izquierda (Posición 4)", "ind": 49.98, "error": -0.02},
                {"pos": "Derecha (Posición 5)", "ind": 50.03, "error": 0.03},
            ],
        }
        res_dict["repetibilidad"] = {
            "carga": 100.0,
            "mediciones": [100.05, 100.05, 100.08, 100.05, 100.06],
            "std_dev": 0.013,
        }

    elif "AUDIOMETRO" in tipo:
        res_dict["es_audiometro"] = True
        res_dict["patron"] = cfg["ACUSTICA"].copy()
        datos_freq = [
            {
                "parametro": f"{f} Hz",
                "referencia": f,
                "indicacion": round(
                    f + (f * (round(random.uniform(-0.5, 0.5), 2) / 100)), 1
                ),
                "correccion": round(-round(random.uniform(-0.5, 0.5), 2), 2),
                "incertidumbre": 0.10,
                "unidad": "Hz",
            }
            for f in [125, 250, 500, 1000, 2000, 4000, 8000]
        ]
        datos_db = [
            {
                "parametro": f"{db} dB HL",
                "referencia": db,
                "indicacion": round(db + round(random.uniform(-1.0, 1.0), 1), 1),
                "correccion": round(-round(random.uniform(-1.0, 1.0), 1), 1),
                "incertidumbre": 0.80,
                "unidad": "dB",
            }
            for db in [20, 40, 60, 80]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "VERIFICACIÓN DE FRECUENCIA ACÚSTICA",
                "unidad": "Hz",
                "datos": datos_freq,
                "error_max": max([abs(d["correccion"]) for d in datos_freq]),
            },
            {
                "nombre": "VERIFICACIÓN DE NIVEL DE PRESIÓN SONORA (INTENSIDAD)",
                "unidad": "dB",
                "datos": datos_db,
                "error_max": max([abs(d["correccion"]) for d in datos_db]),
            },
        ]

    elif "ELECTROCARDIOGRAFO" in tipo or "ECG" in tipo:
        res_dict["es_ecg"] = True
        res_dict["patron"] = cfg["ECG"].copy()
        datos = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-1.0, 1.0), 1), 1),
                "correccion": round(-round(random.uniform(-1.0, 1.0), 1), 1),
                "incertidumbre": 0.49,
            }
            for p in [30, 60, 80, 100, 120, 140]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "FRECUENCIA CARDIACA (ECG)",
                "unidad": "BPM",
                "datos": datos,
                "error_max": max([abs(d["correccion"]) for d in datos]),
            }
        ]

    elif "ESPIROMETRO" in tipo:
        res_dict["es_espirometro"] = True
        res_dict["patron"] = cfg["FLUJO"].copy()
        datos = [
            {
                "referencia": p,
                "indicacion": round(p + round(random.uniform(-0.8, 0.8), 1), 1),
                "correccion": round(-round(random.uniform(-0.8, 0.8), 1), 1),
                "incertidumbre": 0.49,
            }
            for p in [1, 2, 4, 5, 6]
        ]
        res_dict["parametros"] = [
            {
                "nombre": "FLUJO ESPIRATORIO",
                "unidad": "LPM",
                "datos": datos,
                "error_max": max([abs(d["correccion"]) for d in datos]),
            }
        ]

    else:
        puntos_ref = [1, 2, 3, 4, 5]
        unidad = "Unidades"
        rango_error = (-0.5, 0.5)
        incertidumbre_fija = 0.49

        if "CABINA" in tipo:
            puntos_ref = [30, 35, 40, 45, 50]
            unidad = "dB"
            rango_error = (1.0, 2.5)
            res_dict["patron"] = cfg["ACUSTICA"].copy()
        elif "TENSIOMETRO" in tipo:
            puntos_ref = [40, 60, 100, 120, 140]
            unidad = "mmHg"
            rango_error = (-0.8, 0.8)
            res_dict["patron"] = cfg["PRESION"].copy()
        elif "CENTRIFUGA" in tipo:
            puntos_ref = [1000, 2000, 2500, 3000, 4000]
            unidad = "rpm"
            rango_error = (1.0, 4.0)
            res_dict["patron"] = cfg["REVOLUCIONES"].copy()
        elif "AGITADOR" in tipo or "MAZINI" in tipo:
            puntos_ref = [100, 200, 250, 300, 400]
            unidad = "rpm"
            rango_error = (1.0, 3.0)
            res_dict["patron"] = cfg["REVOLUCIONES"].copy()
        elif "TERMOMETRO" in tipo or "NEVERA" in tipo:
            puntos_ref = [-10, -5, 0, 2, 8]
            unidad = "°C"
            rango_error = (-0.3, 0.3)
            res_dict["patron"] = cfg["TEMPERATURA"].copy()
        elif "VISIOMETRO" in tipo:
            puntos_ref = [20, 30, 50, 100, 200]
            unidad = "cd/m2"
            rango_error = (-1.0, 1.0)
            res_dict["patron"] = cfg["OPTICA"].copy()
        elif "ALCOHOLIMETRO" in tipo:
            puntos_ref = [0.10, 0.25, 0.40, 0.60]
            unidad = "mg/L"
            rango_error = (-0.01, 0.01)
            incertidumbre_fija = 0.01
            res_dict["patron"] = cfg["ALCOHOL"].copy()
        elif "MICROPIPETA" in tipo:
            unidad = "uL"
            res_dict["patron"] = cfg["VOLUMEN"].copy()
            if "1000" in modelo or "100" in modelo:
                puntos_ref = [100, 500, 1000]
                rango_error = (-1.8, 1.8)
                incertidumbre_fija = 0.50
            else:
                puntos_ref = [5, 25, 50]
                rango_error = (-0.2, 0.2)
                incertidumbre_fija = 0.05

        datos = []
        prec = 2 if unidad in ["mg/L", "uL"] else 1
        for p in puntos_ref:
            err = round(random.uniform(rango_error[0], rango_error[1]), prec)
            datos.append(
                {
                    "referencia": p,
                    "indicacion": round(p + err, prec),
                    "correccion": round(-err, prec),
                    "incertidumbre": incertidumbre_fija,
                }
            )
        res_dict["parametros"] = [
            {
                "nombre": f"MEDICION DE {tipo}",
                "unidad": unidad,
                "datos": datos,
                "error_max": max([abs(d["correccion"]) for d in datos]),
            }
        ]

    return res_dict


# ==========================================
# 2. GENERADOR DE CÓDIGO QR Y GRÁFICAS
# ==========================================
def generar_qr_temp(texto_qr):
    qr = qrcode.QRCode(version=1, box_size=4, border=1)
    qr.add_data(texto_qr)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#003366", back_color="white")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        return tmp.name


def crear_grafica(datos, unidad):
    referencias = [d["referencia"] for d in datos]
    errores = [d["correccion"] * -1 for d in datos]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot(
        referencias,
        errores,
        marker="o",
        linestyle="-",
        color="#003366",
        linewidth=2,
        markersize=7,
    )
    ax.set_title(
        "CURVA DE ERROR EN LA INDICACION",
        fontsize=9,
        fontweight="bold",
        color="#003366",
        pad=8,
    )
    ax.set_xlabel(f"Valor de Referencia ({unidad})", fontsize=8, labelpad=4)
    ax.set_ylabel(f"Error ({unidad})", fontsize=8, labelpad=4)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.fill_between(referencias, errores, color="#003366", alpha=0.08)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return tmp.name


def crear_grafica_doble_parametros(datos_p1, datos_p2, label_p1, label_p2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    ref_1 = [d["referencia"] for d in datos_p1]
    err_1 = [d["correccion"] * -1 for d in datos_p1]
    ax1.plot(ref_1, err_1, marker="o", color="#003366", linewidth=2)
    ax1.set_title(
        f"ERROR EN {label_p1}", fontsize=9, fontweight="bold", color="#003366"
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.fill_between(ref_1, err_1, color="#003366", alpha=0.08)

    ref_2 = [d["referencia"] for d in datos_p2]
    err_2 = [d["correccion"] * -1 for d in datos_p2]
    ax2.plot(ref_2, err_2, marker="s", color="#006699", linewidth=2)
    ax2.set_title(
        f"ERROR EN {label_p2}", fontsize=9, fontweight="bold", color="#006699"
    )
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.fill_between(ref_2, err_2, color="#006699", alpha=0.08)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return tmp.name


def crear_audiograma_audiometro(datos_freq):
    frecuencias = [d["referencia"] for d in datos_freq]
    db_od = [15, 10, 10, 5, 10, 15, 20]
    db_oi = [20, 15, 10, 10, 15, 20, 25]
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(
        frecuencias,
        db_od,
        marker="o",
        color="red",
        linestyle="-",
        linewidth=1.5,
        label="Oído Derecho (OD)",
    )
    ax.plot(
        frecuencias,
        db_oi,
        marker="x",
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label="Oído Izquierdo (OI)",
    )
    ax.set_title(
        "AUDIOGRAMA TONAL - EVALUACIÓN ACÚSTICA",
        fontsize=9,
        fontweight="bold",
        color="#003366",
    )
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("Nivel de Audición (dB HL)", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(frecuencias)
    ax.set_xticklabels([str(f) for f in frecuencias], fontsize=8)
    ax.set_ylim(100, -10)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return tmp.name


def crear_grafica_ecg():
    t = np.linspace(0, 2.5, 800)
    ecg = np.zeros_like(t)
    for i in range(2):
        shift = i * 1.25
        ecg += 0.15 * np.exp(-(((t - (0.2 + shift)) ** 2) / 0.005))
        ecg -= 0.1 * np.exp(-(((t - (0.38 + shift)) ** 2) / 0.0005))
        ecg += 1.3 * np.exp(-(((t - (0.4 + shift)) ** 2) / 0.0008))
        ecg -= 0.25 * np.exp(-(((t - (0.42 + shift)) ** 2) / 0.0005))
        ecg += 0.35 * np.exp(-(((t - (0.65 + shift)) ** 2) / 0.02))
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(t, ecg, color="#cc0000", linewidth=1.5)
    ax.set_title(
        "TRAZADO SIMULADO DE ELECTROCARDIOGRAMA (DERIVACIÓN II)",
        fontsize=9,
        fontweight="bold",
        color="#003366",
    )
    ax.set_xlabel("Tiempo (s)", fontsize=8)
    ax.set_ylabel("Amplitud (mV)", fontsize=8)
    ax.grid(True, linestyle="-", linewidth=0.5, color="#ffcccc")
    ax.set_facecolor("#fffafa")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return tmp.name


def crear_grafica_espirometria_curva():
    volumen = np.linspace(0, 5, 200)
    flujo = 10 * np.sqrt(np.maximum(0, 1 - (volumen / 5) ** 1.5))
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(
        volumen, flujo, color="#006699", linewidth=2, label="Espiración Forzada"
    )
    ax.set_title(
        "CURVA FLUJO - VOLUMEN (ESPIROMETRÍA)",
        fontsize=9,
        fontweight="bold",
        color="#003366",
    )
    ax.set_xlabel("Volumen (L)", fontsize=8)
    ax.set_ylabel("Flujo (L/s)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.fill_between(volumen, flujo, color="#006699", alpha=0.1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        plt.tight_layout()
        plt.savefig(tmp.name, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return tmp.name


# ==========================================
# 3. CLASES FPDF (CERTIFICADO Y STICKERS)
# ==========================================
class CertificadoProfesional(FPDF):

    def __init__(self, num_cert, qr_path=None):
        super().__init__()
        self.num_cert = str(num_cert)
        self.qr_path = qr_path

    def header(self):
        self.set_font("Arial", "B", 50)
        self.set_text_color(242, 245, 250)
        self.text(30, 140, "EXALUD S.A.S")

        if self.qr_path and os.path.exists(self.qr_path):
            self.image(self.qr_path, x=175, y=8, w=20)
            self.set_xy(172, 29)
            self.set_font("Arial", "I", 6)
            self.set_text_color(100, 100, 100)
            self.cell(26, 3, "Escanear Autenticidad", 0, 0, "C")

        self.set_y(10)
        self.set_font("Arial", "B", 15)
        self.set_text_color(0, 51, 102)
        self.cell(0, 6, "EXALUD S.A.S   |   NIT: 901128787-7", 0, 1, "C")
        self.set_font("Arial", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "DIVISION DE METROLOGIA", 0, 1, "C")
        self.set_font("Arial", "B", 11)
        self.set_text_color(0, 0, 0)
        self.cell(
            0,
            6,
            f"CERTIFICADO DE CALIBRACION N° {self.num_cert}",
            0,
            1,
            "C",
        )
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 7.5)
        self.set_text_color(128, 128, 128)
        self.cell(
            0,
            10,
            f"Página {self.page_no()} | EXALUD S.A.S (NIT: 901128787-7) - Calle 19 # 3-16, Local 115, Barichara | Tel: 3237976148",
            0,
            0,
            "C",
        )


def generar_pdf_profesional(
    datos_excel, num_cert=None, nombre_ejecutor="", firma_path=None
):
    if not num_cert or num_cert == 8000 or str(num_cert).strip() == "":
        num_cert = random.randint(80000, 99999)

    info_sim = simular_mediciones(datos_excel)
    patron = info_sim["patron"]
    tipo_equipo = str(datos_excel.get("equipo", "N/A")).upper()
    sede = str(datos_excel.get("sede", "N/A")).upper()

    fecha_excel = datos_excel.get(
        "fecha de calibracion",
        datos_excel.get("fecha calibracion", datetime.now()),
    )
    if isinstance(fecha_excel, pd.Timestamp) or isinstance(
        fecha_excel, datetime
    ):
        fecha_texto = fecha_excel.strftime("%d/%m/%Y")
    else:
        fecha_texto = (
            str(fecha_excel)
            if str(fecha_excel).strip() != ""
            else datetime.now().strftime("%d/%m/%Y")
        )

    ejecutor_final = str(
        datos_excel.get(
            "metrólogo",
            datos_excel.get(
                "metrologo", datos_excel.get("ejecutó", nombre_ejecutor)
            ),
        )
    ).upper()
    direccion_sede = str(
        datos_excel.get(
            "dirección",
            datos_excel.get(
                "direccion",
                datos_excel.get("dirección sede", "Instalaciones del cliente"),
            ),
        )
    )

    texto_qr = f"Exalud S.A.S\nCertificado N°: {num_cert}\nEquipo: {tipo_equipo}\nSede: {sede}"
    qr_path = generar_qr_temp(texto_qr)

    pdf = CertificadoProfesional(num_cert, qr_path=qr_path)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -------------------------------
    # PÁGINA 1
    # -------------------------------
    pdf.add_page()

    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190,
        6,
        " 1. INFORMACION DEL INSTRUMENTO / Instrument Information",
        1,
        1,
        "L",
        fill=True,
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " EQUIPO:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(60, 6, f" {datos_excel.get('equipo', 'N/A')}", 1, 0, "L")
    pdf.set_font("Arial", "B", 8)
    pdf.cell(40, 6, " FECHA CALIBRACION:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(55, 6, f" {fecha_texto}", 1, 1, "L")

    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " MARCA:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(60, 6, f" {datos_excel.get('marca', 'N/A')}", 1, 0, "L")
    pdf.set_font("Arial", "B", 8)
    pdf.cell(40, 6, " SERIE:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(55, 6, f" {datos_excel.get('serie', 'N/A')}", 1, 1, "L")

    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " MODELO:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(60, 6, f" {datos_excel.get('modelo', 'N/A')}", 1, 0, "L")
    pdf.set_font("Arial", "B", 8)
    pdf.cell(40, 6, " CÓDIGO / ACTIVO:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(55, 6, f" {datos_excel.get('código interno', 'N/A')}", 1, 1, "L")
    pdf.ln(3)

    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190,
        6,
        " 2. INFORMACION DEL SOLICITANTE / Customer Information",
        1,
        1,
        "L",
        fill=True,
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " RAZON SOCIAL:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(155, 6, f" {datos_excel.get('entidad', 'N/A')}", 1, 1, "L")

    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " SEDE:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(60, 6, f" {datos_excel.get('sede', 'N/A')}", 1, 0, "L")
    pdf.set_font("Arial", "B", 8)
    pdf.cell(40, 6, " UBICACION:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(55, 6, f" {datos_excel.get('ubicación', 'N/A')}", 1, 1, "L")

    pdf.set_font("Arial", "B", 8)
    pdf.cell(35, 6, " DIRECCIÓN SEDE:", 1, 0, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(155, 6, f" {direccion_sede}", 1, 1, "L")
    pdf.ln(3)

    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190,
        6,
        " 3. CONDICIONES AMBIENTALES / Environmental Conditions",
        1,
        1,
        "L",
        fill=True,
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(47.5, 6, "Temperatura Inicial", 1, 0, "C")
    pdf.cell(47.5, 6, "Temperatura Final", 1, 0, "C")
    pdf.cell(47.5, 6, "Humedad Relativa Inicial", 1, 0, "C")
    pdf.cell(47.5, 6, "Humedad Relativa Final", 1, 1, "C")

    pdf.set_font("Arial", "", 8)
    pdf.cell(47.5, 6, "20.1 °C", 1, 0, "C")
    pdf.cell(47.5, 6, "20.4 °C", 1, 0, "C")
    pdf.cell(47.5, 6, "52.0 %", 1, 0, "C")
    pdf.cell(47.5, 6, "54.5 %", 1, 1, "C")
    pdf.ln(3)

    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190,
        6,
        " 4. METODOLOGIA, TRAZABILIDAD E INCERTIDUMBRE / Framework",
        1,
        1,
        "L",
        fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 8)

    marco_text = (
        "MÉTODO DE CALIBRACIÓN: Calibración realizada por método de comparación directa contra patrones de referencia "
        "trazables al Sistema Internacional de Unidades (SI). El procedimiento técnico garantiza la verificación "
        "instantánea de los valores indicados por el instrumento en relación con la magnitud de referencia aplicada.\n\n"
        "TRAZABILIDAD METROLÓGICA: Exalud S.A.S. establece la trazabilidad de sus mediciones al SI mediante una cadena "
        "ininterrumpida de calibraciones referidas a patrones primarios custodiados por laboratorios acreditados bajo "
        "la norma ISO/IEC 17025:2017 por organismos firmantes del acuerdo de reconocimiento mutuo ILAC MRA.\n\n"
        "INCERTIDUMBRE DE MEDICIÓN: La incertidumbre expandida reportada se determinó conforme a la guía JCGM 100:2008 (GUM). "
        "Corresponde a la incertidumbre combinada multiplicada por un factor de cobertura k=2, con un nivel de confianza "
        "de aproximadamente 95% bajo una distribución normal."
    )
    pdf.multi_cell(190, 4.2, marco_text, border="LRB", align="J")
    pdf.ln(3)

    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190,
        6,
        " 5. PATRONES DE REFERENCIA UTILIZADOS / Reference Standards",
        1,
        1,
        "L",
        fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 8)

    pdf.cell(55, 6, "EQUIPO PATRON", 1, 0, "C")
    pdf.cell(35, 6, "MARCA", 1, 0, "C")
    pdf.cell(50, 6, "N° CERTIFICADO", 1, 0, "C")
    pdf.cell(50, 6, "FECHA CALIBRACION", 1, 1, "C")

    pdf.set_font("Arial", "", 8)
    pdf.cell(55, 6, patron["equipo"], 1, 0, "C")
    pdf.cell(35, 6, patron["marca"], 1, 0, "C")
    pdf.cell(50, 6, patron["cert"], 1, 0, "C")
    pdf.cell(50, 6, patron["fecha_calibracion"], 1, 1, "C")
    pdf.ln(3)

    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(
        190, 6, " 6. OBSERVACIONES / Observations", 1, 1, "L", fill=True
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 8)
    obs_text = (
        "Este certificado expresa fielmente los resultados de las mediciones realizadas. Los resultados se relacionan "
        "única y exclusivamente con el ítem sometido a calibración en las condiciones del momento de prueba. "
        "Este documento no podrá ser reproducido parcial ni totalmente sin autorización escrita del laboratorio. "
        "Es responsabilidad del usuario determinar la frecuencia de recalibración de sus instrumentos según su sistema de gestión de calidad."
    )
    pdf.multi_cell(190, 4.2, obs_text, border="LRB", align="J")

    pdf.ln(5)

    if firma_path and os.path.exists(firma_path):
        x_actual = pdf.get_x()
        y_actual = pdf.get_y()
        pdf.image(firma_path, x=x_actual + 15, y=y_actual, h=12)
        pdf.ln(12)
    else:
        pdf.ln(15)

    pdf.set_font("Arial", "B", 8)
    pdf.cell(95, 4, "________________________________________", 0, 1, "L")
    pdf.cell(95, 4, f"{ejecutor_final}", 0, 1, "L")
    pdf.set_font("Arial", "", 8)
    pdf.cell(95, 4, "Ingeniero Metrólogo (Ejecutor)", 0, 1, "L")

    # -------------------------------
    # PÁGINA 2
    # -------------------------------
    pdf.add_page()
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(190, 7, "7. RESULTADOS DE LA CALIBRACION", 0, 1, "C")
    pdf.ln(2)

    rutas_a_limpiar = [qr_path]

    if info_sim.get("es_multiparametro", False):
        for p in info_sim["parametros"]:
            pdf.set_text_color(0, 51, 102)
            pdf.set_font("Arial", "B", 9)
            if "%HR" in p["unidad"] or "°C" in p["unidad"]:
                pdf.cell(
                    190,
                    5,
                    f"MAGNITUD EVALUADA: {p['nombre']} ({p['unidad']})",
                    0,
                    1,
                    "L",
                )
            else:
                pdf.cell(
                    190, 5, f"MAGNITUD EVALUADA: {p['nombre']}", 0, 1, "L"
                )

            pdf.set_fill_color(220, 230, 245)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "B", 8)
            anchos_res = [20, 35, 35, 35, 40, 25]
            encabezados = [
                "ÍTEM",
                "VALOR REF.",
                "INDICACIÓN",
                "CORRECCIÓN",
                "INCERTIDUMBRE (U)",
                "FACTOR (k)",
            ]
            for i in range(len(encabezados)):
                pdf.cell(
                    anchos_res[i], 6, encabezados[i], 1, 0, "C", fill=True
                )
            pdf.ln()

            pdf.set_font("Arial", "", 8)
            for i, fila in enumerate(p["datos"]):
                pdf.cell(anchos_res[0], 5.5, str(i + 1), 1, 0, "C")
                pdf.cell(
                    anchos_res[1],
                    5.5,
                    f"{fila['referencia']} {p['unidad']}",
                    1,
                    0,
                    "C",
                )
                pdf.cell(
                    anchos_res[2],
                    5.5,
                    f"{fila['indicacion']} {p['unidad']}",
                    1,
                    0,
                    "C",
                )
                pdf.cell(
                    anchos_res[3],
                    5.5,
                    f"{fila['correccion']} {p['unidad']}",
                    1,
                    0,
                    "C",
                )
                pdf.cell(
                    anchos_res[4],
                    5.5,
                    f"± {fila['incertidumbre']} {p['unidad']}",
                    1,
                    0,
                    "C",
                )
                pdf.cell(anchos_res[5], 5.5, "k = 2", 1, 1, "C")
            pdf.ln(3)

        p1 = info_sim["parametros"][0]
        p2 = info_sim["parametros"][1]
        lbl1 = "%" if p1["unidad"] == "%" else p1["unidad"]
        lbl2 = "BPM" if p2["unidad"] == "BPM" else p2["unidad"]

        ruta_grafica = crear_grafica_doble_parametros(
            p1["datos"], p2["datos"], lbl1, lbl2
        )
        rutas_a_limpiar.append(ruta_grafica)
        pdf.image(ruta_grafica, x=10, w=190)
        pdf.ln(3)

    elif info_sim.get("es_bascula", False):
        p = info_sim["parametros"][0]
        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(190, 5, "7.1. PRUEBA DE EXACTITUD Y LINEALIDAD", 0, 1, "L")

        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 8)
        anchos_res = [20, 35, 35, 35, 40, 25]
        encabezados = [
            "ÍTEM",
            "VALOR REF.",
            "INDICACIÓN",
            "CORRECCIÓN",
            "INCERTIDUMBRE (U)",
            "FACTOR (k)",
        ]
        for i in range(len(encabezados)):
            pdf.cell(anchos_res[i], 6, encabezados[i], 1, 0, "C", fill=True)
        pdf.ln()

        pdf.set_font("Arial", "", 8)
        for i, fila in enumerate(p["datos"]):
            pdf.cell(anchos_res[0], 5, str(i + 1), 1, 0, "C")
            pdf.cell(anchos_res[1], 5, f"{fila['referencia']} kg", 1, 0, "C")
            pdf.cell(anchos_res[2], 5, f"{fila['indicacion']} kg", 1, 0, "C")
            pdf.cell(anchos_res[3], 5, f"{fila['correccion']} kg", 1, 0, "C")
            pdf.cell(
                anchos_res[4], 5, f"± {fila['incertidumbre']} kg", 1, 0, "C"
            )
            pdf.cell(anchos_res[5], 5, "k = 2", 1, 1, "C")
        pdf.ln(3)

        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 9)
        exc = info_sim["excentricidad"]
        pdf.cell(
            190,
            5,
            f"7.2. PRUEBA DE EXCENTRICIDAD (Carga Aplicada: {exc['carga']} kg)",
            0,
            1,
            "L",
        )

        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 8)
        pdf.cell(
            95, 5.5, "POSICIÓN DE CARGA EN PLATILLO", 1, 0, "C", fill=True
        )
        pdf.cell(47.5, 5.5, "VALOR INDICADO", 1, 0, "C", fill=True)
        pdf.cell(47.5, 5.5, "ERROR DE EXCENTRICIDAD", 1, 1, "C", fill=True)

        pdf.set_font("Arial", "", 8)
        for pos in exc["posiciones"]:
            pdf.cell(95, 5, pos["pos"], 1, 0, "L")
            pdf.cell(47.5, 5, f"{pos['ind']} kg", 1, 0, "C")
            pdf.cell(47.5, 5, f"{pos['error']} kg", 1, 1, "C")
        pdf.ln(3)

        rep = info_sim["repetibilidad"]
        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(
            190,
            5,
            f"7.3. PRUEBA DE REPETIBILIDAD (Carga Aplicada: {rep['carga']} kg)",
            0,
            1,
            "L",
        )

        meds_str = " | ".join(
            [f"M{i+1}: {v} kg" for i, v in enumerate(rep["mediciones"])]
        )
        pdf.set_font("Arial", "", 8)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(190, 5, f"Lecturas Consecutivas: {meds_str}", 1, 1, "C")
        pdf.set_font("Arial", "B", 8)
        pdf.cell(
            190,
            5,
            f"Desviación Estándar de Repetibilidad (s): ± {rep['std_dev']} kg",
            1,
            1,
            "C",
        )
        pdf.ln(3)

        ruta_grafica = crear_grafica(p["datos"], "kg")
        rutas_a_limpiar.append(ruta_grafica)
        pdf.image(ruta_grafica, x=20, w=150)
        pdf.ln(2)

    elif info_sim.get("es_audiometro", False):
        p1 = info_sim["parametros"][0]
        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(190, 5, f"7.1. {p1['nombre']}", 0, 1, "L")

        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 8)
        anchos_res = [20, 35, 35, 35, 40, 25]
        encabezados = [
            "ÍTEM",
            "MAGNITUD",
            "VALOR REF.",
            "INDICACIÓN",
            "CORRECCIÓN",
            "INCERTIDUMBRE",
        ]
        for i in range(len(encabezados)):
            pdf.cell(anchos_res[i], 6, encabezados[i], 1, 0, "C", fill=True)
        pdf.ln()

        pdf.set_font("Arial", "", 8)
        for i, fila in enumerate(p1["datos"]):
            pdf.cell(anchos_res[0], 5, str(i + 1), 1, 0, "C")
            pdf.cell(anchos_res[1], 5, fila["parametro"], 1, 0, "C")
            pdf.cell(anchos_res[2], 5, f"{fila['referencia']} Hz", 1, 0, "C")
            pdf.cell(anchos_res[3], 5, f"{fila['indicacion']} Hz", 1, 0, "C")
            pdf.cell(anchos_res[4], 5, f"{fila['correccion']} Hz", 1, 0, "C")
            pdf.cell(
                anchos_res[5], 5, f"± {fila['incertidumbre']} Hz", 1, 1, "C"
            )
        pdf.ln(3)

        p2 = info_sim["parametros"][1]
        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(190, 5, f"7.2. {p2['nombre']}", 0, 1, "L")

        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 8)
        for i in range(len(encabezados)):
            pdf.cell(anchos_res[i], 6, encabezados[i], 1, 0, "C", fill=True)
        pdf.ln()

        pdf.set_font("Arial", "", 8)
        for i, fila in enumerate(p2["datos"]):
            pdf.cell(anchos_res[0], 5, str(i + 1), 1, 0, "C")
            pdf.cell(anchos_res[1], 5, fila["parametro"], 1, 0, "C")
            pdf.cell(anchos_res[2], 5, f"{fila['referencia']} dB", 1, 0, "C")
            pdf.cell(anchos_res[3], 5, f"{fila['indicacion']} dB", 1, 0, "C")
            pdf.cell(anchos_res[4], 5, f"{fila['correccion']} dB", 1, 0, "C")
            pdf.cell(
                anchos_res[5], 5, f"± {fila['incertidumbre']} dB", 1, 1, "C"
            )
        pdf.ln(3)

        ruta_grafica = crear_audiograma_audiometro(p1["datos"])
        rutas_a_limpiar.append(ruta_grafica)
        pdf.image(ruta_grafica, x=20, w=170)
        pdf.ln(3)

    else:
        p = info_sim["parametros"][0]
        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 8)
        pdf.cell(
            190,
            5,
            f'Magnitud Evaluada: {tipo_equipo} | Unidad de medida: {p["unidad"]}',
            0,
            1,
            "C",
        )
        pdf.ln(2)

        pdf.set_fill_color(220, 230, 245)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 8)
        anchos_res = [20, 35, 35, 35, 40, 25]
        encabezados = [
            "ÍTEM",
            "VALOR REF.",
            "INDICACIÓN",
            "CORRECCIÓN",
            "INCERTIDUMBRE (U)",
            "FACTOR (k)",
        ]
        for i in range(len(encabezados)):
            pdf.cell(anchos_res[i], 7, encabezados[i], 1, 0, "C", fill=True)
        pdf.ln()

        pdf.set_font("Arial", "", 8)
        for i, fila in enumerate(p["datos"]):
            pdf.cell(anchos_res[0], 6, str(i + 1), 1, 0, "C")
            pdf.cell(anchos_res[1], 6, str(fila["referencia"]), 1, 0, "C")
            pdf.cell(anchos_res[2], 6, str(fila["indicacion"]), 1, 0, "C")
            pdf.cell(anchos_res[3], 6, str(fila["correccion"]), 1, 0, "C")
            pdf.cell(anchos_res[4], 6, f"± {fila['incertidumbre']}", 1, 0, "C")
            pdf.cell(anchos_res[5], 6, "k = 2", 1, 1, "C")
        pdf.ln(4)

        if info_sim.get("es_ecg", False):
            ruta_grafica = crear_grafica_ecg()
        elif info_sim.get("es_espirometro", False):
            ruta_grafica = crear_grafica_espirometria_curva()
        else:
            ruta_grafica = crear_grafica(p["datos"], p["unidad"])

        rutas_a_limpiar.append(ruta_grafica)
        pdf.image(ruta_grafica, x=20, w=170)
        pdf.ln(4)

    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(0, 51, 102)
    err_max_total = max([p["error_max"] for p in info_sim["parametros"]])
    unid_main = (
        "Hz / dB"
        if info_sim.get("es_audiometro", False)
        else info_sim["parametros"][0]["unidad"]
    )
    pdf.cell(
        190,
        6,
        f"ERROR MAXIMO ENCONTRADO EN EL INSTRUMENTO: {err_max_total} {unid_main}",
        0,
        1,
        "C",
    )

    pdf.ln(6)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(190, 6, "*** FIN DEL CERTIFICADO ***", 0, 1, "C")

    pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")

    for r in rutas_a_limpiar:
        try:
            os.remove(r)
        except:
            pass

    return pdf_bytes


def generar_hoja_stickers(datos_stickers):
    pdf = FPDF("P", "mm", "A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    ancho_st = 46
    alto_st = 27
    margen_x = 10
    margen_y = 10
    espacio_x = 3
    espacio_y = 3

    col = 0
    fila = 0

    for st_data in datos_stickers:
        x = margen_x + col * (ancho_st + espacio_x)
        y = margen_y + fila * (alto_st + espacio_y)

        pdf.set_draw_color(0, 51, 102)
        pdf.set_line_width(0.3)
        pdf.rect(x, y, ancho_st, alto_st)

        pdf.set_xy(x, y + 2)
        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(ancho_st, 4, "EXALUD S.A.S", 0, 1, "C")

        pdf.set_fill_color(0, 204, 102)
        pdf.set_xy(x + 2, y + 7)
        pdf.set_font("Arial", "B", 7.5)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(ancho_st - 4, 4.5, "CALIBRADO", 0, 1, "C", fill=True)

        pdf.set_text_color(0, 51, 102)
        pdf.set_font("Arial", "B", 6)

        pdf.set_xy(x + 2, y + 13.5)
        pdf.cell(17, 3.5, "CERTIFICADO:", 0, 0, "L")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(24, 3.5, str(st_data["cert"]), 0, 1, "L")
        pdf.set_draw_color(180, 180, 180)
        pdf.set_line_width(0.2)
        pdf.line(x + 19, y + 16.5, x + ancho_st - 2, y + 16.5)

        pdf.set_text_color(0, 51, 102)
        pdf.set_xy(x + 2, y + 17.5)
        pdf.cell(17, 3.5, "EQUIPO:", 0, 0, "L")
        pdf.set_text_color(0, 0, 0)
        eq_str = str(st_data["equipo"]).upper()[:18]
        pdf.cell(24, 3.5, eq_str, 0, 1, "L")
        pdf.line(x + 19, y + 20.5, x + ancho_st - 2, y + 20.5)

        pdf.set_text_color(0, 51, 102)
        pdf.set_xy(x + 2, y + 21.5)
        pdf.cell(17, 3.5, "FECHA CAL.:", 0, 0, "L")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(24, 3.5, str(st_data["fecha"]), 0, 1, "L")
        pdf.line(x + 19, y + 24.5, x + ancho_st - 2, y + 24.5)

        col += 1
        if col > 3:
            col = 0
            fila += 1
            if fila > 9:
                fila = 0
                pdf.add_page()

    return pdf.output(dest="S").encode("latin-1", "replace")


# ==========================================
# 4. INTERFAZ WEB STREAMLIT
# ==========================================
st.markdown(
    "<h1 style='text-align: center; color: #004080;'>EXALUD S.A.S - Sistema Integral de Metrología</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuraciones Corporativas")

    st.subheader("1. Identidad Visual")
    firma_file = st.file_uploader(
        "🖋️ Cargar Firma Digital", type=["png", "jpg", "jpeg"]
    )
    if firma_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(firma_file.read())
            st.session_state["firma_tmp_path"] = tmp.name

    firma_tmp_path = st.session_state.get("firma_tmp_path", None)

    nombre_metrologo = st.text_input(
        "👨‍🔬 Nombre del Metrólogo:", "ESTEBAN VERGARA"
    )

    st.markdown("---")
    st.subheader("2. Gestión de Patrones (Asociación Dinámica)")
    st.caption(
        "Edita las 4 variables de cada patrón. Se asignarán automáticamente a los equipos correspondientes."
    )

    with st.expander("🛠️ Editar Fechas, Equipos y Certificados"):
        for clave, valores in st.session_state["patrones_cfg"].items():
            st.markdown(
                f"<span style='color:#004080; font-weight:bold;'>{clave}</span> - *{valores['desc']}*",
                unsafe_allow_html=True,
            )

            cols1 = st.columns(2)
            st.session_state["patrones_cfg"][clave]["equipo"] = cols1[
                0
            ].text_input("Equipo Patrón", valores["equipo"], key=f"e_{clave}")
            st.session_state["patrones_cfg"][clave]["marca"] = cols1[
                1
            ].text_input("Marca", valores["marca"], key=f"m_{clave}")

            cols2 = st.columns(2)
            st.session_state["patrones_cfg"][clave]["cert"] = cols2[
                0
            ].text_input("N° Certificado", valores["cert"], key=f"c_{clave}")
            st.session_state["patrones_cfg"][clave]["fecha_calibracion"] = (
                cols2[1].text_input(
                    "Fecha Calibración",
                    valores["fecha_calibracion"],
                    key=f"f_{clave}",
                )
            )

            st.markdown("---")

archivo_excel = st.file_uploader(
    "📂 Cargar matriz maestra (.xlsx)", type=["xlsx", "xls"]
)

if archivo_excel:
    try:
        df = pd.read_excel(archivo_excel)
        df.columns = df.columns.str.lower().str.strip()
        df = df.fillna("N/A")

        columnas_requeridas = [
            "equipo",
            "marca",
            "modelo",
            "serie",
            "código interno",
            "ubicación",
            "sede",
            "entidad",
        ]
        faltantes = [
            col for col in columnas_requeridas if col not in df.columns
        ]

        if not faltantes:
            opciones_equipos = (
                df["código interno"].astype(str)
                + " - "
                + df["equipo"].astype(str)
            )

            col1, col2 = st.columns([1, 1.3])

            with col1:
                equipo_seleccionado = st.selectbox(
                    "Seleccionar equipo para auditar (Vista Previa):",
                    opciones_equipos,
                )
                indice = opciones_equipos[
                    opciones_equipos == equipo_seleccionado
                ].index[0]
                datos_equipo = {
                    str(k).lower().strip(): v for k, v in df.iloc[indice].items()
                }

                st.write("---")
                st.write("### 🗂️ Exportación Estructural")

                if st.button("🚀 Procesar Lote Completo", use_container_width=True):
                    total_equipos = len(df)

                    progress_text = (
                        "Generando estructura documental y stickers..."
                    )
                    my_bar = st.progress(0, text=progress_text)

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(
                        zip_buffer, "w", zipfile.ZIP_DEFLATED
                    ) as zip_file:

                        lista_stickers = []

                        for idx, fila in df.iterrows():
                            porcentaje = int(((idx + 1) / total_equipos) * 100)
                            my_bar.progress(
                                (idx + 1) / total_equipos,
                                text=f"Procesando: {idx + 1} de {total_equipos} ({porcentaje}%)",
                            )

                            num_cert = random.randint(80000, 99999)
                            d_fila = {
                                str(k).lower().strip(): v
                                for k, v in fila.items()
                            }

                            pdf_bytes = generar_pdf_profesional(
                                d_fila,
                                num_cert=num_cert,
                                nombre_ejecutor=nombre_metrologo,
                                firma_path=firma_tmp_path,
                            )

                            fecha_excel = d_fila.get(
                                "fecha de calibracion",
                                d_fila.get(
                                    "fecha calibracion", datetime.now()
                                ),
                            )
                            if isinstance(
                                fecha_excel, pd.Timestamp
                            ) or isinstance(fecha_excel, datetime):
                                fecha_texto = fecha_excel.strftime("%d/%m/%Y")
                            else:
                                fecha_texto = (
                                    str(fecha_excel)
                                    if str(fecha_excel).strip() != ""
                                    else datetime.now().strftime("%d/%m/%Y")
                                )

                            lista_stickers.append(
                                {
                                    "cert": num_cert,
                                    "equipo": d_fila.get("equipo", "N/A"),
                                    "fecha": fecha_texto,
                                }
                            )

                            cod_interno = limpiar_nombre_archivo(
                                d_fila.get("código interno", "S-COD")
                            )
                            ubicacion = limpiar_nombre_archivo(
                                d_fila.get("ubicación", "S-UBIC")
                            )
                            sede_limpia = limpiar_nombre_archivo(
                                d_fila.get("sede", "Sin Sede")
                            )

                            ruta_en_zip = f"{sede_limpia}/{cod_interno} - {ubicacion}.pdf"
                            zip_file.writestr(ruta_en_zip, pdf_bytes)

                        pdf_stickers_lote = generar_hoja_stickers(
                            lista_stickers
                        )
                        zip_file.writestr(
                            "00_Stickers_Calibracion_Lote.pdf",
                            pdf_stickers_lote,
                        )

                    st.success(
                        f"¡Lote finalizado! {total_equipos} certificados y su hoja de stickers generados con éxito."
                    )
                    st.download_button(
                        "⬇️ Descargar Archivo ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="Metrologia_Exalud_Premium.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                    )

            with col2:
                st.write("### 📄 Visualizador de Documentos en Alta Calidad")
                cert_aleatorio = random.randint(80000, 99999)
                pdf_preview = generar_pdf_profesional(
                    datos_equipo,
                    num_cert=cert_aleatorio,
                    nombre_ejecutor=nombre_metrologo,
                    firma_path=firma_tmp_path,
                )

                fecha_excel = datos_equipo.get(
                    "fecha de calibracion",
                    datos_equipo.get("fecha calibracion", datetime.now()),
                )
                if isinstance(fecha_excel, pd.Timestamp) or isinstance(
                    fecha_excel, datetime
                ):
                    fecha_texto = fecha_excel.strftime("%d/%m/%Y")
                else:
                    fecha_texto = (
                        str(fecha_excel)
                        if str(fecha_excel).strip() != ""
                        else datetime.now().strftime("%d/%m/%Y")
                    )

                sticker_individual = generar_hoja_stickers(
                    [
                        {
                            "cert": cert_aleatorio,
                            "equipo": datos_equipo.get("equipo", "N/A"),
                            "fecha": fecha_texto,
                        }
                    ]
                )

                cod_ind = limpiar_nombre_archivo(
                    datos_equipo.get("código interno", "S-COD")
                )
                ubic_ind = limpiar_nombre_archivo(
                    datos_equipo.get("ubicación", "S-UBIC")
                )
                nombre_archivo_individual = f"{cod_ind} - {ubic_ind}.pdf"

                cols_descarga = st.columns(2)
                with cols_descarga[0]:
                    st.download_button(
                        label="📥 Descargar Certificado",
                        data=pdf_preview,
                        file_name=nombre_archivo_individual,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with cols_descarga[1]:
                    st.download_button(
                        label="🏷️ Descargar Sticker",
                        data=sticker_individual,
                        file_name=f"Sticker_{nombre_archivo_individual}",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                st.write("---")

                doc = fitz.open(stream=pdf_preview, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    st.image(
                        img_bytes,
                        caption=f"Página {page_num + 1} de {len(doc)}",
                        use_container_width=True,
                    )
                doc.close()

        else:
            st.error(
                f"⚠️ Faltan columnas obligatorias: {', '.join(faltantes)}"
            )

    except Exception as e:
        st.error(f"Error crítico en el motor: {e}")
