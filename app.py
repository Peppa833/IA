from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import threading
import os
import datetime
import traceback

from generate import generar
from auto_train import should_train, auto_train

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    message: str

# Variable global para controlar entrenamiento
entrenamiento_activo = False

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(req: ChatReq):
    global entrenamiento_activo
    
    user_msg = req.message.strip()
    
    if not user_msg:
        return {"respuesta": "Por favor, escribe un mensaje"}
    
    # Generar respuesta
    respuesta = generar(user_msg.lower())
    
    # Verificar que la respuesta no esté vacía
    if not respuesta or not respuesta.strip():
        respuesta = "No tengo una respuesta para eso ahora mismo"
    
    # Verificar longitud antes de guardar
    palabras_respuesta = len(respuesta.split())
    palabras_usuario = len(user_msg.split())
    
    print(f"Chat: '{user_msg}' -> '{respuesta}' ({palabras_respuesta} palabras)")
    
    # REGLA: Solo guardar si ambas son CORTAS (1-6 palabras)
    puede_guardar = (1 <= palabras_usuario <= 6) and (1 <= palabras_respuesta <= 6) and respuesta.strip()
    
    if puede_guardar:
        # SOLO guardar en chat_logs.txt
        try:
            with open("chat_logs.txt", "a", encoding="utf-8") as f:
                f.write(f"usuario: {user_msg}\n")
                f.write(f"ia: {respuesta}\n")
            print(f"Guardado en chat_logs: '{user_msg}' -> '{respuesta}'")
        except Exception as e:
            print(f"Error guardando en chat_logs: {e}")
    else:
        print(f"NO guardado: Usuario={palabras_usuario} palabras, Bot={palabras_respuesta} palabras")
        print(f"   (Solo guardo conversaciones de 1-6 palabras)")
    
    # Verificar y ejecutar autoentrenamiento
    try:
        if should_train() and not entrenamiento_activo:
            entrenamiento_activo = True
            print("Suficientes datos! Iniciando autoentrenamiento...")
            threading.Thread(target=ejecutar_entrenamiento, daemon=True).start()
    except Exception as e:
        print(f"Error verificando entrenamiento: {e}")
    
    return {"respuesta": respuesta}

def ejecutar_entrenamiento():
    global entrenamiento_activo
    try:
        print("Hilo de entrenamiento iniciado")
        auto_train()
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        traceback.print_exc()
    finally:
        entrenamiento_activo = False
        print("Hilo de entrenamiento finalizado")

@app.get("/estado")
def get_estado():
    """Ver estado completo del sistema"""
    estado = {
        "modelo_existe": os.path.exists("model.pth"),
        "chat_logs_existe": os.path.exists("chat_logs.txt"),
        "data_txt_existe": os.path.exists("data.txt"),
        "entrenamiento_activo": entrenamiento_activo,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    try:
        estado["debe_entrenar"] = should_train()
    except Exception as e:
        estado["debe_entrenar"] = False
        estado["error_should_train"] = str(e)
    
    # Contar líneas en chat_logs.txt
    if os.path.exists("chat_logs.txt"):
        try:
            with open("chat_logs.txt", "r", encoding="utf-8") as f:
                lineas = f.readlines()
                estado["lineas_chat_logs"] = len(lineas)
                estado["conversaciones_chat"] = len([l for l in lineas if l.startswith("usuario:")])
                
                # Verificar calidad de los datos
                lineas_validas = 0
                for linea in lineas:
                    if linea.strip() and len(linea.split()) <= 8:
                        lineas_validas += 1
                estado["lineas_validas_chat"] = lineas_validas
                
                # Contar caracteres totales
                estado["caracteres_chat"] = sum(len(l) for l in lineas)
        except Exception as e:
            estado["error_chat_logs"] = str(e)
    else:
        estado["lineas_chat_logs"] = 0
        estado["conversaciones_chat"] = 0
    
    # Contar líneas en data.txt
    if os.path.exists("data.txt"):
        try:
            with open("data.txt", "r", encoding="utf-8") as f:
                lineas = f.readlines()
                estado["lineas_data_txt"] = len(lineas)
                
                # Analizar longitud promedio
                longitudes = []
                for linea in lineas:
                    linea_limpia = linea.strip()
                    if linea_limpia:
                        palabras = len(linea_limpia.split())
                        longitudes.append(palabras)
                
                if longitudes:
                    estado["promedio_palabras_data"] = sum(longitudes) / len(longitudes)
                    estado["max_palabras_data"] = max(longitudes)
                    estado["lineas_largas_data"] = len([l for l in longitudes if l > 6])
                    estado["lineas_validas_data"] = len([l for l in longitudes if 1 <= l <= 6])
                else:
                    estado["promedio_palabras_data"] = 0
                    estado["max_palabras_data"] = 0
        except Exception as e:
            estado["error_data_txt"] = str(e)
    else:
        estado["lineas_data_txt"] = 0
    
    # Ver tamaño del modelo
    if os.path.exists("model.pth"):
        try:
            tamano = os.path.getsize("model.pth")
            estado["tamano_modelo_bytes"] = tamano
            estado["tamano_modelo_kb"] = tamano / 1024
            estado["tamano_modelo_mb"] = tamano / (1024 * 1024)
        except Exception as e:
            estado["error_tamano_modelo"] = str(e)
    
    # Verificar si hay archivo de lock
    estado["lock_file_existe"] = os.path.exists("training.lock")
    
    return JSONResponse(estado)

@app.post("/forzar_entrenamiento")
def forzar_entrenamiento():
    """Forzar entrenamiento manualmente"""
    global entrenamiento_activo
    
    if entrenamiento_activo:
        return JSONResponse({"mensaje": "Ya hay un entrenamiento en curso", "status": "ocupado"})
    
    # Verificar archivos necesarios
    archivos_necesarios = ["build_dataset.py", "train.py", "model.py", "tokenizer.py"]
    faltantes = []
    for archivo in archivos_necesarios:
        if not os.path.exists(archivo):
            faltantes.append(archivo)
    
    if faltantes:
        return JSONResponse({
            "mensaje": f"Faltan archivos: {', '.join(faltantes)}",
            "status": "error"
        }, status_code=500)
    
    # Asegurar que hay datos mínimos
    if not os.path.exists("chat_logs.txt") or os.path.getsize("chat_logs.txt") == 0:
        with open("chat_logs.txt", "w", encoding="utf-8") as f:
            f.write("usuario: hola\n")
            f.write("ia: hola como estas\n")
            f.write("usuario: que tal\n")
            f.write("ia: bien gracias\n")
        print("chat_logs.txt inicializado con datos básicos")
    
    if not os.path.exists("data.txt") or os.path.getsize("data.txt") == 0:
        with open("data.txt", "w", encoding="utf-8") as f:
            f.write("hola\ncomo estas\nbien\nque haces\nnada\nadios\n")
        print("data.txt inicializado con datos básicos")
    
    entrenamiento_activo = True
    threading.Thread(target=ejecutar_entrenamiento, daemon=True).start()
    
    return JSONResponse({
        "mensaje": "Entrenamiento forzado iniciado en segundo plano",
        "status": "iniciado",
        "timestamp": datetime.datetime.now().isoformat(),
        "nota": "El entrenamiento puede tardar varios minutos"
    })

@app.post("/limpiar_datos")
def limpiar_datos():
    """Limpiar datos corruptos o largos"""
    try:
        eliminadas_total = 0
        
        # 1. Limpiar chat_logs.txt (mantener solo líneas cortas)
        if os.path.exists("chat_logs.txt"):
            with open("chat_logs.txt", "r", encoding="utf-8") as f:
                lineas_originales = f.readlines()
            
            lineas_limpias = []
            for linea in lineas_originales:
                linea = linea.strip()
                if linea and len(linea.split()) <= 8:  # Máximo 8 palabras
                    lineas_limpias.append(linea + "\n")
            
            with open("chat_logs.txt", "w", encoding="utf-8") as f:
                f.writelines(lineas_limpias)
            
            eliminadas = len(lineas_originales) - len(lineas_limpias)
            eliminadas_total += eliminadas
            print(f"chat_logs.txt limpiado: {eliminadas} líneas largas eliminadas")
        
        # 2. Limpiar data.txt (mantener solo frases cortas)
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8") as f:
                lineas_originales = f.readlines()
            
            lineas_limpias = []
            for linea in lineas_originales:
                linea = linea.strip()
                if linea and 1 <= len(linea.split()) <= 6:  # 1-6 palabras
                    lineas_limpias.append(linea + "\n")
            
            with open("data.txt", "w", encoding="utf-8") as f:
                f.writelines(lineas_limpias)
            
            eliminadas = len(lineas_originales) - len(lineas_limpias)
            eliminadas_total += eliminadas
            print(f"data.txt limpiado: {eliminadas} líneas largas eliminadas")
        
        return JSONResponse({
            "mensaje": f"Datos limpiados exitosamente ({eliminadas_total} líneas eliminadas)",
            "status": "exito",
            "eliminadas_total": eliminadas_total
        })
        
    except Exception as e:
        print(f"Error limpiando datos: {e}")
        traceback.print_exc()
        return JSONResponse({
            "mensaje": f"Error limpiando datos: {e}",
            "status": "error"
        }, status_code=500)

@app.post("/reiniciar_modelo")
def reiniciar_modelo():
    """Borrar modelo y empezar desde cero"""
    try:
        if os.path.exists("model.pth"):
            os.remove("model.pth")
            print("Modelo anterior eliminado")
        
        # Crear data.txt básico si está vacío
        if not os.path.exists("data.txt") or os.path.getsize("data.txt") == 0:
            with open("data.txt", "w", encoding="utf-8") as f:
                f.write("hola\n")
                f.write("como estas\n")
                f.write("bien\n")
                f.write("que haces\n")
                f.write("nada\n")
                f.write("adios\n")
            print("data.txt básico creado")
        
        # Limpiar chat_logs.txt para empezar fresco
        if os.path.exists("chat_logs.txt"):
            open("chat_logs.txt", "w").close()
            print("chat_logs.txt limpiado")
        
        return JSONResponse({
            "mensaje": "Modelo reiniciado. Ejecuta /forzar_entrenamiento para entrenar",
            "status": "reiniciado"
        })
        
    except Exception as e:
        print(f"Error reiniciando modelo: {e}")
        traceback.print_exc()
        return JSONResponse({
            "mensaje": f"Error reiniciando modelo: {e}",
            "status": "error"
        }, status_code=500)

# Verificar si hay que entrenar al inicio
@app.on_event("startup")
def startup_event():
    try:
        print("=" * 50)
        print("Chatbot Autoentrenable Iniciando")
        print("=" * 50)
        
        # Crear directorio de backups si no existe
        os.makedirs("backups", exist_ok=True)
        
        # Crear archivos si no existen
        if not os.path.exists("data.txt"):
            with open("data.txt", "w", encoding="utf-8") as f:
                f.write("hola\ncomo estas\nbien\nque haces\nnada\nadios\n")
            print("data.txt creado con datos básicos")
        
        if not os.path.exists("chat_logs.txt"):
            open("chat_logs.txt", "w").close()
            print("chat_logs.txt creado vacío")
        
        # Verificar si hay datos para entrenar
        if should_train():
            print("Datos suficientes detectados al inicio")
            print("Usa /forzar_entrenamiento para entrenar manualmente")
        else:
            print("Necesitas al menos 3 conversaciones para autoentrenar")
            print("Chatea un poco y luego usa /forzar_entrenamiento")
        
        print("=" * 50)
        print("Servidor listo en http://localhost:8000")
        print("Estado disponible en /estado")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error en startup: {e}")
        traceback.print_exc()

# Página de administración
@app.get("/admin")
def admin_panel():
    return FileResponse("static/admin.html")

# Endpoint de salud
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "chatbot-autoentrenable"
    }

# Nuevo endpoint para ver logs
@app.get("/logs")
def get_logs():
    """Ver los logs de entrenamiento"""
    try:
        if os.path.exists("train.log"):
            with open("train.log", "r", encoding="utf-8") as f:
                contenido = f.read()
            return {"logs": contenido, "status": "ok"}
        else:
            return {"logs": "No hay logs aún", "status": "no_logs"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Nuevo endpoint para ver archivo específico
@app.get("/ver_archivo/{nombre}")
def ver_archivo(nombre: str):
    """Ver contenido de archivo específico"""
    archivos_permitidos = ["chat_logs.txt", "data.txt", "train.log"]
    
    if nombre not in archivos_permitidos:
        return JSONResponse({"error": "Archivo no permitido"}, status_code=403)
    
    try:
        if os.path.exists(nombre):
            with open(nombre, "r", encoding="utf-8") as f:
                contenido = f.read()
            return {"archivo": nombre, "contenido": contenido, "status": "ok"}
        else:
            return {"archivo": nombre, "contenido": "", "status": "no_existe"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)