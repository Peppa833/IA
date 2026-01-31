import os
import subprocess
import datetime
import time

THRESHOLD = 6  # Reducir para probar más fácil
LOCK_FILE = "training.lock"
CHAT_LOGS = "chat_logs.txt"
TRAIN_LOG = "train.log"

def log(msg):
    """Registrar en log con timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)  # Mostrar en consola también
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def should_train():
    """Verificar si hay suficientes datos para entrenar"""
    if not os.path.exists(CHAT_LOGS):
        log(f"❌ {CHAT_LOGS} no existe")
        return False
    
    try:
        with open(CHAT_LOGS, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Contar líneas que son conversación real
        conv_lines = [l for l in lines if l.startswith(("usuario:", "ia:"))]
        
        log(f"📊 Líneas en chat_logs.txt: {len(lines)}")
        log(f"📊 Líneas de conversación: {len(conv_lines)}")
        log(f"📊 Umbral necesario: {THRESHOLD}")
        
        return len(conv_lines) >= THRESHOLD
    
    except Exception as e:
        log(f"❌ Error en should_train: {e}")
        return False

def auto_train():
    """Función principal de autoentrenamiento"""
    
    # 🔒 Evitar múltiples entrenamientos simultáneos
    if os.path.exists(LOCK_FILE):
        log("⏸️ Entrenamiento ya en curso (lock file existe)")
        return
    
    # Crear archivo de bloqueo
    with open(LOCK_FILE, "w") as f:
        f.write(f"Entrenamiento iniciado: {datetime.datetime.now()}")
    
    try:
        log("=" * 50)
        log("🧠 INICIANDO AUTOENTRENAMIENTO")
        log("=" * 50)
        
        # 1. Verificar archivos necesarios
        archivos_necesarios = ["build_dataset.py", "train.py", "model.py", "tokenizer.py"]
        for archivo in archivos_necesarios:
            if not os.path.exists(archivo):
                log(f"❌ Falta archivo: {archivo}")
                return
        
        # 2. Verificar que hay datos
        if not os.path.exists(CHAT_LOGS):
            log("❌ No hay chat_logs.txt para entrenar")
            return
        
        # 3. Construir dataset (transferir de chat_logs.txt a data.txt)
        log("📦 Paso 1: Construyendo dataset...")
        try:
            resultado = subprocess.run(
                ["python", "build_dataset.py"],
                capture_output=True,
                text=True,
                check=True
            )
            log(f"✅ Dataset construido: {resultado.stdout[:100]}...")
        except subprocess.CalledProcessError as e:
            log(f"❌ Error construyendo dataset: {e}")
            log(f"Salida de error: {e.stderr}")
            return
        
        # 4. Verificar que data.txt tiene contenido
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8") as f:
                lineas_data = len(f.readlines())
            log(f"📊 data.txt tiene {lineas_data} líneas")
            
            if lineas_data < 10:
                log("⚠️ Poco contenido en data.txt, entrenamiento puede ser pobre")
        
        # 5. Entrenar modelo
        log("🚀 Paso 2: Entrenando modelo...")
        tiempo_inicio = time.time()
        
        try:
            resultado = subprocess.run(
                ["python", "train.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos máximo
            )
            
            tiempo_total = time.time() - tiempo_inicio
            
            if resultado.returncode == 0:
                log(f"✅ Entrenamiento exitoso en {tiempo_total:.1f} segundos")
                if "Loss:" in resultado.stdout:
                    log(f"📈 Salida del entrenamiento: {resultado.stdout[-200:]}")
            else:
                log(f"⚠️ Entrenamiento completado con código: {resultado.returncode}")
                log(f"Salida: {resultado.stdout[-500:]}")
                
        except subprocess.TimeoutExpired:
            log("⏰ Entrenamiento excedió el tiempo límite (5 minutos)")
        except Exception as e:
            log(f"❌ Error en entrenamiento: {e}")
        
        # 6. Limpiar y respaldar
        log("🧹 Paso 3: Limpiando logs...")
        if os.path.exists(CHAT_LOGS):
            # Crear respaldo
            fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backups/chat_logs_{fecha}.txt"
            
            # Crear directorio de backups si no existe
            os.makedirs("backups", exist_ok=True)
            
            import shutil
            shutil.copy2(CHAT_LOGS, backup_file)
            log(f"📁 Backup creado: {backup_file}")
            
            # Limpiar archivo actual (no borrar, solo vaciar)
            open(CHAT_LOGS, "w").close()
            log("✅ chat_logs.txt limpiado")
        
        log("=" * 50)
        log("🎉 AUTOENTRENAMIENTO COMPLETADO")
        log("=" * 50)
        
    except Exception as e:
        log(f"🔥 ERROR CRÍTICO: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # 🔓 Eliminar archivo de bloqueo
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            log("🔓 Lock file removido")

if __name__ == "__main__":
    print("🔍 Verificando si hay que entrenar...")
    if should_train():
        print("🚀 Iniciando entrenamiento...")
        auto_train()
    else:
        print("⏸️ No hay suficientes datos para entrenar")
        print(f"Revisa {CHAT_LOGS}")