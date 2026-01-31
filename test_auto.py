# test_auto.py - Para probar el autoentrenamiento fácilmente
import os
import auto_train

# 1. Limpiar archivos viejos
if os.path.exists("training.lock"):
    os.remove("training.lock")

# 2. Crear datos de prueba en chat_logs.txt
with open("chat_logs.txt", "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(f"usuario: mensaje de prueba {i}\n")
        f.write(f"ia: respuesta de prueba {i}\n")

print("📝 Datos de prueba creados en chat_logs.txt")

# 3. Verificar si debe entrenar
print("\n🔍 Verificando should_train():")
debe = auto_train.should_train()
print(f"Debe entrenar: {debe}")

# 4. Ejecutar autoentrenamiento
if debe:
    print("\n🚀 Ejecutando auto_train()...")
    auto_train.auto_train()
else:
    print("\n❌ No hay suficientes datos (algo anda mal)")