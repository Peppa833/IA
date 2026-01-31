# limpiar_data.py
import re

def limpiar_data_txt():
    """Limpiar data.txt de conversaciones largas"""
    
    with open("data.txt", "r", encoding="utf-8") as f:
        lineas = f.readlines()
    
    nuevas_lineas = []
    
    for linea in lineas:
        linea = linea.strip()
        
        # Ignorar líneas vacías
        if not linea:
            continue
        
        # Ignorar líneas muy largas (>5 palabras)
        if len(linea.split()) > 6:
            print(f"❌ Ignorando línea larga: '{linea[:30]}...'")
            continue
        
        # Ignorar líneas que parecen concatenaciones
        if "como estas estoy bien gracias" in linea.lower():
            print(f"❌ Ignorando concatenación: '{linea[:40]}...'")
            continue
        
        # Mantener solo líneas cortas y naturales
        if 1 <= len(linea.split()) <= 6:
            nuevas_lineas.append(linea + "\n")
    
    # Guardar limpiado
    with open("data.txt", "w", encoding="utf-8") as f:
        f.writelines(nuevas_lineas)
    
    print(f"✅ data.txt limpiado: {len(nuevas_lineas)} líneas cortas")
    print(f"📝 Ejemplos: {nuevas_lineas[:5]}")

if __name__ == "__main__":
    limpiar_data_txt()