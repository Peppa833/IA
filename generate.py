# generate.py - VERSION CORREGIDA CON CARGA MEJORADA
import torch
import os
import random
from model import NeuralChat

# Verificar modelo
if not os.path.exists("model.pth"):
    print("Modelo no encontrado. Ejecuta: python train.py")
    exit(1)

# CARGA CORREGIDA DEL MODELO
try:
    checkpoint = torch.load("model.pth", map_location="cpu", weights_only=False)
    
    # Formato nuevo (diccionario)
    if isinstance(checkpoint, dict):
        model_state = checkpoint['model_state_dict']
        stoi = checkpoint['stoi']
        itos = checkpoint['itos']
    # Formato antiguo (tupla)
    elif isinstance(checkpoint, tuple) and len(checkpoint) == 3:
        model_state, stoi, itos = checkpoint
    else:
        raise ValueError("Formato de modelo no reconocido")
    
    model = NeuralChat(len(stoi))
    model.load_state_dict(model_state)
    model.eval()
    print("Modelo cargado correctamente")
    
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit(1)

def generar(seed, max_palabras=8):
    """Generar respuesta CORTA y COHERENTE"""
    
    if not seed or len(seed.strip()) == 0:
        return "Hola, ¿cómo estás?"
    
    # Limpiar y tokenizar
    seed = seed.lower().strip()
    palabras = seed.split()
    
    # CORRECCIÓN: Manejar palabras OOV (Out of Vocabulary) con índice 0
    ids = [stoi.get(p, 0) for p in palabras]  # Usar get() en lugar de verificación directa
    
    # CORRECCIÓN: Verificar si hay suficientes palabras conocidas
    palabras_conocidas = sum(1 for id in ids if id != 0)
    
    if not ids or palabras_conocidas < max(1, len(palabras) * 0.3):  # Al menos 30% conocidas
        return random.choice([
            "No entiendo completamente",
            "¿Puedes explicar mejor?",
            "Interesante pregunta",
            "Hablemos de otra cosa"
        ])
    
    # CORRECCIÓN: Filtrar IDs cero (palabras desconocidas)
    ids = [id for id in ids if id != 0]
    
    # Tensor de entrada
    x = torch.tensor([ids], dtype=torch.long)
    
    # Generar con límite estricto
    resultado = []
    palabras_usadas = set()
    
    for i in range(max_palabras):
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out[0, -1], dim=0)
            
            # Penalizar palabras repetidas
            for idx, palabra in itos.items():
                if palabra in palabras_usadas:
                    probs[idx] *= 0.2  # Reducir 80%
            
            # Añadir creatividad
            probs = probs ** 0.7  # Temperatura
            probs = probs / probs.sum()
            
            # Escoger siguiente palabra
            next_id = torch.multinomial(probs, 1).item()
        
        palabra = itos.get(next_id, "")
        
        # CORRECCIÓN: Condiciones de parada mejoradas
        stop_conditions = [
            not palabra,
            palabra in [".", "!", "?", "fin", "adiós", "adios", "bye", "chao", "luego", "stop", "parar"],
            i >= 5 and len(resultado) >= 3,  # Parar después de 3-5 palabras
            palabra in resultado[-2:] if resultado else False,  # No repetir
            len(palabras_usadas) >= 7  # No más de 7 palabras únicas
        ]
        
        if any(stop_conditions):
            break
        
        resultado.append(palabra)
        palabras_usadas.add(palabra)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
    
    # Formatear respuesta
    if resultado:
        respuesta = " ".join(resultado)
        respuesta = respuesta.capitalize()
        
        # Asegurar que termine con punto
        if not respuesta.endswith((".", "!", "?")):
            respuesta += "."
        
        # LIMITAR LONGITUD (máximo 8 palabras)
        palabras_respuesta = respuesta.split()
        if len(palabras_respuesta) > 8:
            respuesta = " ".join(palabras_respuesta[:8]) + "."
        
        # CORRECCIÓN: Asegurar que la respuesta no sea demasiado corta
        if len(palabras_respuesta) < 2 and respuesta.endswith("."):
            # Si solo hay una palabra, añadir algo más
            opciones_extra = ["bien", "gracias", "si", "no", "tal vez", "claro"]
            extra = random.choice(opciones_extra)
            respuesta = respuesta.replace(".", f" {extra}.")
        
        return respuesta
    else:
        # CORRECCIÓN: Respuestas de fallback mejoradas
        fallback_responses = [
            "No sé qué decir sobre eso.",
            "Podrías reformular la pregunta?",
            "Eso es interesante, dime más.",
            "No estoy seguro de cómo responder.",
            "Hablemos de otra cosa."
        ]
        return random.choice(fallback_responses)

# Prueba mejorada
if __name__ == "__main__":
    print("Probando generador CORREGIDO:")
    print("=" * 50)
    
    tests = [
        "hola",
        "como estas", 
        "que haces",
        "adiós",
        "nombre tu cual es",  # Test con palabras posiblemente desconocidas
        "",  # Test con entrada vacía
        "esta es una frase muy larga para probar el limite de palabras"  # Test largo
    ]
    
    for test in tests:
        print(f"Entrada: '{test}'")
        resp = generar(test)
        palabras_resp = len(resp.split())
        print(f"Respuesta: '{resp}' ({palabras_resp} palabras)")
        print("-" * 30)
    
    # Test adicional: verificar que funciona con palabras desconocidas
    print("\nTest con palabras fuera del vocabulario:")
    test_oov = "supercalifragilistico espialidoso"  # Palabras que probablemente no están en el vocabulario
    resp_oov = generar(test_oov)
    print(f"Entrada OOV: '{test_oov}'")
    print(f"Respuesta: '{resp_oov}'")
    print("=" * 50)