import torch
import time
from model import NeuralChat
from tokenizer import tokenize, build_vocab

print("🔧 ENTRENAMIENTO CORREGIDO - Pares Pregunta-Respuesta")
print("=" * 60)

# Cargar datos con estructura pregunta\trespuesta
try:
    with open("data.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Separar en pares (líneas alternas: pregunta, respuesta, pregunta, respuesta)
    preguntas = []
    respuestas = []
    
    for i in range(0, len(lines) - 1, 2):
        pregunta = lines[i]
        respuesta = lines[i + 1] if i + 1 < len(lines) else ""
        
        if pregunta and respuesta:
            preguntas.append(pregunta)
            respuestas.append(respuesta)
    
    print(f"📊 Pares de entrenamiento: {len(preguntas)}")
    
    if len(preguntas) < 3:
        print("❌ Error: Necesitas al menos 3 pares de conversación")
        print("   Formato en data.txt debe ser:")
        print("   línea 1: pregunta")
        print("   línea 2: respuesta")
        print("   línea 3: pregunta")
        print("   línea 4: respuesta")
        exit(1)
    
    # Construir vocabulario de TODAS las palabras
    all_text = " ".join(preguntas + respuestas)
    stoi, itos = build_vocab(all_text)
    print(f"📖 Vocabulario: {len(stoi)} palabras únicas")
    
    # Preparar datos de entrenamiento
    X_data = []
    Y_data = []
    
    for pregunta, respuesta in zip(preguntas, respuestas):
        # Tokenizar pregunta y respuesta
        pregunta_tokens = tokenize(pregunta)
        respuesta_tokens = tokenize(respuesta)
        
        # Convertir a IDs
        pregunta_ids = [stoi.get(t, 0) for t in pregunta_tokens]
        respuesta_ids = [stoi.get(t, 0) for t in respuesta_tokens]
        
        # Crear secuencia: pregunta + respuesta
        # El modelo aprenderá: dada la pregunta, predecir la respuesta
        secuencia_completa = pregunta_ids + respuesta_ids
        
        if len(secuencia_completa) > 1:
            X_data.append(secuencia_completa[:-1])
            Y_data.append(secuencia_completa[1:])
    
    print(f"✅ Secuencias de entrenamiento: {len(X_data)}")
    
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    exit(1)

# Crear modelo
model = NeuralChat(len(stoi))

# Intentar cargar modelo existente
try:
    checkpoint = torch.load("model.pth", map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'vocab_size' in checkpoint and checkpoint['vocab_size'] == len(stoi):
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Modelo anterior cargado (vocabulario compatible)")
        else:
            print(f"⚠️ Vocabulario cambió, entrenando desde cero")
    else:
        print("🧪 Formato antiguo, entrenando desde cero")
        
except FileNotFoundError:
    print("🧪 Entrenando desde cero (no hay modelo previo)")
except Exception as e:
    print(f"⚠️ Error: {e}, entrenando desde cero")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\n🚀 Comenzando entrenamiento...")
print("=" * 60)
start_time = time.time()

# Entrenar con cada par pregunta-respuesta
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    
    for X_seq, Y_seq in zip(X_data, Y_data):
        optimizer.zero_grad()
        
        # Convertir a tensores
        X = torch.tensor([X_seq], dtype=torch.long)
        Y = torch.tensor([Y_seq], dtype=torch.long)
        
        # Forward pass
        out = model(X)
        
        # Calcular loss
        loss = loss_fn(out.view(-1, len(stoi)), Y.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(X_data)
    
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"📈 Época {epoch}/{epochs}: Loss = {avg_loss:.4f}")
        
        # Mostrar ejemplo de generación
        if preguntas:
            test_pregunta = preguntas[0]  # "hola"
            test_tokens = tokenize(test_pregunta)
            test_ids = [stoi.get(t, 0) for t in test_tokens]
            
            with torch.no_grad():
                test_input = torch.tensor([test_ids], dtype=torch.long)
                
                # Generar respuesta (máximo 5 palabras)
                generated = []
                for _ in range(5):
                    out_test = model(test_input)
                    probs = torch.softmax(out_test[0, -1], dim=0)
                    next_id = torch.multinomial(probs, 1).item()
                    
                    palabra = itos.get(next_id, "")
                    if not palabra or palabra in [".", "!", "?"]:
                        break
                    
                    generated.append(palabra)
                    test_input = torch.cat([test_input, torch.tensor([[next_id]])], dim=1)
                
                print(f"   📝 '{test_pregunta}' → '{' '.join(generated)}'")

# Guardar modelo
try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(stoi)
    }, "model.pth")
    print("\n✅ Modelo guardado correctamente en model.pth")
except Exception as e:
    print(f"\n❌ Error guardando modelo: {e}")

tiempo_total = time.time() - start_time
print(f"✅ Entrenamiento completado en {tiempo_total:.1f} segundos")
print(f"🔤 Vocabulario: {len(stoi)} palabras")
print(f"💾 Pares entrenados: {len(preguntas)}")
print("=" * 60)