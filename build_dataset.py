def build_dataset():
    with open("data.txt", "a", encoding="utf-8") as base, \
         open("chat_logs.txt", "r", encoding="utf-8") as logs:
        
        lines = [l.strip() for l in logs if l.strip()]  # Filtrar líneas vacías
        
        i = 0
        while i < len(lines) - 1:
            if lines[i].startswith("usuario:") and lines[i + 1].startswith("ia:"):
                pregunta = lines[i].replace("usuario:", "").strip()
                respuesta = lines[i + 1].replace("ia:", "").strip()
                
                if pregunta and respuesta:  # Solo si no están vacías
                    base.write(pregunta + "\n")
                    base.write(respuesta + "\n")
                    i += 2
                else:
                    i += 1
            else:
                i += 1  # Saltar línea mal formada