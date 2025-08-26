"""
Módulo de normalización de especies de frutas.
Consolida variaciones de nombres y estandariza en inglés para exportación.
"""

import pandas as pd

# ===================== DICCIONARIO DE NORMALIZACIÓN =====================
SPECIES_NORMALIZATION_MAP = {
    # 🍓 STRAWBERRIES (Consolidar 8 variaciones en 3)
    "Frutilla Sub": "Strawberry Grade B",
    "STRAWBERRIES": "Strawberry",
    "Frutilla": "Strawberry",
    "DICE STRAWBERRY": "Strawberry Diced",
    "SLICED STRAWBERRIES": "Strawberry Sliced",
    "STRAWBERRIES STEM IN": "Strawberry",  # Ajuste: igual que Strawberry normal
    "Frutilla Diced": "Strawberry Diced",
    
    # 🫐 BLUEBERRIES (Consolidar 3 variaciones en 2)
    "BLUEBERRIES": "Blueberry",
    "Arandano": "Blueberry",
    "Arandano Sub": "Blueberry Grade B",
    
    # 🍒 CHERRIES (Consolidar 4 variaciones en 1)
    "CHERRY BERRY": "Cherry",
    "CHERRIES": "Cherry",
    "Cereza": "Cherry",
    "CHERRY": "Cherry",
    
    # 🫐 RASPBERRIES (Consolidar 3 variaciones en 2)
    "RASPEBERRIES": "Raspberry",
    "Frambuesa": "Raspberry",
    "Frambuesa Sub": "Raspberry Grade B",
    
    # 🫐 BLACKBERRIES (Consolidar 2 variaciones en 1)
    "BLACKBERRIES": "Blackberry",
    "Mora": "Blackberry",
    
    # 🍇 GRAPES (Consolidar 2 variaciones en 1)
    "GRAPE": "Grape",
    "Uva": "Grape",
    
    # 🍍 PINEAPPLE (Consolidar 2 variaciones en 1)
    "Piña": "Pineapple",
    "PINEAPPLE": "Pineapple",
    
    # 🥭 MANGO
    "MANGO": "Mango",
    
    # 🍑 PEACHES
    "PEACHES": "Peach",
    
    # 🥝 EXOTIC FRUITS
    "PAPAYA": "Papaya",
    "DRAGON FRUIT": "Dragon Fruit",
    "CHIRIMOYA": "Cherimoya",
    "MARACUYA": "Passion Fruit",
    "LUCUMA": "Lucuma",
    "POMEGRANATE": "Pomegranate",
    
    # 🍌 BANANA
    "BANANA": "Banana",
    
    # 🍊 CITRUS
    "ORANGE": "Orange",
    "LEMON": "Lemon",
    "MELON": "Melon",
    
    # 🥑 AVOCADO
    "AVOCADO": "Avocado",
    
    # 🍇 MIXED BERRIES (Consolidar 4 variaciones en 1)
    "4 BERRIES": "Mixed Berries",
    "3 BERRIES": "Mixed Berries",
    "Mix Beries": "Mixed Berries",
    "MIX BERRIES": "Mixed Berries",

    # 🍇 MIXED FRUITS (Consolidar 3 variaciones en 1)
    "MIX FRUIT": "Mixed Fruits",
    "FRUIT MIX": "Mixed Fruits",
    "Mix Fruta": "Mixed Fruits",
}

# ===================== FUNCIONES DE NORMALIZACIÓN =====================

def normalize_species_name(species_name: str) -> str:
    """
    Normaliza el nombre de una especie según el mapeo predefinido.
    
    Args:
        species_name: Nombre original de la especie
        
    Returns:
        Nombre normalizado en inglés
    """
    if not species_name or pd.isna(species_name):
        return species_name
    
    # Convertir a string y limpiar
    species_str = str(species_name).strip()
    
    # Buscar en el mapeo de normalización
    if species_str in SPECIES_NORMALIZATION_MAP:
        return SPECIES_NORMALIZATION_MAP[species_str]
    
    # Si no está en el mapeo, retornar el nombre original
    return species_str

def normalize_species_column(df: pd.DataFrame, species_column: str = "Especie") -> pd.DataFrame:
    """
    Normaliza la columna de especies en un DataFrame.
    
    Args:
        df: DataFrame a normalizar
        species_column: Nombre de la columna de especies
        
    Returns:
        DataFrame con especies normalizadas
    """
    if species_column not in df.columns:
        return df
    
    df_normalized = df.copy()
    
    # Aplicar normalización
    df_normalized[species_column] = df_normalized[species_column].apply(normalize_species_name)
    
    return df_normalized