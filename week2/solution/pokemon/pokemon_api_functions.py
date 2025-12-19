import requests

BASE_URL = "https://pokeapi.co/api/v2"

SYSTEM_PROMPT ="""# System Prompt: Professor Eich (Pokémon API Agent)

## 1. Rolle und Persönlichkeit
Du bist **Professor Eich (Professor Oak)**, der renommierte Pokémon-Forscher aus Alabastia.
* Dein Ziel ist es, Trainern bei ihren Fragen zu helfen, indem du den **Pokédex** (die PokeAPI) konsultierst.
* Du bist hilfreich, enzyklopädisch und freundlich.
* Du antwortest **immer auf Deutsch**, egal in welcher Sprache die API-Daten vorliegen.

## 2. Deine Werkzeuge (The Tools)
Du hast Zugriff auf externe Python-Funktionen, um Live-Daten abzurufen. Rate niemals Stats oder Werte – **nutze immer die Tools.**

* `get_pokemon_details`: Für Kampfwerte (Stats), Typen, Gewicht, Größe.
* `get_species_info`: Für Entwicklungen (URL), Fangraten, Legendär-Status, Lore-Texte.
* `get_evolution_chain`: Um den Entwicklungsbaum zu sehen (benötigt ID aus `species_info`).
* `get_move_details`: Für Details zu Attacken (Stärke, Genauigkeit).
* `get_encounters`: Um Fundorte in der Wildnis zu finden.
* `get_nature_info`: Um Wesen (Natures) und deren Stat-Bonus/Malus zu erklären.
* `get_type_info`: Um Schwächen und Stärken (Matchups) zu finden.
* `get_pokemon_list_by_type`: Um Beispiele für einen bestimmten Typ zu finden.
* `get_ability_details`: Um komplexe Fähigkeiten zu erklären.

## 3. Der Übersetzungs-Prozess (Kritisch!)
Die API versteht nur **Englisch**. Der Nutzer spricht **Deutsch**. Du bist der Dolmetscher.

### Schritt A: Input (Vor dem Tool-Aufruf)
Wenn der Nutzer einen deutschen Begriff nennt, musst du ihn **intern ins Englische übersetzen**, bevor du das Tool aufrufst.
* *Nutzer:* "Wie stark ist **Glurak**?" -> *Tool:* `get_pokemon_details("charizard")`
* *Nutzer:* "Was macht das Wesen **Hart**?" -> *Tool:* `get_nature_info("adamant")`
* *Nutzer:* "Zeige mir **Unlicht** Pokémon." -> *Tool:* `get_pokemon_list_by_type("dark")`

**Wichtige Übersetzungs-Hilfe:**
* **Typen:** Feuer=Fire, Flug=Flying, Pflanze=Grass, Unlicht=Dark, Fee=Fairy, Kampf=Fighting, Psycho=Psychic, Geist=Ghost, Drache=Dragon, Käfer=Bug, Gestein=Rock, Boden=Ground, Stahl=Steel, Eis=Ice.
* **Wesen:** Hart=Adamant, Froh=Jolly, Mäßig=Modest, Scheu=Timid, Kühn=Bold, Pfiffig=Impish, Still=Calm, Sacht=Careful.
* **Items:** Überreste=Leftovers, Wahlband=Choice Band, Leben-Orb=Life Orb.

### Schritt B: Output (Nach dem Tool-Aufruf)
Die API liefert englisches JSON zurück. Übersetze die relevanten Felder im Antwortsatz zurück ins Deutsche.
* API: `{"type": "fire", "weak_against": ["water", "rock"]}`
* Antwort: "Es ist vom Typ **Feuer** und ist schwach gegen **Wasser** und **Gestein**."
* API Location: `viridian-forest` -> Antwort: "Vertania Wald".

## 4. Strategie für komplexe Fragen (Chain of Thought)
Wenn eine Antwort mehrere Schritte erfordert, plane selbstständig.

**Szenario: "Wie entwickle ich Evoli zu Nachtara?"**
1.  Ich brauche Entwicklungsdaten -> Rufe `get_species_info("eevee")`.
2.  Ich erhalte die `evolution_chain_url` -> Extrahiere ID -> Rufe `get_evolution_chain(ID)`.
3.  Ich analysiere den JSON-Baum nach "umbreon".
4.  Ich sehe `time_of_day: night` und `min_happiness`.
5.  **Antwort:** "Du musst Evoli bei **Nacht** trainieren, während es eine hohe **Freundschaft** zu dir hat."

## 5. Formatierung
* Nutze **Fettgedrucktes** für Pokémon-Namen, Orte und wichtige Werte.
* Nutze Aufzählungszeichen für Listen (z.B. bei Attacken oder Fundorten).
* Wenn Daten fehlen (z.B. API Error), entschuldige dich im Charakter ("Mein Pokédex liefert hierzu gerade keine Daten").

---
**Beginne nun die Interaktion.**
"""

def _clean_text(text):
    """Hilfsfunktion: Bereinigt Text von Zeilenumbrüchen."""
    return text.replace("\n", " ").replace("\f", " ") if text else ""

def get_pokemon_details(name):
    """Holt Kampfwerte, Typen und Fähigkeiten."""
    # Hinweis: Die API benötigt englische Namen (z.B. 'charizard' statt 'glurak').
    # Das LLM wird dies automatisch übersetzen.
    response = requests.get(f"{BASE_URL}/pokemon/{name.lower()}")
    if response.status_code != 200: return {"error": "Pokémon nicht gefunden"}
    data = response.json()
    
    return {
        "name": data['name'],
        "id": data['id'],
        "types": [t['type']['name'] for t in data['types']],
        "base_stats": {s['stat']['name']: s['base_stat'] for s in data['stats']},
        "abilities": [a['ability']['name'] for a in data['abilities']],
        "height": data['height'],
        "weight": data['weight']
    }

def get_species_info(name):
    """
    Holt 'Lore'-Infos: Pokédex-Einträge, Legendär-Status, Entwicklungs-URL.
    Wichtig für Fragen wie 'Wie entwickelt es sich?' oder 'Ist es legendär?'
    """
    response = requests.get(f"{BASE_URL}/pokemon-species/{name.lower()}")
    if response.status_code != 200: return {"error": "Spezies nicht gefunden"}
    data = response.json()
    
    # Wir holen englische Texte, da nicht alle Pokémon deutsche Einträge in der API haben.
    # Das LLM übersetzt diese dann.
    flavor_text = [
        _clean_text(entry['flavor_text']) 
        for entry in data['flavor_text_entries'] 
        if entry['language']['name'] == 'en'
    ]
    
    return {
        "is_legendary": data['is_legendary'],
        "is_mythical": data['is_mythical'],
        "base_happiness": data['base_happiness'],
        "capture_rate": data['capture_rate'],
        "evolution_chain_url": data['evolution_chain']['url'], 
        "pokedex_entries": list(set(flavor_text))[:3] 
    }

def get_evolution_chain(chain_id):
    """Holt den kompletten Entwicklungsbaum anhand der ID."""
    response = requests.get(f"{BASE_URL}/evolution-chain/{chain_id}")
    if response.status_code != 200: return {"error": "Entwicklungskette nicht gefunden"}
    data = response.json()
    return {"chain": data['chain']}

def get_move_details(name):
    """Holt Details zu einer Attacke (Stärke, Genauigkeit, AP)."""
    response = requests.get(f"{BASE_URL}/move/{name.lower().replace(' ', '-')}")
    if response.status_code != 200: return {"error": "Attacke nicht gefunden"}
    data = response.json()
    
    effect = next((e['effect'] for e in data['effect_entries'] if e['language']['name'] == 'en'), "Keine Beschreibung.")
    
    return {
        "name": data['name'],
        "type": data['type']['name'],
        "power": data['power'],
        "accuracy": data['accuracy'],
        "pp": data['pp'],
        "damage_class": data['damage_class']['name'], # physisch oder spezial
        "effect_description": _clean_text(effect)
    }

def get_type_info(name):
    """Holt Stärken und Schwächen eines Typs."""
    response = requests.get(f"{BASE_URL}/type/{name.lower()}")
    if response.status_code != 200: return {"error": "Typ nicht gefunden"}
    data = response.json()
    
    damage = data['damage_relations']
    return {
        "name": data['name'],
        "weak_against": [x['name'] for x in damage['double_damage_from']], # Schwach gegen
        "strong_against": [x['name'] for x in damage['double_damage_to']], # Stark gegen
        "immune_to": [x['name'] for x in damage['no_damage_from']] # Immun gegen
    }

def get_encounters(name):
    """
    Findet heraus, wo man ein Pokémon im Spiel fangen kann.
    """
    # Die API hat einen speziellen Endpunkt für Begegnungen
    response = requests.get(f"{BASE_URL}/pokemon/{name.lower()}/encounters")
    
    if response.status_code != 200: 
        return {"error": "Fundorte konnten nicht abgerufen werden."}
    
    data = response.json()
    
    if not data:
        return {"locations": [], "message": "Dieses Pokémon kann man in der Wildnis nicht fangen (oder es ist nur durch Entwicklung/Tausch erhältlich)."}

    # Wir nehmen die ersten 5 Fundorte, um den Chat nicht zu fluten
    locations = [loc['location_area']['name'].replace("-", " ") for loc in data[:5]]
    
    return {
        "pokemon": name,
        "locations": locations,
        "total_locations_found": len(data)
    }

def get_nature_info(name):
    """
    Erklärt ein 'Wesen' (Nature). Wichtig für Strategie, da Wesen Werte beeinflussen.
    Beispiel: 'Adamant' (Hart) erhöht Angriff, senkt Spezial-Angriff.
    """
    response = requests.get(f"{BASE_URL}/nature/{name.lower()}")
    
    if response.status_code != 200: 
        return {"error": "Wesen nicht gefunden."}
    
    data = response.json()
    
    return {
        "name": data['name'],
        "increased_stat": data['increased_stat']['name'] if data['increased_stat'] else "None",
        "decreased_stat": data['decreased_stat']['name'] if data['decreased_stat'] else "None",
        "flavor_profile": data['likes_flavor']['name'] if data['likes_flavor'] else "None" # Welches Essen es mag
    }

def get_pokemon_list_by_type(type_name, limit=10):
    """
    Listet Pokémon eines bestimmten Typs auf.
    Nützlich für Fragen wie: "Nenne mir 5 Feuer-Pokémon".
    """
    response = requests.get(f"{BASE_URL}/type/{type_name.lower()}")
    
    if response.status_code != 200: 
        return {"error": "Typ nicht gefunden."}
    
    data = response.json()
    pokemon_list = [p['pokemon']['name'] for p in data['pokemon']]
    
    # Wir geben nur eine Auswahl zurück, da manche Typen 100+ Pokémon haben
    import random
    selected_pokemon = random.sample(pokemon_list, min(len(pokemon_list), limit))
    
    return {
        "type": type_name,
        "pokemon_examples": selected_pokemon,
        "total_count": len(pokemon_list)
    }

def get_ability_details(name):
    """
    Detaillierte Infos zu einer Fähigkeit (Ability).
    Wird oft benötigt, wenn 'get_pokemon_details' nur den Namen der Fähigkeit liefert.
    """
    clean_name = name.lower().replace(" ", "-")
    response = requests.get(f"{BASE_URL}/ability/{clean_name}")
    
    if response.status_code != 200: 
        return {"error": "Fähigkeit nicht gefunden."}
    
    data = response.json()
    
    # Suche den englischen Erklärungstext
    effect = next((e['effect'] for e in data['effect_entries'] if e['language']['name'] == 'en'), "Keine Beschreibung verfügbar.")
    
    return {
        "name": data['name'],
        "effect": _clean_text(effect),
        "pokemon_candidates": [p['pokemon']['name'] for p in data['pokemon'][:5]] # Welche Pokémon haben das?
    }

TOOLS = [
  {
    "name": "get_pokemon_details",
    "description": "Ruft technische Daten für ein Pokémon ab: Basiswerte (Angriff, Init), Typen, Größe, Gewicht und Fähigkeiten. Nutze dies für allgemeine Stats-Fragen.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name des Pokémon (z.B. 'charizard' für Glurak). Das LLM muss den deutschen Namen intern übersetzen." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_species_info",
    "description": "Ruft Hintergrundinformationen ab: Pokédex-Einträge, Fangrate, ob es legendär ist, und die 'evolution_chain_url'. Nutze dies für Fragen zur Entwicklung oder zum Verhalten.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name des Pokémon." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_evolution_chain",
    "description": "Ruft den kompletten Entwicklungsbaum ab. Benötigt eine 'chain_id' (Integer), die man vorher über 'get_species_info' herausfindet.",
    "parameters": {
      "type": "object",
      "properties": {
        "chain_id": { "type": "integer", "description": "Die numerische ID aus der evolution_chain URL." }
      },
      "required": ["chain_id"]
    }
  },
  {
    "name": "get_move_details",
    "description": "Ruft Daten zu einer Attacke ab: Stärke, Genauigkeit, AP und Schadensklasse (physisch/spezial).",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name der Attacke (z.B. 'fireball')." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_type_info",
    "description": "Ruft Typen-Effektivität ab. Gibt Listen zurück, wogegen dieser Typ schwach oder stark ist.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name des Elementar-Typs (z.B. 'electric')." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_encounters",
    "description": "Findet Orte (Routen, Höhlen, Gebiete), an denen man ein bestimmtes Pokémon wild fangen kann.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name des Pokémon (z.B. 'pikachu')." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_nature_info",
    "description": "Ruft Details zu einem Wesen (Nature) ab. Zeigt, welcher Statuswert erhöht und welcher gesenkt wird. Wichtig für strategische Fragen.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name des Wesens (z.B. 'adamant' für Hart, 'jolly' für Froh)." }
      },
      "required": ["name"]
    }
  },
  {
    "name": "get_pokemon_list_by_type",
    "description": "Gibt eine Liste von Pokémon zurück, die einen bestimmten Elementar-Typ haben. Nutze dies, wenn der Nutzer nach Beispielen für einen Typ fragt.",
    "parameters": {
      "type": "object",
      "properties": {
        "type_name": { "type": "string", "description": "Der englische Name des Typs (z.B. 'fire', 'dragon')." },
        "limit": { "type": "integer", "description": "Wie viele Beispiele zurückgegeben werden sollen (Standard: 10)." }
      },
      "required": ["type_name"]
    }
  },
  {
    "name": "get_ability_details",
    "description": "Erklärt genau, was eine passive Fähigkeit (Ability) im Kampf bewirkt und welche Pokémon sie haben können.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string", "description": "Der englische Name der Fähigkeit (z.B. 'static', 'levitate')." }
      },
      "required": ["name"]
    }
  }
]