"""
Unified PokéAPI v2 client with local caching and specific data retrieval functions.
Merged from pokemon_api_client.py and pokemon_api_functions.py.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import random

import requests


class PokemonAPIClient:
    """
    Generic client for PokéAPI v2 with built-in local caching and specific helper methods.

    Fair Use Policy:
    - Use local caching to avoid unnecessary requests.
    - Do not spam the API with high-frequency polling.
    - Handle errors gracefully.
    """

    BASE_URL = "https://pokeapi.co/api/v2"

    # Resource endpoints that support ID/name lookup
    NAMED_RESOURCES = {
        "ability",
        "berry",
        "berry-firmness",
        "berry-flavor",
        "characteristic",
        "contest-effect",
        "contest-type",
        "egg-group",
        "encounter-condition",
        "encounter-condition-value",
        "encounter-method",
        "evolution-chain",
        "evolution-trigger",
        "gender",
        "generation",
        "growth-rate",
        "item",
        "item-attribute",
        "item-category",
        "item-fusing",
        "item-pocket",
        "language",
        "location",
        "location-area",
        "machine",
        "move",
        "move-ailment",
        "move-battle-style",
        "move-category",
        "move-damage-class",
        "move-learn-method",
        "move-target",
        "nature",
        "pal-park-area",
        "pokeathlon-stat",
        "pokedex",
        "pokemon",
        "pokemon-color",
        "pokemon-form",
        "pokemon-habitat",
        "pokemon-shape",
        "pokemon-species",
        "region",
        "stat",
        "super-contest-effect",
        "type",
        "version",
        "version-group",
        "evolution-trigger",
        "item-category",
        "item-attribute",
    }

    def __init__(self, cache_dir: Union[str, Path] = ".pokemon_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache expires after 24 hours by default
        self.cache_ttl = 86400

    def _get_url(self, endpoint: str, identifier: Union[str, int, None] = None) -> str:
        url = f"{self.BASE_URL}/{endpoint}"
        if identifier is not None:
            url = f"{url}/{identifier}"
        return url

    def _get_cache_path(self, endpoint: str, identifier: Union[str, int, None]) -> Path:
        if identifier is not None:
            safe_id = str(identifier).replace(" ", "_").lower()
            filename = f"{endpoint}_{safe_id}.json"
        else:
            filename = f"{endpoint}_list.json"
        return self.cache_dir / filename

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.cache_ttl == 0:
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (datetime.now() - mtime).total_seconds() < self.cache_ttl

    def _load_from_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if self._is_cache_valid(path):
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _save_to_cache(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Cache write failure should not break main logic
            pass

    def _get(
        self, endpoint: str, identifier: Union[str, int, None] = None
    ) -> Dict[str, Any]:
        """
        Generic GET request with caching.
        """
        cache_path = self._get_cache_path(endpoint, identifier)
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data

        url = self._get_url(endpoint, identifier)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        self._save_to_cache(cache_path, data)
        return data

    def _clean_text(self, text: str) -> str:
        """Removes newlines and form feeds from text."""
        return text.replace("\n", " ").replace("\f", " ")

    def get_pokemon_details(self, name: str) -> Dict[str, Any]:
        """
        Ruft technische Daten für ein Pokémon ab.
        """
        try:
            data = self._get("pokemon", name.lower())

            stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
            types = [t["type"]["name"] for t in data["types"]]
            # Convert decimetres to meters
            height_m = data["height"] / 10
            # Convert hectograms to kg
            weight_kg = data["weight"] / 10
            abilities = [
                a["ability"]["name"] for a in data["abilities"] if not a["is_hidden"]
            ]

            return {
                "name": data["name"],
                "stats": stats,
                "types": types,
                "height_m": height_m,
                "weight_kg": weight_kg,
                "abilities": abilities,
                "base_experience": data.get("base_experience"),
                "sprites": {
                    "front_default": data["sprites"].get("front_default"),
                    "back_default": data["sprites"].get("back_default"),
                },
            }
        except requests.exceptions.RequestException:
            return {"error": f"Pokémon '{name}' nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_species_info(self, name: str) -> Dict[str, Any]:
        """
        Ruft Hintergrundinformationen (Spezies) ab.
        """
        try:
            # Get basic info first to get species URL if needed, but here we can try querying species directly
            # assuming name matches. Sometimes species name != pokemon name (e.g. varieties),
            # but usually it works for base forms.
            data = self._get("pokemon-species", name.lower())

            # Find German flavor text, fallback to English
            flavor_text = "Keine Beschreibung verfügbar."
            for entry in data["flavor_text_entries"]:
                if entry["language"]["name"] == "de":
                    flavor_text = self._clean_text(entry["flavor_text"])
                    break
                elif entry["language"]["name"] == "en":  # Fallback
                    flavor_text = self._clean_text(entry["flavor_text"])

            is_legendary = data["is_legendary"] or data["is_mythical"]
            genus = (
                next(
                    (
                        g["genus"]
                        for g in data["generap"]
                        if g["language"]["name"] == "en"
                    ),
                    "Keine Kategorie",
                )
                if "generap" in data
                else "Keine Kategorie"
            )

            # Correct key is 'genera' not 'generap', fixing typo from previous thought process if any.
            # Actually poking API key is 'genera'.
            genus = "Unbekannt"
            for g in data.get("genera", []):
                if g["language"]["name"] in [
                    "de",
                    "en",
                ]:  # Prefer German if available, else English?
                    # Usually "Seed Pokémon" is English. German is "Samen-Pokémon".
                    # Let's try to find German first if this is for German user.
                    # But prompt says "translate relevant fields".
                    # Let's stick to English as fallback, but since we want German output eventually:
                    if g["language"]["name"] == "de":
                        genus = g["genus"]
                        break
                    elif g["language"]["name"] == "en":
                        genus = g["genus"]

            return {
                "name": data["name"],
                "flavor_text": flavor_text,
                "is_legendary": is_legendary,
                "capture_rate": data["capture_rate"],
                "evolution_chain_url": data["evolution_chain"]["url"],
                "genus": genus,
                "generation": data.get("generation", {}).get("name"),
            }
        except requests.exceptions.RequestException:
            return {"error": "Spezies-Info nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_evolution_chain(self, chain_id: int) -> Dict[str, Any]:
        """
        Ruft den Entwicklungsbaum ab.
        """
        try:
            # chain_id must be extracted from URL by caller usually, but logic here takes ID directly
            data = self._get("evolution-chain", chain_id)

            # Recursively parse the chain
            def parse_evolution(node):
                species_name = node["species"]["name"]
                evo_details = []

                # Check evolution details for this node (how it evolved from previous)
                for detail in node.get("evolution_details", []):
                    # Extract only relevant non-null/false triggers
                    conditions = {}
                    if detail.get("trigger"):
                        conditions["trigger"] = detail["trigger"]["name"]
                    if detail.get("item"):
                        conditions["item"] = detail["item"]["name"]
                    if detail.get("min_level"):
                        conditions["min_level"] = detail["min_level"]
                    if detail.get("min_happiness"):
                        conditions["min_happiness"] = detail["min_happiness"]
                    if detail.get("time_of_day"):
                        conditions["time_of_day"] = detail["time_of_day"]
                    if detail.get("held_item"):
                        conditions["held_item"] = detail["held_item"]["name"]
                    if detail.get("known_move"):
                        conditions["known_move"] = detail["known_move"]["name"]
                    if detail.get("known_move_type"):
                        conditions["known_move_type"] = detail["known_move_type"][
                            "name"
                        ]
                    if detail.get("location"):
                        conditions["location"] = detail["location"]["name"]
                    evo_details.append(conditions)

                result = {
                    "species_name": species_name,
                    "evolution_details": evo_details,
                }

                if node.get("evolves_to"):
                    result["evolves_to"] = [
                        parse_evolution(sub_node) for sub_node in node["evolves_to"]
                    ]
                else:
                    result["evolves_to"] = []

                return result

            chain_data = parse_evolution(data["chain"])

            return {"chain_id": data["id"], "chain": chain_data}
        except requests.exceptions.RequestException:
            return {"error": "Entwicklungslinie nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_move_details(self, name: str) -> Dict[str, Any]:
        """
        Ruft Details zu einer Attacke ab.
        """
        try:
            data = self._get("move", name.lower())

            effect = "Keine Beschreibung."
            if data["effect_entries"]:
                effect = data["effect_entries"][0]["effect"]

            return {
                "name": data["name"],
                "type": data["type"]["name"],
                "power": data["power"],
                "accuracy": data["accuracy"],
                "pp": data["pp"],
                "damage_class": data["damage_class"]["name"],
                "effect_description": self._clean_text(effect),
                "priority": data.get("priority"),
                "target": data.get("target", {}).get("name"),
            }
        except requests.exceptions.RequestException:
            return {"error": "Attacke nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_type_info(self, name: str) -> Dict[str, Any]:
        """
        Holt Stärken und Schwächen eines Typs.
        """
        try:
            data = self._get("type", name.lower())

            damage = data["damage_relations"]
            return {
                "name": data["name"],
                "weak_against": [x["name"] for x in damage["double_damage_from"]],
                "strong_against": [x["name"] for x in damage["double_damage_to"]],
                "immune_to": [x["name"] for x in damage["no_damage_from"]],
            }
        except requests.exceptions.RequestException:
            return {"error": "Typ nicht gefunden"}
        except Exception as e:
            return {"error": str(e)}

    def get_encounters(self, name: str) -> Dict[str, Any]:
        """
        Findet heraus, wo man ein Pokémon im Spiel fangen kann.
        """
        try:
            # Special endpoint: pokemon/{id}/encounters
            # This is NOT a standard named resource lookup, so we might need a custom handling
            # or just use _get with a constructed string if identifiers can be paths
            # BUT _get assumes resource/identifier structure.
            # Let's manually construct to use our caching with a trick or just implement manually with cache

            endpoint = "pokemon"
            identifier = f"{name.lower()}/encounters"

            # _get works if identifier is passed as the subpath
            # But the caching naming might get weird: pokemon_pikachu/encounters.json -> might fail on filesystem
            # So we better custom implement with safe cache key

            # Custom cache logic for encounters
            safe_name = name.lower().replace(" ", "_")
            cache_path = self.cache_dir / f"encounters_{safe_name}.json"

            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data

            url = f"{self.BASE_URL}/pokemon/{name.lower()}/encounters"
            response = requests.get(url)

            if response.status_code != 200:
                return {"error": "Fundorte konnten nicht abgerufen werden."}

            data = response.json()

            # Cache the raw list
            self._save_to_cache(cache_path, data)

            # Process data
            if not data:
                return {
                    "locations": [],
                    "message": "Dieses Pokémon kann man in der Wildnis nicht fangen (oder es ist nur durch Entwicklung/Tausch erhältlich).",
                }

            locations = [
                loc["location_area"]["name"].replace("-", " ") for loc in data[:5]
            ]

            return {
                "pokemon": name,
                "locations": locations,
                "total_locations_found": len(data),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_nature_info(self, name: str) -> Dict[str, Any]:
        """
        Erklärt ein 'Wesen' (Nature).
        """
        try:
            data = self._get("nature", name.lower())

            return {
                "name": data["name"],
                "increased_stat": data["increased_stat"]["name"]
                if data["increased_stat"]
                else "None",
                "decreased_stat": data["decreased_stat"]["name"]
                if data["decreased_stat"]
                else "None",
                "flavor_profile": data["likes_flavor"]["name"]
                if data["likes_flavor"]
                else "None",
            }
        except requests.exceptions.RequestException:
            return {"error": "Wesen nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_pokemon_list_by_type(
        self, type_name: str, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Listet Pokémon eines bestimmten Typs auf.
        """
        try:
            data = self._get("type", type_name.lower())

            pokemon_list = [p["pokemon"]["name"] for p in data["pokemon"]]

            selected_pokemon = random.sample(
                pokemon_list, min(len(pokemon_list), limit)
            )

            return {
                "type": type_name,
                "pokemon_examples": selected_pokemon,
                "total_count": len(pokemon_list),
            }
        except requests.exceptions.RequestException:
            return {"error": "Typ nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_ability_details(self, name: str) -> Dict[str, Any]:
        """
        Detaillierte Infos zu einer Fähigkeit (Ability).
        """
        try:
            clean_name = name.lower().replace(" ", "-")
            data = self._get("ability", clean_name)

            effect = "Keine Beschreibung verfügbar."
            for entry in data["effect_entries"]:
                if entry["language"]["name"] == "en":
                    effect = entry["effect"]
                    break

            pokemon_candidates = [p["pokemon"]["name"] for p in data["pokemon"][:5]]

            return {
                "name": data["name"],
                "effect": self._clean_text(effect),
                "pokemon_candidates": pokemon_candidates,
            }
        except requests.exceptions.RequestException:
            return {"error": "Fähigkeit nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_item_info(self, name: str) -> Dict[str, Any]:
        """
        Ruft Infos zu einem Item ab.
        """
        try:
            clean_name = name.lower().replace(" ", "-")
            data = self._get("item", clean_name)

            effect = "Keine Beschreibung verfügbar."
            for entry in data["effect_entries"]:
                if entry["language"]["name"] == "en":
                    effect = entry["effect"]
                    break

            return {
                "name": data["name"],
                "cost": data["cost"],
                "category": data["category"]["name"],
                "effect": self._clean_text(effect),
                "attributes": [a["name"] for a in data.get("attributes", [])],
                "fling_power": data.get("fling_power"),
                "fling_effect": data.get("fling_effect", {}).get("name")
                if data.get("fling_effect")
                else None,
            }
        except requests.exceptions.RequestException:
            return {"error": "Item nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_evolution_trigger_info(self, name: str) -> Dict[str, Any]:
        """
        Ruft Infos zu einem Entwicklungsauslöser ab.
        """
        try:
            data = self._get("evolution-trigger", name.lower())
            return {
                "name": data["name"],
                "pokemon_species": [s["name"] for s in data["pokemon_species"]],
            }
        except requests.exceptions.RequestException:
            return {"error": "Entwicklungsauslöser nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_item_category_info(self, name: str) -> Dict[str, Any]:
        """
        Ruft Infos zu einer Item-Kategorie ab.
        """
        try:
            data = self._get("item-category", name.lower())
            return {
                "name": data["name"],
                "items": [i["name"] for i in data["items"]],
            }
        except requests.exceptions.RequestException:
            return {"error": "Item-Kategorie nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_item_attribute_info(self, name: str) -> Dict[str, Any]:
        """
        Ruft Infos zu einem Item-Attribut ab.
        """
        try:
            data = self._get("item-attribute", name.lower())

            desc = "Keine Beschreibung verfügbar."
            for d in data.get("descriptions", []):
                if d["language"]["name"] == "en":
                    desc = d["description"]
                    break

            return {
                "name": data["name"],
                "description": self._clean_text(desc),
                "items": [i["name"] for i in data["items"]],
            }
        except requests.exceptions.RequestException:
            return {"error": "Item-Attribut nicht gefunden."}
        except Exception as e:
            return {"error": str(e)}

    def get_system_prompt(self) -> str:
        return """# System Prompt: Professor Eich (Pokémon API Agent)

## 1. Rolle und Persönlichkeit
Du bist **Professor Eich (Professor Oak)**, der renommierte Pokémon-Forscher aus Alabastia.
* Dein Ziel ist es, Trainern bei ihren Fragen zu helfen, indem du den **Pokédex** (die PokeAPI) konsultierst.
* Du bist hilfreich, enzyklopädisch und freundlich.
* Du antwortest **immer auf Deutsch**, egal in welcher Sprache die API-Daten vorliegen.

## 2. Deine Werkzeuge (The Tools)
Du hast Zugriff auf externe Python-Funktionen, um Live-Daten abzurufen. Rate **niemals** Stats, Werte oder andere Details – **nutze immer die Tools.** Die Tools sind von entscheidender Bedeutung für korrekte Antworten.

* `get_pokemon_details`: Für Kampfwerte, Sprites, Base Exp.
* `get_species_info`: Für Entwicklungs-URL, Genus, Generation, Fangraten, Lore.
* `get_evolution_chain`: Für präzise Entwicklungs-Bedingungen (Level, Item, etc.).
* `get_evolution_trigger_info`: Für Infos zu Tausch/Level-up Auslösern.
* `get_move_details`: Für Details zu Attacken (Stärke, Prio, Ziel).
* `get_encounters`: Um Fundorte in der Wildnis zu finden.
* `get_nature_info`: Um Wesen (Natures) und deren Stat-Bonus/Malus zu erklären.
* `get_type_info`: Um Schwächen und Stärken (Matchups) zu finden.
* `get_pokemon_list_by_type`: Um Beispiele für einen bestimmten Typ zu finden.
* `get_ability_details`: Um komplexe Fähigkeiten zu erklären.
* `get_item_info`: Für Item-Details (Attribute, Fling, usw.)
* `get_item_category_info`: Wenn nach Kategorien von Items gefragt wird (z.B. Pokébälle).
* `get_item_attribute_info`: Wenn nach Item-Eigenschaften gefragt wird (z.B. consumable).

## 3. Der Übersetzungs-Prozess (Kritisch!)
Die API versteht nur **Englisch**. Der Nutzer spricht **Deutsch**. Du bist der Dolmetscher.

### Schritt A: Input (Vor dem Tool-Aufruf)
Wenn der Nutzer einen deutschen Begriff nennt, musst du ihn **intern ins Englische übersetzen**, bevor du das Tool aufrufst.
* *Nutzer:* "Wie stark ist **Glurak**?" -> *Tool:* `get_pokemon_details("charizard")`
* *Nutzer:* "Was macht das Wesen **Hart**?" -> *Tool:* `get_nature_info("adamant")`
* *Nutzer:* "Zeige mir **Unlicht** Pokémon." -> *Tool:* `get_pokemon_list_by_type("dark")`
* *Nutzer:* "Was kostet ein **Hypertrank**?" -> *Tool:* `get_item_info("hyper-potion")`

**Wichtige Übersetzungs-Hilfe:**
* **Typen:** Feuer=Fire, Flug=Flying, Pflanze=Grass, Unlicht=Dark, Fee=Fairy, Kampf=Fighting, Psycho=Psychic, Geist=Ghost, Drache=Dragon, Käfer=Bug, Gestein=Rock, Boden=Ground, Stahl=Steel, Eis=Ice.
* **Wesen:** Hart=Adamant, Froh=Jolly, Mäßig=Modest, Scheu=Timid, Kühn=Bold, Pfiffig=Impish, Still=Calm, Sacht=Careful.
* **Items:** Überreste=Leftovers, Wahlband=Choice Band, Leben-Orb=Life Orb, Hypertrank=Hyper Potion.

### Schritt B: Output (Nach dem Tool-Aufruf)
Die API liefert englisches JSON zurück. Übersetze die relevanten Felder im Antwortsatz zurück ins Deutsche.
* API: `{"type": "fire", "weak_against": ["water", "rock"]}`
* Antwort: "Es ist vom Typ **Feuer** und ist schwach gegen **Wasser** und **Gestein**."
* API Location: `viridian-forest` -> Antwort: "Vertania Wald".
* API Item: `{"cost": 200, "effect": "Heals 50 HP"}` -> Antwort: "Es kostet 200 Pokédollar und heilt 50 KP."

## 4. Strategie für komplexe Fragen (Chain of Thought)
Wenn eine Antwort mehrere Schritte erfordert, plane selbstständig. Folge den Referenzen des Tool Outputs.

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
**Beginne nun die Interaktion.**"""


# Tool definitions matching the class methods
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_pokemon_details",
            "description": "Ruft technische Daten für ein Pokémon ab: Basiswerte (Angriff, Init), Typen, Größe, Gewicht und Fähigkeiten. Nutze dies für allgemeine Stats-Fragen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Pokémon (z.B. 'charizard' für Glurak). Das LLM muss den deutschen Namen intern übersetzen.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_species_info",
            "description": "Ruft Hintergrundinformationen ab: Pokédex-Einträge, Fangrate, ob es legendär ist, und die 'evolution_chain_url'. Nutze dies für Fragen zur Entwicklung oder zum Verhalten.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Pokémon.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evolution_chain",
            "description": "Ruft den kompletten Entwicklungsbaum ab. Benötigt eine 'chain_id' (Integer), die man vorher über 'get_species_info' herausfindet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain_id": {
                        "type": "integer",
                        "description": "Die numerische ID aus der evolution_chain URL.",
                    }
                },
                "required": ["chain_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_move_details",
            "description": "Ruft Daten zu einer Attacke ab: Stärke, Genauigkeit, AP und Schadensklasse (physisch/spezial).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name der Attacke (z.B. 'fireball').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_type_info",
            "description": "Ruft Typen-Effektivität ab. Gibt Listen zurück, wogegen dieser Typ schwach oder stark ist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Elementar-Typs (z.B. 'electric').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_encounters",
            "description": "Findet Orte (Routen, Höhlen, Gebiete), an denen man ein bestimmtes Pokémon wild fangen kann.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Pokémon (z.B. 'pikachu').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nature_info",
            "description": "Ruft Details zu einem Wesen (Nature) ab. Zeigt, welcher Statuswert erhöht und welcher gesenkt wird. Wichtig für strategische Fragen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Wesens (z.B. 'adamant' für Hart, 'jolly' für Froh).",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pokemon_list_by_type",
            "description": "Gibt eine Liste von Pokémon zurück, die einen bestimmten Elementar-Typ haben. Nutze dies, wenn der Nutzer nach Beispielen für einen Typ fragt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_name": {
                        "type": "string",
                        "description": "Der englische Name des Typs (z.B. 'fire', 'dragon').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Wie viele Beispiele zurückgegeben werden sollen (Standard: 10).",
                    },
                },
                "required": ["type_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ability_details",
            "description": "Erklärt genau, was eine passive Fähigkeit (Ability) im Kampf bewirkt und welche Pokémon sie haben können.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name der Fähigkeit (z.B. 'static', 'levitate').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_info",
            "description": "Ruft Details zu einem Item ab (Kosten, Effekt, Attribute, Fling-Power).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Items (z.B. 'leftovers' für Überreste).",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evolution_trigger_info",
            "description": "Gibt Infos zurück, welche Pokémon durch einen bestimmten Auslöser entwickelt werden (z.B. 'trade').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Name des Triggers (z.B. 'level-up', 'trade', 'use-item').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_category_info",
            "description": "Listet Items in einer bestimmten Kategorie auf (z.B. 'standard-balls', 'healing').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Kategoriename.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_attribute_info",
            "description": "Listet Items mit einem bestimmten Attribut auf (z.B. 'consumable', 'holdable').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Der englische Attributname.",
                    }
                },
                "required": ["name"],
            },
        },
    },
]
