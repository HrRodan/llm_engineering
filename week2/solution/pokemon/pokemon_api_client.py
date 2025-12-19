"""
Generic PokéAPI v2 client with local caching.

- No dependency on any LLM or provider SDK
- Pure HTTP via `requests`
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests


class PokemonAPIClient:
    """
    Generic client for PokéAPI v2 with built-in local caching.

    Fair Use Policy:
    - Use local caching to avoid unnecessary requests.
    - Do not spam the API with high-frequency polling.
    - Handle errors gracefully.
    """

    BASE_URL = "https://pokeapi.co/api/v2"

    # Resource endpoints that support ID/name lookup
    NAMED_RESOURCES = {
        "ability", "berry", "berry-firmness", "berry-flavor",
        "contest-type", "egg-group", "encounter-condition", "encounter-condition-value",
        "encounter-method", "evolution-trigger", "generation", "gender",
        "growth-rate", "item", "item-attribute", "item-category", "item-fling-effect",
        "item-pocket", "language", "location", "location-area", "machine",
        "move", "move-ailment", "move-battle-style", "move-category",
        "move-damage-class", "move-learn-method", "move-target",
        "nature", "pal-park-area", "pokeathlon-stat", "pokedex",
        "pokemon", "pokemon-color", "pokemon-form", "pokemon-habitat",
        "pokemon-shape", "pokemon-species", "region", "stat", "type", "version",
        "version-group",
    }

    # ID-only resources
    UNNAMED_RESOURCES = {
        "characteristic", "contest-effect", "evolution-chain", "super-contest-effect",
    }

    def __init__(self, cache_dir: str = ".pokemon_cache", cache_ttl_hours: int = 24) -> None:
        """
        Initialize the API client.

        Args:
            cache_dir: Directory for JSON cache files.
            cache_ttl_hours: Cache TTL in hours. 0 = never expires.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PokemonAPIClient/1.0 (+https://pokeapi.co/)"
        })

    # ----------------------
    # Cache helpers
    # ----------------------

    def _get_cache_path(self, endpoint: str, identifier: Optional[str] = None) -> Path:
        if identifier:
            filename = f"{endpoint}_{identifier}.json"
        else:
            filename = f"{endpoint}_list.json"
        return self.cache_dir / filename

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.cache_ttl.total_seconds() == 0:
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < self.cache_ttl

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

    # ----------------------
    # HTTP helper
    # ----------------------

    def _make_request(self, url: str) -> Dict[str, Any]:
        resp = self.session.get(url, timeout=10)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            if resp.status_code == 404:
                raise ValueError(f"Resource not found: {url}") from e
            raise ValueError(f"PokéAPI error ({resp.status_code}): {e}") from e
        except requests.RequestException as e:
            raise ValueError(f"Request failed: {e}") from e
        return resp.json()

    # ----------------------
    # Public API
    # ----------------------

    def query(
        self,
        endpoint: str,
        identifier: Optional[Union[str, int]] = None,
        limit: int = 20,
        offset: int = 0,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generic query for any PokéAPI endpoint.

        - If `identifier` is given → returns a single resource.
        - If `identifier` is None → returns a paginated list with `results`.

        Args:
            endpoint: e.g. "pokemon", "move", "ability", "type", "item", "evolution-chain", ...
            identifier: ID or name. Examples: "pikachu", "25", "thunderbolt".
            limit: page size for list queries (ignored when identifier is set).
            offset: pagination offset for list queries.
            use_cache: whether to use local cache for single-resource queries.

        Returns:
            Response JSON as a dictionary.

        Raises:
            ValueError: if endpoint is invalid or resource is not found.
        """
        valid = self.NAMED_RESOURCES | self.UNNAMED_RESOURCES
        if endpoint not in valid:
            raise ValueError(
                f"Invalid endpoint '{endpoint}'. "
                f"Valid endpoints: {', '.join(sorted(valid))}"
            )

        if identifier is not None:
            url = f"{self.BASE_URL}/{endpoint}/{identifier}/"
            cache_path = self._get_cache_path(endpoint, str(identifier))
        else:
            url = f"{self.BASE_URL}/{endpoint}/?limit={limit}&offset={offset}"
            cache_path = self._get_cache_path(endpoint)

        # cache only for single-resource queries
        if use_cache and identifier is not None:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                cached["_from_cache"] = True
                return cached

        data = self._make_request(url)
        data["_from_cache"] = False

        if use_cache and identifier is not None:
            self._save_to_cache(cache_path, data)

        return data

    def query_nested(
        self,
        endpoint: str,
        identifier: Union[str, int],
        nested_endpoint: str,
    ) -> Dict[str, Any]:
        """
        Query nested resources for a given resource.

        Examples:
            - /pokemon/{id}/encounters
            - /location-area/{id}/pokemon-encounters

        Args:
            endpoint: parent endpoint, e.g. "pokemon" or "location-area".
            identifier: parent ID or name.
            nested_endpoint: nested path, e.g. "encounters" or "pokemon-encounters".
        """
        url = f"{self.BASE_URL}/{endpoint}/{identifier}/{nested_endpoint}/"
        data = self._make_request(url)
        data["_from_cache"] = False
        return data

    def get_type_coverage(self, type_name: str) -> Dict[str, List[str]]:
        """
        Get type effectiveness information.

        Returns dictionary with:
        - strong_against: types that take double damage from this type
        - weak_to: types that deal double damage to this type
        - resists: types that deal half damage to this type
        - not_effective_against: types that take no damage from this type
        """
        type_data = self.query("type", type_name)
        rel = type_data.get("damage_relations", {})
        return {
            "strong_against": [t["name"] for t in rel.get("double_damage_to", [])],
            "weak_to": [t["name"] for t in rel.get("double_damage_from", [])],
            "resists": [t["name"] for t in rel.get("half_damage_from", [])],
            "not_effective_against": [t["name"] for t in rel.get("no_damage_to", [])],
        }

    def get_move_details(self, move_name: str) -> Dict[str, Any]:
        """
        Get high-level details for a move: power, accuracy, type, etc.
        """
        move = self.query("move", move_name)
        effect_entry = next(
            (
                e
                for e in move.get("effect_entries", [])
                if e.get("language", {}).get("name") == "en"
            ),
            {},
        )
        return {
            "name": move.get("name"),
            "power": move.get("power"),
            "accuracy": move.get("accuracy"),
            "pp": move.get("pp"),
            "type": move.get("type", {}).get("name"),
            "damage_class": move.get("damage_class", {}).get("name"),
            "effect": effect_entry.get("short_effect"),
            "priority": move.get("priority"),
        }

    def get_pokemon_complete(self, identifier: Union[str, int]) -> Dict[str, Any]:
        """
        Return combined Pokémon data with species information.
        """
        pokemon = self.query("pokemon", identifier)
        species_name = pokemon.get("species", {}).get("name")
        if species_name:
            try:
                species = self.query("pokemon-species", species_name)
                pokemon["_species"] = species
            except ValueError:
                pass
        return pokemon

    def clear_cache(self) -> None:
        """Delete all cache files."""
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pokemon_api_query",
            "description": (
                "Low-level access to the PokéAPI v2.\n\n"
                "Use this function whenever you need raw data from PokéAPI. "
                "It can retrieve any supported resource by endpoint name, "
                "optionally filtered by ID or name.\n\n"
                "Examples of valid endpoints: "
                "pokemon, pokemon-species, move, ability, type, item, evolution-chain, "
                "location, region, pokedex, generation, stat, nature, berry, machine ..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": (
                            "PokéAPI resource name. "
                            "Typical values include: "
                            "'pokemon', 'pokemon-species', 'ability', 'move', 'type', "
                            "'item', 'evolution-chain', 'nature', 'growth-rate', "
                            "'egg-group', 'stat', 'pokemon-form', 'pokemon-habitat', "
                            "'pokemon-shape', 'pokemon-color', 'gender', 'berry', "
                            "'berry-flavor', 'contest-type', 'encounter-method', "
                            "'location', 'location-area', 'region', 'pokedex', "
                            "'generation', 'version', 'version-group', 'language'."
                        ),
                    },
                    "identifier": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional ID or name of the resource.\n"
                            "- If provided, returns a single resource, e.g. "
                            "endpoint='pokemon', identifier='pikachu'.\n"
                            "- If omitted or null, returns a paginated list "
                            "with 'results', 'next', 'previous', 'count'."
                        ),
                        "default": None,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of items to return when listing resources. "
                            "Ignored if 'identifier' is set."
                        ),
                        "default": 20,
                        "minimum": 1,
                        "maximum": 1000,
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Pagination offset when listing resources (start index). "
                            "Ignored if 'identifier' is set."
                        ),
                        "default": 0,
                        "minimum": 0,
                    },
                },
                "required": ["endpoint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pokemon_get_type_coverage",
            "description": (
                "Returns how effective a Pokémon type is in battles.\n\n"
                "Use this function when the user asks questions like:\n"
                "- 'What is Electric super effective against?'\n"
                "- 'Which types beat Fire?'\n"
                "- 'What does Dragon resist?'\n\n"
                "The result contains four lists: 'strong_against', 'weak_to', "
                "'resists', and 'not_effective_against'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type_name": {
                        "type": "string",
                        "description": (
                            "Name of the Pokémon type in lowercase.\n"
                            "Examples: 'fire', 'water', 'electric', 'grass', "
                            "'ice', 'fighting', 'poison', 'ground', 'flying', "
                            "'psychic', 'bug', 'rock', 'ghost', 'dragon', "
                            "'dark', 'steel', 'fairy'."
                        ),
                    }
                },
                "required": ["type_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pokemon_get_move_details",
            "description": (
                "Returns high-level information about a Pokémon move, including "
                "its power, accuracy, type and a short effect description.\n\n"
                "Use this when the user asks about specific moves, move power, "
                "or how a move works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "move_name": {
                        "type": "string",
                        "description": (
                            "Move name in lowercase, with hyphens where applicable.\n"
                            "Examples: 'thunderbolt', 'earthquake', 'dragon-dance', "
                            "'swords-dance', 'solar-beam', 'ice-beam'."
                        ),
                    }
                },
                "required": ["move_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pokemon_get_pokemon_complete",
            "description": (
                "Returns a combined view of a Pokémon, including base data and species data.\n\n"
                "Use this when you need a rich representation with:\n"
                "- stats (hp, attack, defense, etc.)\n"
                "- types\n"
                "- abilities\n"
                "- moves\n"
                "- sprites\n"
                "- species information (e.g. flavor text, egg groups) in `_species`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "Pokémon identifier: either numeric ID or name.\n"
                            "Examples: '25', 'pikachu', '6', 'charizard'."
                        ),
                    }
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pokemon_get_encounters",
            "description": (
                "Returns all locations where a specific Pokémon can be encountered.\n\n"
                "Use this when the user asks questions such as:\n"
                "- 'Where can I find Pikachu?'\n"
                "- 'In which locations does Bulbasaur appear?'\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pokemon_name": {
                        "type": "string",
                        "description": (
                            "Pokémon name in lowercase. Example: 'pikachu', 'bulbasaur'. "
                            "This must be a valid PokéAPI Pokémon name."
                        ),
                    }
                },
                "required": ["pokemon_name"],
            },
        },
    },
]