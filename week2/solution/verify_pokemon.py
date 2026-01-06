from pokemon.unified_pokemon_client import PokemonAPIClient
import json

client = PokemonAPIClient()
details = client.get_pokemon_details("lucario")
print(json.dumps(details, indent=2))
