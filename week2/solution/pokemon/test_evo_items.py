from unified_pokemon_client import PokemonAPIClient
import json


def test_items_and_evos():
    client = PokemonAPIClient()

    print("--- Testing get_evolution_trigger_info ---")
    res = client.get_evolution_trigger_info("trade")
    print(json.dumps(res, indent=2))
    assert "machamp" in res.get("pokemon_species", []), "Machamp should evolve by trade"

    print("\n--- Testing get_item_category_info ---")
    res = client.get_item_category_info("standard-balls")
    print(json.dumps(res, indent=2))
    assert "poke-ball" in res.get("items", []), "Poke Ball should be in standard-balls"

    print("\n--- Testing get_item_attribute_info ---")
    res = client.get_item_attribute_info("consumable")
    print(json.dumps(res, indent=2))
    assert "potion" in res.get("items", []), "Potion should be consumable"

    print("\n--- Testing get_item_info (Refined) ---")
    res = client.get_item_info("leftovers")
    print(json.dumps(res, indent=2))
    assert "attributes" in res
    assert "holdable" in res["attributes"]

    print("\n--- Testing get_pokemon_details (Refined) ---")
    res = client.get_pokemon_details("pikachu")
    print(json.dumps(res, indent=2))
    assert "sprites" in res
    assert "front_default" in res["sprites"]
    assert "base_experience" in res

    print("\n--- Testing get_species_info (Refined) ---")
    res = client.get_species_info("pikachu")
    print(json.dumps(res, indent=2))
    assert "genus" in res
    assert "generation" in res

    print("\n--- Testing get_move_details (Refined) ---")
    res = client.get_move_details("quick-attack")
    print(json.dumps(res, indent=2))
    assert "priority" in res
    assert res["priority"] > 0

    print("\n--- Testing get_evolution_chain (Deep Parsing) ---")
    # Eevee chain (chain id 67) has complex triggers
    # First get species to find chain ID
    species = client.get_species_info("eevee")
    chain_url = species["evolution_chain_url"]
    chain_id = int(chain_url.split("/")[-2])

    res = client.get_evolution_chain(chain_id)
    print(json.dumps(res, indent=2))

    # Verify deeply nested structure
    evolves_to = res["chain"]["evolves_to"]
    vaporeon = next((x for x in evolves_to if x["species_name"] == "vaporeon"), None)
    assert vaporeon is not None
    assert vaporeon["evolution_details"][0]["item"] == "water-stone"

    umbreon = next((x for x in evolves_to if x["species_name"] == "umbreon"), None)
    assert umbreon is not None
    assert umbreon["evolution_details"][0]["time_of_day"] == "night"

    print("\nSUCCESS: All tests passed!")


if __name__ == "__main__":
    test_items_and_evos()
