from week2.solution.pokemon.unified_pokemon_client import UnifiedPokemonClient


def test_get_item_info():
    client = UnifiedPokemonClient()

    # Test with a known item
    item_name = "potion"
    info = client.get_item_info(item_name)

    print(f"Info for {item_name}:")
    print(info)

    assert "name" in info
    assert info["name"] == "potion"
    assert "cost" in info
    assert "category" in info
    assert "effect" in info

    # Test with an invalid item
    invalid_item = "invalid-item-name-123"
    error_info = client.get_item_info(invalid_item)
    print(f"\nInfo for {invalid_item}:")
    print(error_info)
    assert "error" in error_info


if __name__ == "__main__":
    test_get_item_info()
