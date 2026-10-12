"""The words the library tags photos with, and the everyday words people search with.

Auto-tagging scores each photo against every tag (a vision model compares the photo with a
short description of each). Search maps the words in a query onto these tags, through the
synonyms below, so "puppy" finds photos tagged "dog".
"""

from __future__ import annotations

import re

TAGS: dict[str, str] = {
    # People
    "people": "a photo of people",
    "portrait": "a portrait photo of a person",
    "group": "a group photo of several people",
    "child": "a photo of a child",
    "baby": "a photo of a baby",
    "selfie": "a selfie",
    # Animals
    "dog": "a photo of a dog",
    "cat": "a photo of a cat",
    "bird": "a photo of a bird",
    "horse": "a photo of a horse",
    "wildlife": "a photo of wild animals",
    # Places and nature
    "beach": "a photo of a beach",
    "ocean": "a photo of the ocean",
    "lake": "a photo of a lake",
    "river": "a photo of a river",
    "waterfall": "a photo of a waterfall",
    "mountains": "a photo of mountains",
    "forest": "a photo of a forest",
    "nature": "a photo of nature and greenery",
    "flowers": "a photo of flowers",
    "garden": "a photo of a garden",
    "desert": "a photo of a desert",
    "snow": "a photo of snow",
    "sky": "a photo of the sky",
    "sunset": "a photo of a sunset",
    "night": "a photo taken at night",
    "city": "a photo of a city",
    "street": "a photo of a street",
    "architecture": "a photo of architecture and buildings",
    "landmark": "a photo of a famous landmark",
    "indoors": "a photo taken indoors",
    "home": "a photo inside a home",
    # Things and occasions
    "food": "a photo of food",
    "drinks": "a photo of drinks",
    "car": "a photo of a car",
    "bicycle": "a photo of a bicycle",
    "boat": "a photo of a boat",
    "airplane": "a photo of an airplane",
    "train": "a photo of a train",
    "party": "a photo of a party",
    "wedding": "a photo of a wedding",
    "concert": "a photo of a concert",
    "sports": "a photo of people playing sports",
    "hiking": "a photo of people hiking",
    "document": "a photo of a document or text",
    "screenshot": "a screenshot",
    # Looks
    "black and white": "a black and white photo",
    "colorful": "a very colorful photo",
}

SYNONYMS: dict[str, str] = {
    "person": "people",
    "persons": "people",
    "friends": "people",
    "family": "people",
    "faces": "people",
    "face": "portrait",
    "portraits": "portrait",
    "headshot": "portrait",
    "kid": "child",
    "kids": "child",
    "children": "child",
    "toddler": "child",
    "babies": "baby",
    "puppy": "dog",
    "puppies": "dog",
    "dogs": "dog",
    "kitten": "cat",
    "cats": "cat",
    "birds": "bird",
    "animals": "wildlife",
    "animal": "wildlife",
    "beaches": "beach",
    "seaside": "beach",
    "coast": "beach",
    "shore": "beach",
    "sea": "ocean",
    "waves": "ocean",
    "mountain": "mountains",
    "hills": "mountains",
    "woods": "forest",
    "trees": "forest",
    "flower": "flowers",
    "sunsets": "sunset",
    "sunrise": "sunset",
    "dusk": "sunset",
    "evening": "sunset",
    "nighttime": "night",
    "stars": "night",
    "winter": "snow",
    "snowy": "snow",
    "skiing": "snow",
    "cities": "city",
    "downtown": "city",
    "buildings": "architecture",
    "building": "architecture",
    "inside": "indoors",
    "indoor": "indoors",
    "meal": "food",
    "dinner": "food",
    "lunch": "food",
    "breakfast": "food",
    "cars": "car",
    "bike": "bicycle",
    "bikes": "bicycle",
    "boats": "boat",
    "plane": "airplane",
    "planes": "airplane",
    "trains": "train",
    "birthday": "party",
    "hike": "hiking",
    "monochrome": "black and white",
    "b&w": "black and white",
    "bw": "black and white",
    "receipt": "document",
    "receipts": "document",
}


def tags_in(text: str) -> tuple[list[str], str]:
    """The tags a piece of text mentions, and what is left of it once they are taken out.

    Multi-word tags ("black and white") are found before single words.
    """
    rest = f" {text.lower()} "
    found: list[str] = []
    phrases = sorted([*TAGS, *SYNONYMS], key=lambda p: -len(p))
    for phrase in phrases:
        pattern = rf"(?<![\w&]){re.escape(phrase)}(?![\w&])"
        if re.search(pattern, rest):
            tag = SYNONYMS.get(phrase, phrase)
            if tag not in found:
                found.append(tag)
            rest = re.sub(pattern, " ", rest)
    return found, " ".join(rest.split())
