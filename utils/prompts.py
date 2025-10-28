import random
from typing import Iterable, List, Optional


PROMPTS: List[str] = [
    "A raven arrives with a sealed message from King's Landing. What does it say?",
    "You stand upon the battlements of Dragonstone, storm winds howling. Who joins you?",
    "A feast at the Red Keep turns tense when a toast cuts too deep. How do you respond?",
    "Your dragon stirs at midnight, restless and hungry. Where do you fly?",
    "The Small Council seeks your counsel on succession. What path do you argue for?",
    "Two knights cross blades in the training yard. You place a bet. On whom and why?",
    "Rumors whisper that a Valyrian steel blade changed hands. Do you pursue it?",
    "House alliances shift like sand. Who do you court as an ally, and at what cost?",
    "A secret door opens behind a tapestry in the Red Keep. What lies beyond?",
    "You find a torn page from a maester's ledger, inked with a single name. Whose?",
    "Dawn breaks over Driftmark's harbors. What deal do you strike with House Velaryon?",
    "A duel by moonlight has been declared. What are the stakes?",
    "A masked revel at the court hides daggers beneath silk. Who do you watch?",
    "The dragonkeepers request aid with a clutch of eggs. What do you decide?",
    "A song is sung of your deeds—embellished, of course. Which verse is true?",
    "The Sept's bells toll thrice at an odd hour. What omen do you take?",
    "You’re summoned to sit the Iron Throne while the King rests. What decree is made?",
    "A ship from Braavos brings a ledger of debts past due. How do you pay?",
    "Your bannermen grumble about harvest tithes. What bargain eases their worry?",
    "A dragon's shadow crosses the tourney grounds. What spectacle follows?",
    "An ancient Valyrian scroll hints at lost rites of bonding. Do you attempt them?",
    "A blood-soaked cloak is found at the foot of the Tower of the Hand. Whose?",
    "You awaken with an unfamiliar ring on your finger. Who placed it there?",
    "A quiet garden meeting under torchlight escalates into a conspiracy. Your move?",
    "A wildfire cache is rumored beneath the city. Do you hunt or hush it?",
    "A lordling insults your house sigil. Do you laugh, duel, or scheme?",
    "Your dragon refuses your command mid-flight. What changed?",
    "You receive a marriage proposal with impossible terms. Which do you negotiate?",
    "A masked singer wins the court. What secret do they hide in lyrics?",
    "The painted table at Dragonstone reveals a hidden mark. Where does it point?",
    "A maester warns of omens in comets and crows. Do you heed him?",
    "A knight in green seeks hospitality at your hall. Friend or foe?",
    "A long-lost cousin of your house returns with a claim. How do you test it?",
    "The King names you Master of Laws for a day. Which law do you change?",
    "Your dragon circles over the Gods Eye. What brings you to its shores?",
    "A riderless saddle is found by the Dragonpit. Who fell, and why?",
    "A sellsail captain offers secrets for a steep price. What do you trade?",
    "A quiet cup of arbor gold turns to poison. Who was the true target?",
    "An anonymous note reads: ""Blood demands blood."" What past debt is called in?",
    "At the tourney, a mystery knight bears your colors. Who are they?",
]


TOPIC_KEYWORDS = {
    "dragon": ["dragon", "Dragonstone", "dragonkeepers", "eggs"],
    "court": ["court", "King's Landing", "Red Keep", "Small Council"],
    "duel": ["duel", "tourney", "knight", "blades"],
    "intrigue": ["secret", "masked", "conspiracy", "poison"],
    "sea": ["ship", "Driftmark", "harbor", "Braavos"],
}


def get_random_prompt(topic: Optional[str] = None) -> str:
    if not topic:
        return random.choice(PROMPTS)

    keywords: Iterable[str] = TOPIC_KEYWORDS.get(topic.lower(), [])
    if not keywords:
        return random.choice(PROMPTS)

    filtered = [p for p in PROMPTS if any(k.lower() in p.lower() for k in keywords)]
    return random.choice(filtered) if filtered else random.choice(PROMPTS)


def get_topic_choices() -> List[str]:
    return list(TOPIC_KEYWORDS.keys())

