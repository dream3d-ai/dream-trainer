import random


def generate_friendly_name() -> str:
    """
    Generate a friendly, human-readable name for a job.

    This function creates a name by combining a random adjective, a random animal noun,
    and a random 4-digit number. The name format is 'adjective-noun-number'.

    Returns:
        str: A friendly name string in the format 'adjective-noun-number'.

    Example:
        >>> generate_friendly_name()
        'happy-panda-3721'
    """
    adjectives = [
        "happy",
        "sunny",
        "clever",
        "swift",
        "brave",
        "bright",
        "calm",
        "daring",
        "eager",
        "gentle",
        "jolly",
        "kind",
        "lively",
        "nice",
        "proud",
        "wise",
        "agile",
        "bold",
        "cheerful",
        "diligent",
        "elegant",
        "friendly",
        "graceful",
        "honest",
        "innovative",
        "joyful",
        "keen",
        "loyal",
        "merry",
        "noble",
        "optimistic",
        "patient",
        "quick",
        "resilient",
        "sincere",
        "thoughtful",
        "unique",
        "vibrant",
        "witty",
        "zealous",
    ]
    nouns = [
        "panda",
        "tiger",
        "eagle",
        "dolphin",
        "fox",
        "owl",
        "wolf",
        "bear",
        "hawk",
        "lion",
        "deer",
        "rabbit",
        "otter",
        "koala",
        "lynx",
        "raven",
        "elephant",
        "giraffe",
        "kangaroo",
        "penguin",
        "cheetah",
        "gorilla",
        "leopard",
        "octopus",
        "rhino",
        "squirrel",
        "turtle",
        "whale",
        "zebra",
        "alpaca",
        "bison",
        "camel",
        "flamingo",
        "hedgehog",
        "iguana",
        "jaguar",
        "lemur",
        "meerkat",
        "narwhal",
        "ocelot",
        "platypus",
        "quokka",
        "zircon",
        "abalone",
        "abyss",
        "acacia",
        
    ]
    system_random = random.SystemRandom()
    return f"{system_random.choice(adjectives)}-{system_random.choice(nouns)}-{system_random.randint(1000, 9999)}"
